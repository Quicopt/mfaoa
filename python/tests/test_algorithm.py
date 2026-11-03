"""The solver against a NumPy implementation of the algorithm, and the
QUBO-to-Ising conversion against brute force."""

import itertools

import mfaoa
import numpy as np
import pytest
import scipy.sparse as sp


def numpy_mfaoa(h, J, beta, gamma, *, sense=1, scale=None, delta=None):
    """The algorithm as the mfaoa.h header describes it, written directly in NumPy."""
    n = h.shape[0]
    if scale is None:
        scale = np.sqrt((h @ h + (J.multiply(J).sum() if sp.issparse(J) else (J * J).sum())) / n)
    delta = np.ones(n) if delta is None else delta
    x, y, z = np.ones(n), np.zeros(n), np.zeros(n)
    for b, g in zip(beta, gamma):
        m = h + J @ z
        a = 2 * g * sense * m / scale
        x, y = np.cos(a) * x - np.sin(a) * y, np.sin(a) * x + np.cos(a) * y
        c, s = np.cos(2 * b * delta), np.sin(2 * b * delta)
        y, z = c * y + s * z, c * z - s * y
    return x, y, z


def random_ising(rng, n, density, sparse):
    """A random h and a random symmetric J with zero diagonal, sparse or dense.

    About a fraction `density` of the pairs i < j are coupled, with weights
    uniform in [0, 1).
    """
    upper = np.triu(rng.random((n, n)) * (rng.random((n, n)) < density), k=1)
    J = upper + upper.T
    return rng.standard_normal(n), (sp.csc_array(J) if sparse else J)


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("sense", ["min", "max"])
@pytest.mark.parametrize("with_delta", [False, True])
def test_solver_matches_numpy(sparse, sense, with_delta):
    """The final vectors agree with the NumPy implementation of the algorithm."""
    rng = np.random.default_rng(1)
    n = 50
    h, J = random_ising(rng, n, 0.2, sparse)
    delta = rng.uniform(0.5, 1.5, n) if with_delta else None
    schedule = mfaoa.linear_schedule(200, 0.4)
    result = mfaoa.solve(mfaoa.IsingProblem(h, J, sense=sense, delta=delta), schedule)
    x, y, z = numpy_mfaoa(
        h, J, schedule.beta, schedule.gamma, sense=1 if sense == "min" else -1, delta=delta
    )
    np.testing.assert_allclose(result.state.x, x, atol=1e-9)
    np.testing.assert_allclose(result.state.y, y, atol=1e-9)
    np.testing.assert_allclose(result.state.z, z, atol=1e-9)
    np.testing.assert_array_equal(result.sigma, np.where(z > 0, 1, -1))


def test_objective_mf_is_the_objective_at_z():
    """objective_mf is the objective at the final z components."""
    rng = np.random.default_rng(2)
    h, J = random_ising(rng, 30, 0.3, sparse=False)
    problem = mfaoa.IsingProblem(h, J, offset=2.5)
    result = mfaoa.solve(problem, p=20)
    assert result.objective_mf == pytest.approx(problem.objective(result.state.z))
    assert result.objective == problem.objective(result.sigma)


def test_dense_and_sparse_couplings_agree():
    """The same couplings, stored dense and sparse, give the same run."""
    rng = np.random.default_rng(3)
    h, J = random_ising(rng, 40, 0.1, sparse=True)
    sparse = mfaoa.solve(mfaoa.IsingProblem(h, J))
    dense = mfaoa.solve(mfaoa.IsingProblem(h, J.toarray()))
    np.testing.assert_allclose(sparse.state.z, dense.state.z, atol=1e-9)
    np.testing.assert_array_equal(sparse.sigma, dense.sigma)


def test_fortran_ordered_couplings_give_the_same_result():
    """A symmetric J may be stored in either memory order."""
    rng = np.random.default_rng(4)
    h, J = random_ising(rng, 20, 0.5, sparse=False)
    c = mfaoa.solve(mfaoa.IsingProblem(h, np.ascontiguousarray(J)), p=50)
    f = mfaoa.solve(mfaoa.IsingProblem(h, np.asfortranarray(J)), p=50)
    np.testing.assert_array_equal(c.state.z, f.state.z)


def all_assignments(n):
    """Every vector of n zeros and ones, as the rows of an array."""
    return np.array(list(itertools.product([0, 1], repeat=n)), dtype=np.float64)


@pytest.mark.parametrize("sparse", [False, True])
def test_qubo_to_ising_preserves_every_objective_value(sparse):
    """The Ising form has the QUBO's value at every one of the 2^n points."""
    rng = np.random.default_rng(5)
    n = 8
    Q = rng.standard_normal((n, n)) * (rng.random((n, n)) < 0.5)  # not symmetric, with a diagonal
    qubo = mfaoa.QUBOProblem(sp.csr_array(Q) if sparse else Q, c=rng.standard_normal(n), constant=1.25)
    ising = qubo.to_ising()
    assert sp.issparse(ising.J) == sparse
    for x in all_assignments(n):
        assert ising.objective(2 * x - 1) == pytest.approx(qubo.objective(x))


@pytest.mark.parametrize("sense", ["min", "max"])
def test_small_qubo_solution_is_reported_consistently(sense):
    """The reported objective is the QUBO's value at x, and close to the best."""
    rng = np.random.default_rng(6)
    n = 10
    Q = np.triu(rng.standard_normal((n, n)))
    qubo = mfaoa.QUBOProblem(Q, c=rng.standard_normal(n), sense=sense)
    result = mfaoa.solve(qubo)
    assert result.objective == pytest.approx(qubo.objective(result.x))
    values = [qubo.objective(x) for x in all_assignments(n)]
    best, worst = (min(values), max(values)) if sense == "min" else (max(values), min(values))
    # a good solution: much closer to the best value than to the worst
    assert abs(result.objective - best) < 0.1 * abs(worst - best)
