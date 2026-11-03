"""Building problems and schedules: what is accepted, stored and rejected."""

import mfaoa
import numpy as np
import pytest
import scipy.sparse as sp

J2 = np.array([[0.0, 1.0], [1.0, 0.0]])


def test_float64_arrays_are_kept_without_copy():
    """float64 arrays are stored as given, not copied."""
    h = np.array([0.5, -0.25])
    problem = mfaoa.IsingProblem(h, J2)
    assert problem.h is h and problem.J is J2


def test_fieldscale_is_the_rms_field():
    """fieldscale is sqrt((Σh² + ΣJ²) / n), 1 for a zero objective, or the scale given."""
    problem = mfaoa.IsingProblem([3.0, 0.0], J2)
    assert problem.fieldscale == pytest.approx(np.sqrt((9 + 2) / 2))
    assert mfaoa.IsingProblem([0.0, 0.0], np.zeros((2, 2))).fieldscale == 1.0
    assert mfaoa.IsingProblem([3.0, 0.0], J2, scale=2).fieldscale == 2.0


def test_scaling_the_objective_does_not_change_the_solution():
    """Multiplying h and J by 1000 leaves the solution unchanged."""
    rng = np.random.default_rng(7)
    h = rng.standard_normal(20)
    A = np.triu(rng.standard_normal((20, 20)), 1)
    J = A + A.T
    a = mfaoa.solve(mfaoa.IsingProblem(h, J), p=100)
    b = mfaoa.solve(mfaoa.IsingProblem(1000 * h, 1000 * J), p=100)
    np.testing.assert_array_equal(a.sigma, b.sigma)


def test_sparse_input_in_any_format_becomes_canonical_csc():
    """Sparse input in COO form, with duplicates, becomes canonical CSC."""
    rows, cols, vals = [0, 1, 0, 1], [1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5]  # duplicates add up
    J = sp.coo_matrix((vals, (rows, cols)), shape=(2, 2))
    problem = mfaoa.IsingProblem([0.1, 0.0], J)
    assert problem.J.format == "csc" and problem.J.has_canonical_format
    assert problem.J[0, 1] == 1.0
    assert problem._colptr.dtype == np.int64


def test_objective_includes_offset_and_counts_each_pair_once():
    """objective counts each pair i < j once and adds the offset."""
    problem = mfaoa.IsingProblem([1.0, 0.0], J2, offset=10)
    assert problem.objective([1, 1]) == 1 + 1 + 10
    assert problem.objective([-1, 1]) == -1 - 1 + 10


@pytest.mark.parametrize(
    "build, message",
    [
        (lambda: mfaoa.IsingProblem([0.0, 0.0], J2, sense="minimize"), "sense must be 'min' or 'max'"),
        (lambda: mfaoa.IsingProblem([], np.zeros((0, 0))), "at least one variable"),
        (lambda: mfaoa.IsingProblem([[1.0]], J2), "h must be a vector"),
        (lambda: mfaoa.IsingProblem([0.0, 0.0, 0.0], J2), "J must be 3×3 to match h"),
        (lambda: mfaoa.IsingProblem([0.0, 0.0], [[0.0, 1.0], [2.0, 0.0]]),
         r"J must be symmetric, but J\[0, 1\] = 1.0 and J\[1, 0\] = 2.0"),
        (lambda: mfaoa.IsingProblem([0.0, 0.0], sp.csc_array([[0.0, 1.0], [2.0, 0.0]])),
         "J must be symmetric"),
        (lambda: mfaoa.IsingProblem([0.0, 0.0], [[0.0, 1.0], [1.0, 0.5]]),
         r"J must be zero on the diagonal, but J\[1, 1\] = 0.5.*add it to offset"),
        (lambda: mfaoa.IsingProblem([0.0, 0.0], J2, delta=[1.0, 0.0]), "must be positive"),
        (lambda: mfaoa.IsingProblem([0.0, 0.0], J2, delta=[1.0]), "delta must have one entry per variable"),
        (lambda: mfaoa.IsingProblem([0.0, 0.0], J2, scale=0), "scale must be positive"),
        (lambda: mfaoa.QUBOProblem(np.zeros((2, 3))), "Q must be square"),
        (lambda: mfaoa.QUBOProblem(J2, c=[1.0]), "c must have one entry per variable"),
        (lambda: mfaoa.Schedule([0.1, 0.2], [0.1]), "one entry per step; got 2 and 1"),
        (lambda: mfaoa.Schedule([], []), "at least one step"),
        (lambda: mfaoa.linear_schedule(0), "need at least one step, got p = 0"),
    ],
)
def test_invalid_input_is_rejected_with_a_message(build, message):
    """Each kind of invalid input raises a ValueError that says what is wrong."""
    with pytest.raises(ValueError, match=message):
        build()


def test_solve_rejects_wrong_arguments():
    """solve rejects a schedule together with p, and objects that are not problems."""
    problem = mfaoa.IsingProblem([0.5, 0.0], J2)
    with pytest.raises(TypeError, match="either a schedule or p and tau"):
        mfaoa.solve(problem, mfaoa.linear_schedule(10), p=10)
    with pytest.raises(TypeError, match="IsingProblem or a QUBOProblem"):
        mfaoa.solve(J2)
    with pytest.raises(TypeError):
        mfaoa.linear_schedule(10.5)


def test_p_and_tau_select_the_linear_schedule():
    """p and tau give the same run as the linear schedule they describe."""
    problem = mfaoa.IsingProblem([0.5, -0.3], J2)
    a = mfaoa.solve(problem, p=30, tau=0.3)
    b = mfaoa.solve(problem, mfaoa.linear_schedule(30, 0.3))
    np.testing.assert_array_equal(a.state.z, b.state.z)
