"""The dimod sampler."""

import itertools

import numpy as np
import pytest

dimod = pytest.importorskip("dimod")

import mfaoa
from mfaoa.dimod import MFAOASampler, ising_problem


def random_bqm(seed, n, vartype, labels=None):
    """A random BQM with every pair coupled, so that it is not flip-symmetric."""
    rng = np.random.default_rng(seed)
    labels = list(range(n)) if labels is None else labels
    linear = {v: rng.standard_normal() for v in labels}
    quadratic = {(u, v): rng.standard_normal() for u, v in itertools.combinations(labels, 2)}
    return dimod.BinaryQuadraticModel(linear, quadratic, rng.standard_normal(), vartype)


def test_follows_the_sampler_api():
    """dimod's own check of the sampler interface passes."""
    dimod.testing.assert_sampler_api(MFAOASampler())


@pytest.mark.parametrize("vartype", [dimod.SPIN, dimod.BINARY])
def test_sample_has_the_bqms_vartype_labels_and_energy(vartype):
    """The sample uses the BQM's vartype and labels, and its energy is the BQM's."""
    bqm = random_bqm(1, 6, vartype, labels=["a", "b", "c", ("d", 1), 7, "f"])
    sampleset = MFAOASampler().sample(bqm, p=200)
    dimod.testing.assert_sampleset_energies(sampleset, bqm)
    assert sampleset.vartype is vartype
    assert set(sampleset.variables) == set(bqm.variables)
    assert len(sampleset) == 1
    assert sampleset.info["steps"] == 200


@pytest.mark.parametrize("vartype", [dimod.SPIN, dimod.BINARY])
def test_small_models_against_the_exact_solver(vartype):
    """On 8 variables, the energy is close to the best that ExactSolver finds."""
    for seed in range(5):
        bqm = random_bqm(seed, 8, vartype)
        best = dimod.ExactSolver().sample(bqm).first.energy
        worst = -dimod.ExactSolver().sample(-1 * bqm).first.energy
        energy = MFAOASampler().sample(bqm).first.energy
        # a good solution: much closer to the best energy than to the worst
        assert energy - best < 0.1 * (worst - best)


def test_the_ising_problem_has_the_bqms_energies():
    """The Ising problem built from a BINARY BQM has the BQM's energy everywhere."""
    bqm = random_bqm(2, 5, dimod.BINARY)
    problem, labels = ising_problem(bqm)
    for spins in itertools.product([-1, 1], repeat=5):
        sample = {v: (s + 1) // 2 for v, s in zip(labels, spins)}
        assert problem.objective(np.array(spins)) == pytest.approx(bqm.energy(sample))


def test_same_result_as_solve():
    """The sampler returns what solve returns on the same problem."""
    bqm = random_bqm(3, 10, dimod.SPIN)
    problem, labels = ising_problem(bqm)
    sigma = mfaoa.solve(problem, p=100).sigma
    sample = MFAOASampler().sample(bqm, p=100).first.sample
    assert [sample[v] for v in labels] == sigma.tolist()


def test_flip_symmetric_models_name_their_labels():
    """A flip-symmetric BQM is rejected with its labels, and solvable once a variable is fixed."""
    bqm = dimod.BinaryQuadraticModel({}, {("a", "b"): 1.0, ("b", "c"): -1.0}, 0.0, dimod.SPIN)
    with pytest.raises(mfaoa.FlipSymmetricError) as error:
        MFAOASampler().sample(bqm)
    assert error.value.groups == [["a", "b", "c"]]
    assert "(σ → -σ)" in str(error.value)
    assert "Fix one variable to -1 or +1 with bqm.fix_variable, for example 'a'." in str(error.value)

    bqm.fix_variable("a", 1)
    sample = MFAOASampler().sample(bqm).first
    assert sample.energy == -2


def test_binary_messages_speak_of_x():
    """For a BINARY BQM the message describes the flip as x → 1 - x."""
    bqm = dimod.BinaryQuadraticModel({0: 1.0, 1: 1.0}, {(0, 1): -2.0}, 0.0, dimod.BINARY)
    with pytest.raises(mfaoa.FlipSymmetricError, match=r"\(x → 1 - x\).*0 or 1 with bqm.fix_variable, for example 0\."):
        MFAOASampler().sample(bqm)


def test_options_and_an_empty_model():
    """Schedules, unknown keywords and an empty BQM."""
    bqm = random_bqm(4, 4, dimod.SPIN)
    schedule = mfaoa.linear_schedule(20)
    sampleset = MFAOASampler().sample(bqm, beta=schedule.beta, gamma=schedule.gamma)
    assert sampleset.info["steps"] == 20
    with pytest.raises(ValueError, match="both beta and gamma"):
        MFAOASampler().sample(bqm, beta=schedule.beta)
    with pytest.warns(dimod.exceptions.SamplerUnknownArgWarning):
        MFAOASampler().sample(bqm, num_reads=10)
    assert len(MFAOASampler().sample(dimod.BinaryQuadraticModel(dimod.SPIN))) == 0
