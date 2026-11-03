"""Flip-symmetric problems: detection, the groups reported, and the messages."""

import numpy as np
import pytest
import scipy.sparse as sp

import mfaoa


def ring(n):
    J = np.zeros((n, n))
    for i in range(n):
        J[i, (i + 1) % n] = J[(i + 1) % n, i] = 1.0
    return J


def pairs(k):
    """k separate pairs of variables, each pair coupled, no fields."""
    J = np.zeros((2 * k, 2 * k))
    for g in range(k):
        J[2 * g, 2 * g + 1] = J[2 * g + 1, 2 * g] = 1.0
    return J


def test_max_cut_is_flip_symmetric_as_a_whole():
    problem = mfaoa.IsingProblem(np.zeros(6), ring(6))
    assert problem.is_flip_symmetric()
    with pytest.raises(mfaoa.FlipSymmetricError) as error:
        mfaoa.solve(problem)
    assert error.value.groups == [list(range(6))]
    assert str(error.value) == (
        "The objective does not change when every variable is flipped (σ → -σ). "
        "MFAOA starts all variables at 0, halfway between -1 and +1, where such an objective "
        "has zero slope in every variable, so no variable would ever move. "
        "Fix one variable to -1 or +1 (see help(mfaoa.IsingProblem.is_flip_symmetric) for how), "
        "for example σ[0]. "
        "This loses nothing, since flipping every variable does not change the objective value."
    )


def test_one_group_among_other_variables():
    J = np.zeros((5, 5))
    J[:3, :3] = ring(3)
    J[3, 4] = J[4, 3] = 1.0
    h = np.array([0.0, 0.0, 0.0, 0.5, 0.0])  # the pair 3-4 has a field
    with pytest.raises(mfaoa.FlipSymmetricError, match=r"the variables σ\[0\], σ\[1\] and σ\[2\] are all flipped") as error:
        mfaoa.solve(mfaoa.IsingProblem(h, sp.csc_array(J)))
    assert error.value.groups == [[0, 1, 2]]
    assert "for example σ[0]." in str(error.value)


def test_many_groups_are_listed_briefly():
    with pytest.raises(mfaoa.FlipSymmetricError) as error:
        mfaoa.solve(mfaoa.IsingProblem(np.zeros(10), pairs(5)))
    assert error.value.groups == [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    message = str(error.value)
    assert "any one of these 5 groups are flipped (σ → -σ): {σ[0], σ[1]}, {σ[2], σ[3]}, {σ[4], σ[5]} and 2 more groups." in message
    assert "for example σ[0], σ[2], σ[4], σ[6] and σ[8]." in message


def test_qubo_messages_name_x():
    # x0 + x1 - 2 x0 x1 is unchanged when both flip: x → 1 - x
    qubo = mfaoa.QUBOProblem(np.array([[0.0, -2.0], [0.0, 0.0]]), c=[1.0, 1.0])
    assert qubo.is_flip_symmetric()
    with pytest.raises(mfaoa.FlipSymmetricError, match=r"\(x → 1 - x\).*Fix one variable to 0 or 1 \(see help\(mfaoa.QUBOProblem.is_flip_symmetric\) for how\), for example x\[0\]"):
        mfaoa.solve(qubo)


def test_fixing_one_variable_makes_it_solvable():
    J = ring(6)
    # fix σ[0] = +1: remove it, and add J[j, 0] to h[j]
    problem = mfaoa.IsingProblem(J[1:, 0], np.ascontiguousarray(J[1:, 1:]), offset=0.0)
    assert not problem.is_flip_symmetric()
    result = mfaoa.solve(problem)
    assert result.objective == -6 + 0  # the even ring: every edge cut, E = -6 with σ[0] = +1


def test_a_lone_variable_is_not_a_group():
    problem = mfaoa.IsingProblem([0.0, 0.5, 0.0], np.array([[0, 0, 0], [0, 0, 1.0], [0, 1.0, 0]]))
    assert not problem.is_flip_symmetric()
    assert mfaoa.solve(problem).sigma[0] == 1
