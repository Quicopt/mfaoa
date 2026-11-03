"""Running the algorithm on a problem."""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from . import _lib
from ._problems import IsingProblem, QUBOProblem, Schedule, linear_schedule
from ._symmetry import ISING_WORDS, QUBO_WORDS, FlipSymmetricError, flip_groups, flip_message


class SpinState(NamedTuple):
    """The final unit vectors that represent the variables, one component per array."""

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray


@dataclass(frozen=True)
class Result:
    """The result of `solve`.

    - `sigma`: the solution in Ising form, entries -1 and +1 (int8).
    - `x`: the same solution in QUBO form, x = (1 + sigma) / 2, entries 0
      and 1 (int8).
    - `objective`: the objective value of the solution, in the problem's
      sense and including its constant.
    - `objective_mf`: the objective at the values between -1 and +1 that the
      algorithm ends with, before they are rounded to sigma. It equals
      `objective` when every value has reached -1 or +1. A large gap means
      that variables were still undecided at the end; more steps usually
      close it.
    - `state`: the final vectors that represent the variables (see
      `Schedule`).
    """

    sigma: np.ndarray
    x: np.ndarray
    objective: float
    objective_mf: float
    state: SpinState

    def __repr__(self):
        return f"Result: {self.sigma.shape[0]} variables, objective = {self.objective}"


def solve(problem, schedule=None, *, p=None, tau=None):
    """Solve an `IsingProblem` or a `QUBOProblem`.

    The algorithm runs the `Schedule` given, or else the `linear_schedule`
    with p steps (default 1000) and step size tau (default 0.5). The run is
    deterministic. Returns a `Result`.

    Raises a `FlipSymmetricError` when the problem has groups of variables
    that the algorithm cannot move (see `IsingProblem.is_flip_symmetric`);
    the message names them.
    """
    if schedule is None:
        schedule = linear_schedule(1000 if p is None else p, 0.5 if tau is None else tau)
    elif p is not None or tau is not None:
        raise TypeError("pass either a schedule or p and tau, not both")
    elif not isinstance(schedule, Schedule):
        raise TypeError(f"schedule must be a Schedule, got {type(schedule).__name__}")
    if isinstance(problem, QUBOProblem):
        ising, words, name = problem.to_ising(), QUBO_WORDS, lambda i: f"x[{i}]"
    elif isinstance(problem, IsingProblem):
        ising, words, name = problem, ISING_WORDS, lambda i: f"σ[{i}]"
    else:
        raise TypeError(f"solve takes an IsingProblem or a QUBOProblem, got {type(problem).__name__}")

    code, ws = ising._run(schedule.beta, schedule.gamma)
    if code == _lib.FLIP_SYMMETRIC:
        groups = flip_groups(ws)
        raise FlipSymmetricError(flip_message(groups, ising.n, name, words), groups)
    if code != _lib.OK:
        raise RuntimeError(
            f"the C library rejected a problem that mfaoa's own checks accepted (code {code}); "
            "this is a bug in mfaoa, please report it"
        )
    sigma = ws["sigma"]
    x = ((sigma + 1) // 2).astype(np.int8)
    # The objective at the continuous values z, from their field m = h + J z:
    # h @ z + z @ J @ z / 2 = Σ_i z[i] (h[i] + m[i]) / 2.
    objective_mf = float(ws["z"] @ (ising.h + ws["m"])) / 2 + ising.offset
    state = SpinState(ws["x"], ws["y"], ws["z"])
    return Result(sigma, x, ising.objective(sigma), objective_mf, state)
