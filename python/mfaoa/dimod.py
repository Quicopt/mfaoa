"""MFAOA as a dimod sampler, for D-Wave's Ocean tools.

    import dimod
    from mfaoa.dimod import MFAOASampler

    bqm = dimod.BinaryQuadraticModel.from_qubo({("a", "b"): 2.0, ("a", "a"): -1.0})
    sampleset = MFAOASampler().sample(bqm)
    sampleset.first.sample, sampleset.first.energy

The sampler takes a binary quadratic model (BQM) in either vartype, SPIN or
BINARY, and returns a SampleSet with one sample, in the BQM's own vartype and
variable labels. MFAOA is deterministic, so a second sample would repeat the
first. Like every dimod sampler, it minimizes the energy.

The options are those of `mfaoa.solve`: ``p`` and ``tau`` for the linear
schedule (defaults 1000 and 0.5), or ``beta`` and ``gamma`` for a schedule of
your own.
"""

import time

import dimod
import numpy as np
import scipy.sparse as sp

from ._problems import IsingProblem, coupling_storage, schedule_from
from ._solve import solve as _solve
from ._symmetry import ISING_WORDS, QUBO_WORDS, FlipSymmetricError, flip_message

# The advice to fix a variable, in dimod's terms.
_FIX = " with bqm.fix_variable"


def ising_problem(bqm):
    """The BQM in Ising form, as an `IsingProblem`.

    Args:
        bqm: a dimod BinaryQuadraticModel, SPIN or BINARY.

    Returns:
        (problem, labels): the problem, and the BQM's variable labels in the
        order of the problem's variables.
    """
    spin = bqm.change_vartype(dimod.SPIN, inplace=False)
    labels = list(spin.variables)
    vectors = spin.to_numpy_vectors(variable_order=labels)
    n = len(labels)
    rows, cols = vectors.quadratic.row_indices, vectors.quadratic.col_indices
    biases = vectors.quadratic.biases
    # The BQM stores each interaction once; J holds it at (u, v) and (v, u).
    couplings = sp.coo_array(
        (np.concatenate([biases, biases]), (np.concatenate([rows, cols]), np.concatenate([cols, rows]))),
        shape=(n, n),
    ).tocsc()
    problem = IsingProblem(vectors.linear_biases, coupling_storage(couplings), offset=float(vectors.offset))
    return problem, labels


class MFAOASampler(dimod.Sampler):
    """MFAOA as a dimod sampler; see the module docstring."""

    parameters = {"p": [], "tau": [], "beta": [], "gamma": []}
    properties = {}

    def sample(self, bqm, p=None, tau=None, beta=None, gamma=None, **kwargs):
        """Solve a binary quadratic model.

        Args:
            bqm: a dimod BinaryQuadraticModel, SPIN or BINARY.
            p: the number of steps of the linear schedule (default 1000).
            tau: its step size (default 0.5).
            beta, gamma: an explicit schedule, used instead of the linear one.
            **kwargs: other keywords, which dimod's conventions say to ignore
                with a warning.

        Returns:
            A SampleSet with one sample. Its info holds "steps", the number
            of steps run, and "solve_time", in seconds.

        Raises:
            FlipSymmetricError: the energy does not change when a group of
                variables is flipped; the message names them by their labels,
                and the error's `groups` lists those labels.
        """
        kwargs = self.remove_unknown_kwargs(**kwargs)
        schedule = schedule_from(p, tau, beta, gamma)
        start = time.perf_counter()
        if bqm.num_variables == 0:
            return dimod.SampleSet.from_samples_bqm(([], []), bqm, info={"steps": 0, "solve_time": 0.0})
        problem, labels = ising_problem(bqm)
        try:
            result = _solve(problem, schedule)
        except FlipSymmetricError as error:
            words = dict(ISING_WORDS if bqm.vartype is dimod.SPIN else QUBO_WORDS, fix=_FIX)
            message = flip_message(error.groups, len(labels), lambda k: repr(labels[k]), words)
            raise FlipSymmetricError(message, [[labels[k] for k in g] for g in error.groups]) from None
        values = result.sigma if bqm.vartype is dimod.SPIN else result.x
        info = {"steps": len(schedule), "solve_time": time.perf_counter() - start}
        return dimod.SampleSet.from_samples_bqm((values.reshape(1, -1), labels), bqm, info=info)


__all__ = ["MFAOASampler", "ising_problem"]
