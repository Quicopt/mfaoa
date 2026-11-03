"""MFAOA as a Pyomo solver.

Importing this module registers the solver with Pyomo under the name
"mfaoa":

    import pyomo.environ as pyo
    import mfaoa.pyomo

    results = pyo.SolverFactory("mfaoa").solve(model)

The model may contain binary variables, a linear or quadratic objective in
either sense, and nothing else: no constraints. A variable can be held at 0
or 1 with its ``fix`` method, or with bounds that allow only one of the two
values. Bounds that allow both are ignored. Integer variables whose bounds
allow only 0 and 1 count as binary. Anything else raises Pyomo's
`IncompatibleModelError`, with the reason.

Options go in ``solver_options`` (``options`` in Pyomo's older interface):

- ``"p"``: the number of steps of the linear schedule (default 1000);
- ``"tau"``: its step size (default 0.5);
- ``"beta"``, ``"gamma"``: an explicit schedule as two sequences of equal
  length, used instead of the linear schedule; see `mfaoa.Schedule`.

MFAOA returns a good solution without a proof that no better one exists.
The results report it as `TerminationCondition.convergenceCriteriaSatisfied`
with `SolutionStatus.feasible`, which is Pyomo's status for that case. In
Pyomo's older interface, ``pyo.check_optimal_termination(results)`` then
returns True, although the solution is not proven optimal.
"""

import datetime
import math
import time

import numpy as np
import scipy.sparse as sp
from pyomo.common.collections import ComponentMap
from pyomo.contrib.solver.common.base import Availability, SolverBase
from pyomo.contrib.solver.common.config import SolverConfig
from pyomo.contrib.solver.common.factory import SolverFactory
from pyomo.contrib.solver.common.results import (
    Results,
    SolutionStatus,
    TerminationCondition,
    get_infeasible_results,
)
from pyomo.contrib.solver.common.solution_loader import SolutionLoader
from pyomo.contrib.solver.common.util import (
    IncompatibleModelError,
    NoDualsError,
    NoOptimalSolutionError,
    NoReducedCostsError,
)
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.objective import Objective, maximize
from pyomo.core.base.sos import SOSConstraint
from pyomo.core.expr.visitor import identify_variables
from pyomo.core.staleflag import StaleFlagManager
from pyomo.repn.standard_repn import generate_standard_repn

from . import __version__
from ._problems import QUBOProblem, coupling_storage, schedule_from
from ._solve import solve as _solve
from ._symmetry import FlipSymmetricError, flip_message

# The words in which a flip-symmetric objective is explained to a Pyomo user.
PYOMO_WORDS = dict(
    flip="x → 1 - x",
    start="1/2",
    values="0 or 1",
    fix=" with its fix method",
)

_OPTIONS = ("p", "tau", "beta", "gamma")


class MFAOAConfig(SolverConfig):
    """Pyomo's solver configuration, with one default changed.

    `raise_exception_on_nonoptimal_result` is False here. MFAOA never proves
    a solution optimal, so with Pyomo's default, True, every solve would end
    in `NoOptimalSolutionError`.
    """

    def __init__(self, description=None, doc=None, implicit=False, implicit_domain=None, visibility=0):
        """Declare Pyomo's solver options and set raise_exception_on_nonoptimal_result to False."""
        super().__init__(
            description=description,
            doc=doc,
            implicit=implicit,
            implicit_domain=implicit_domain,
            visibility=visibility,
        )
        option = self.get("raise_exception_on_nonoptimal_result")
        option.set_default_value(False)
        option.reset()


class _Infeasible(Exception):
    """A variable whose fixing or bounds leave it neither 0 nor 1."""


class MFAOASolutionLoader(SolutionLoader):
    """The solution of a solve, for Pyomo: values of the variables, but no duals."""

    def __init__(self, model, values):
        """Keep the model and the solution.

        Args:
            model: the Pyomo model that was solved.
            values: a ComponentMap from each variable of the objective to its
                value, 0.0 or 1.0.
        """
        self._pyomo_model = model
        self._values = values

    def get_number_of_solutions(self):
        """One: MFAOA returns a single solution."""
        return 1

    def get_vars(self, vars_to_load=None):
        """The values of vars_to_load, or of all variables of the objective.

        Args:
            vars_to_load: a sequence of variables of the objective, or None.

        Returns:
            A ComponentMap from variable to value.
        """
        if vars_to_load is None:
            return ComponentMap(self._values.items())
        return ComponentMap((v, self._values[v]) for v in vars_to_load)

    def get_duals(self, cons_to_load=None):
        """Raises NoDualsError: the model has no constraints, and MFAOA computes no duals."""
        raise NoDualsError()

    def get_reduced_costs(self, vars_to_load=None):
        """Raises NoReducedCostsError: MFAOA computes no reduced costs."""
        raise NoReducedCostsError()


def _active_objective(model):
    """The model's single active objective.

    Raises:
        IncompatibleModelError: the model has an active constraint, or no
            active objective, or more than one.
    """
    for kind in (Constraint, SOSConstraint):
        for constraint in model.component_data_objects(kind, active=True, descend_into=True):
            raise IncompatibleModelError(
                f"The model has the constraint {constraint.name}; "
                "MFAOA solves problems without constraints only. "
                "Hold a variable at 0 or 1 with its fix method instead."
            )
    objectives = list(model.component_data_objects(Objective, active=True, descend_into=True))
    if len(objectives) != 1:
        raise IncompatibleModelError(
            f"The model has {len(objectives)} active objectives; MFAOA needs exactly one."
        )
    return objectives[0]


def _check_fixed(objective):
    """Check that every fixed variable of the objective is fixed to 0 or 1.

    Pyomo replaces fixed variables by their values when it reads the
    objective, so this is the only place that sees them.

    Raises:
        IncompatibleModelError: a variable is fixed to another value.
        _Infeasible: a variable is fixed outside its bounds.
    """
    for var in identify_variables(objective.expr, include_fixed=True):
        if not var.fixed:
            continue
        value = var.value
        if value not in (0, 1):
            raise IncompatibleModelError(
                f"{var.name} is fixed to {value}; MFAOA accepts variables fixed to 0 or 1 only."
            )
        lower, upper = _bounds(var)
        if not lower <= value <= upper:
            raise _Infeasible(f"{var.name} is fixed to {value}, but its bounds require {lower} ≤ {var.name} ≤ {upper}.")


def _bounds(var):
    """The bounds of var, with -inf and inf where it has none."""
    lower, upper = var.bounds
    return (-math.inf if lower is None else lower, math.inf if upper is None else upper)


def _held_value(var):
    """The value at which a variable that is not fixed is held.

    Args:
        var: a variable of the objective, not fixed.

    Returns:
        nan when the variable is free to be 0 or 1, and 0.0 or 1.0 when its
        bounds allow only that value.

    Raises:
        IncompatibleModelError: the variable can take values other than 0 and 1.
        _Infeasible: the bounds allow neither 0 nor 1.
    """
    lower, upper = _bounds(var)
    binary = var.is_binary() or (var.is_integer() and lower > -1 and upper < 2)
    if not binary:
        raise IncompatibleModelError(
            f"{var.name} is neither binary nor fixed; MFAOA solves problems in binary variables only."
        )
    allows_0, allows_1 = lower <= 0 <= upper, lower <= 1 <= upper
    if allows_0 and allows_1:
        return math.nan
    if allows_0 or allows_1:
        return 0.0 if allows_0 else 1.0
    raise _Infeasible(f"The bounds {lower} ≤ {var.name} ≤ {upper} allow neither 0 nor 1.")


def _qubo_terms(objective):
    """The objective as x @ Q @ x + c @ x + constant over its variables.

    Fixed variables are already replaced by their values.

    Args:
        objective: an active Pyomo objective.

    Returns:
        (variables, Q, c, constant): the variables that are not fixed, in the
        order in which the objective names them; Q as a SciPy CSR matrix with
        each product term once, a squared variable on the diagonal; c as a
        vector; the constant as a float.

    Raises:
        IncompatibleModelError: the objective is not linear or quadratic.
    """
    repn = generate_standard_repn(objective.expr, quadratic=True, compute_values=True)
    if repn.nonlinear_expr is not None:
        raise IncompatibleModelError(
            f"The objective {objective.name} is not linear or quadratic; MFAOA needs a quadratic objective."
        )
    index = ComponentMap()
    variables = []
    for var in list(repn.linear_vars) + [v for pair in repn.quadratic_vars for v in pair]:
        if var not in index:
            index[var] = len(variables)
            variables.append(var)
    n = len(variables)
    c = np.zeros(n)
    for var, coefficient in zip(repn.linear_vars, repn.linear_coefs):
        c[index[var]] += coefficient
    rows = [index[a] for a, _ in repn.quadratic_vars]
    cols = [index[b] for _, b in repn.quadratic_vars]
    Q = sp.csr_array((np.asarray(repn.quadratic_coefs, dtype=float), (rows, cols)), shape=(n, n))
    return variables, Q, c, float(repn.constant)


def _reduce(Q, c, constant, x):
    """The objective as a function of the free variables alone.

    With f the free variables (x[i] is nan) and F the held ones,

        x'Qx + c'x = x_f' Q_ff x_f + (c_f + Q_fF x_F + Q_Ff' x_F)' x_f
                     + x_F' Q_FF x_F + c_F' x_F,

    so the held variables add to the linear term and to the constant.

    Args:
        Q, c, constant: the objective, as from `_qubo_terms`.
        x: the held values, nan for a free variable.

    Returns:
        (free, Q_ff, c_f, constant_f): the indices of the free variables and
        the reduced objective.
    """
    free = np.flatnonzero(np.isnan(x))
    held = np.flatnonzero(~np.isnan(x))
    x_F = x[held]
    Q_ff = Q[free][:, free]
    c_f = c[free] + Q[free][:, held] @ x_F + Q[held][:, free].T @ x_F
    constant_f = constant + c[held] @ x_F + x_F @ (Q[held][:, held] @ x_F)
    return free, Q_ff, c_f, float(constant_f)


def _schedule(options):
    """The schedule that the solver options ask for.

    Args:
        options: the solver options, a mapping with the keys p, tau, beta
            and gamma, each optional.

    Returns:
        A `mfaoa.Schedule`; see `mfaoa._problems.schedule_from`.

    Raises:
        ValueError: an unknown option, or options that `schedule_from` rejects.
    """
    unknown = sorted(set(options) - set(_OPTIONS))
    if unknown:
        raise ValueError(f"MFAOA has no option {unknown[0]!r}; its options are {', '.join(_OPTIONS)}.")
    return schedule_from(**options)


@SolverFactory.register("mfaoa", doc="MFAOA for binary quadratic problems without constraints")
class MFAOA(SolverBase):
    """The MFAOA solver in Pyomo's solver interface; see the module docstring."""

    CONFIG = MFAOAConfig()

    def available(self):
        """Always available: the solver comes with this package."""
        return Availability.FullLicense

    def version(self):
        """The version of mfaoa, as a tuple of integers."""
        return tuple(int(part) for part in __version__.split("."))

    def solve(self, model, **kwds):
        """Solve a Pyomo model.

        Args:
            model: a Pyomo model or block.
            **kwds: Pyomo's solver configuration, such as ``solver_options``
                or ``load_solutions``.

        Returns:
            Pyomo's `Results`.

        Raises:
            IncompatibleModelError: the model is not one MFAOA can solve.
            FlipSymmetricError: the objective does not change when a group of
                variables is flipped (x → 1 - x); the message names them, and
                the error's `groups` lists them.
            NoSolutionError: the variables' fixings and bounds leave no
                solution, and ``load_solutions`` is True.
        """
        start = time.perf_counter()
        started = datetime.datetime.now(datetime.timezone.utc)
        config = self.config(value=kwds, preserve_implicit=True)
        if config.time_limit is not None:
            raise ValueError("MFAOA runs a fixed number of steps and has no time limit; set p instead.")
        schedule = _schedule(config.solver_options.value())
        StaleFlagManager.mark_all_as_stale()

        objective = _active_objective(model)
        try:
            _check_fixed(objective)
            variables, Q, c, constant = _qubo_terms(objective)
            x = np.array([_held_value(v) for v in variables])
        except _Infeasible as reason:
            return get_infeasible_results(model, self, config, str(reason))
        sense = "max" if objective.sense == maximize else "min"
        free, Q_ff, c_f, constant_f = _reduce(Q, c, constant, x)

        if free.size == 0:
            status, log = SolutionStatus.optimal, "all variables held"
        else:
            problem = QUBOProblem(coupling_storage(Q_ff), c=c_f, constant=constant_f, sense=sense)
            try:
                x[free] = _solve(problem, schedule).x
            except FlipSymmetricError as error:
                groups = [[variables[free[k]] for k in group] for group in error.groups]
                message = flip_message(error.groups, free.size, lambda k: variables[free[k]].name, PYOMO_WORDS)
                raise FlipSymmetricError(message, groups) from None
            status = SolutionStatus.feasible
            log = f"MFAOA ran {len(schedule)} steps on {free.size} free variables."
        for stream in config.tee:
            stream.write(log + "\n")

        results = Results()
        results.solver_name = self.name
        results.solver_version = self.version()
        results.solver_config = config
        results.solver_log = log
        results.termination_condition = TerminationCondition.convergenceCriteriaSatisfied
        results.solution_status = status
        results.incumbent_objective = float(x @ (Q @ x) + c @ x + constant)
        results.objective_bound = math.inf if sense == "max" else -math.inf
        results.solution_loader = MFAOASolutionLoader(model, ComponentMap(zip(variables, x.tolist())))
        results.timing_info.start_timestamp = started
        if config.raise_exception_on_nonoptimal_result and status != SolutionStatus.optimal:
            raise NoOptimalSolutionError()
        if config.load_solutions:
            results.solution_loader.load_solution()
        results.timing_info.wall_time = time.perf_counter() - start
        return results


__all__ = ["MFAOA", "MFAOAConfig", "MFAOASolutionLoader"]
