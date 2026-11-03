"""The Pyomo interface. The tests mirror test/MOI_wrapper.jl of MFAOA.jl, so
that a Pyomo user can do what a JuMP user can do."""

import io
import itertools

import numpy as np
import pytest

pyo = pytest.importorskip("pyomo.environ")

import mfaoa
import mfaoa.pyomo
from pyomo.contrib.solver.common.factory import SolverFactory
from pyomo.contrib.solver.common.results import SolutionStatus, TerminationCondition
from pyomo.contrib.solver.common.util import (
    IncompatibleModelError,
    NoDualsError,
    NoOptimalSolutionError,
    NoSolutionError,
)


def solver():
    """A fresh MFAOA solver in Pyomo's current interface."""
    return SolverFactory("mfaoa")


def values(var):
    """The values of an indexed variable, as a list."""
    return [pyo.value(v) for v in var.values()]


def three_variable_objective(x):
    """The objective of the JuMP tests: minimal, -3, at x = (0, 1, 1)."""
    return 2 * x[0] * x[1] - 3 * x[1] * x[2] - x[0] - x[1] + x[2]


def cut(x, edges):
    """The number of edges whose ends have different values."""
    return sum(x[i] + x[j] - 2 * x[i] * x[j] for i, j in edges)


def binary_model(n, **kwds):
    """A model with n binary variables x[0], …, x[n-1]."""
    model = pyo.ConcreteModel()
    model.x = pyo.Var(range(n), domain=pyo.Binary, **kwds)
    return model


def test_min_and_max_of_a_qubo():
    """A three-variable QUBO, minimized and maximized, with the results Pyomo reports."""
    model = binary_model(3)
    model.obj = pyo.Objective(expr=three_variable_objective(model.x))
    results = solver().solve(model)
    assert results.termination_condition == TerminationCondition.convergenceCriteriaSatisfied
    assert results.solution_status == SolutionStatus.feasible
    assert values(model.x) == [0, 1, 1]
    assert results.incumbent_objective == -3 == pyo.value(model.obj)
    assert results.objective_bound == -np.inf
    assert results.timing_info.wall_time >= 0
    assert "steps" in results.solver_log
    assert results.solver_version == tuple(int(k) for k in mfaoa.__version__.split("."))

    model.obj.deactivate()
    model.obj_max = pyo.Objective(expr=-three_variable_objective(model.x) + 1, sense=pyo.maximize)
    results = solver().solve(model)
    assert values(model.x) == [0, 1, 1]
    assert results.incumbent_objective == 4
    assert results.objective_bound == np.inf


def test_through_pyomos_older_interface():
    """pyo.SolverFactory finds the solver and solves through Pyomo's older interface."""
    model = binary_model(3)
    model.obj = pyo.Objective(expr=three_variable_objective(model.x))
    results = pyo.SolverFactory("mfaoa").solve(model, options={"p": 200})
    assert pyo.check_optimal_termination(results)  # documented: not proven optimal
    assert values(model.x) == [0, 1, 1]


def test_objective_forms():
    """Linear objectives, a single variable, and a squared variable."""
    model = binary_model(2)
    model.obj = pyo.Objective(expr=model.x[0] - model.x[1])  # linear
    assert solver().solve(model).incumbent_objective == -1
    assert values(model.x) == [0, 1]
    model.obj.set_value(model.x[0])  # a single variable
    model.obj.sense = pyo.maximize
    assert solver().solve(model).incumbent_objective == 1
    model.obj.set_value(model.x[0] ** 2 - 2 * model.x[0] + 0.5)  # x² = x on binaries
    model.obj.sense = pyo.minimize
    assert solver().solve(model).incumbent_objective == -0.5
    assert pyo.value(model.x[0]) == 1


def test_fixing_variables_breaks_the_symmetry():
    """Max-cut of a triangle: rejected until one variable is fixed."""
    edges = [(0, 1), (1, 2), (0, 2)]  # max-cut of a triangle
    model = binary_model(3)
    model.obj = pyo.Objective(expr=cut(model.x, edges), sense=pyo.maximize)
    with pytest.raises(mfaoa.FlipSymmetricError) as error:
        solver().solve(model)
    assert "every variable is flipped (x → 1 - x)" in str(error.value)
    assert "Fix one variable to 0 or 1 with its fix method, for example x[0]." in str(error.value)
    assert error.value.groups == [[model.x[0], model.x[1], model.x[2]]]

    model.x[0].fix(1)
    results = solver().solve(model)
    assert pyo.value(model.x[0]) == 1
    assert results.incumbent_objective == 2 == pyo.value(model.obj)  # every cut of a triangle

    model.x[1].fix(0)
    model.x[2].fix(0)
    results = solver().solve(model)  # nothing left to solve
    assert results.solution_status == SolutionStatus.optimal
    assert values(model.x) == [1, 0, 0]
    assert results.incumbent_objective == 2


def test_every_separate_part_needs_a_fixed_variable():
    """Two separate edges: each needs a fixed variable of its own."""
    model = binary_model(4)
    model.obj = pyo.Objective(expr=cut(model.x, [(0, 1), (2, 3)]), sense=pyo.maximize)
    model.x[0].fix(0)
    with pytest.raises(mfaoa.FlipSymmetricError) as error:
        solver().solve(model)
    assert "the variables x[2] and x[3] are all flipped" in str(error.value)
    assert "Fix one of them to 0 or 1 with its fix method, for example x[2]." in str(error.value)
    model.x[2].fix(1)
    assert solver().solve(model).incumbent_objective == 2


def test_bounds_and_integer_variables():
    """Bounds that allow 0 and 1, integers within [0, 1], held, rejected and infeasible variables."""
    model = binary_model(3, bounds=(0, 1))  # bounds that allow 0 and 1 change nothing
    model.obj = pyo.Objective(expr=three_variable_objective(model.x))
    solver().solve(model)
    assert values(model.x) == [0, 1, 1]

    model = pyo.ConcreteModel()
    model.x = pyo.Var(range(3), domain=pyo.Integers, bounds=(0, 1))  # binary
    model.obj = pyo.Objective(expr=three_variable_objective(model.x))
    solver().solve(model)
    assert values(model.x) == [0, 1, 1]

    model = binary_model(3)
    model.x[0].setlb(1)  # a bound that allows only 1 holds the variable there
    model.obj = pyo.Objective(expr=three_variable_objective(model.x))
    results = solver().solve(model)
    assert values(model.x) == [1, 1, 1]
    assert results.incumbent_objective == -2
    model.x[1].setub(0.5)
    model.x[2].setub(0)
    results = solver().solve(model)
    assert results.solution_status == SolutionStatus.optimal  # nothing left to solve
    assert values(model.x) == [1, 0, 0]

    model = pyo.ConcreteModel()
    model.y = pyo.Var(domain=pyo.Integers, bounds=(0, 2))
    model.obj = pyo.Objective(expr=model.y)
    with pytest.raises(IncompatibleModelError, match=r"^y is neither binary nor fixed; MFAOA solves problems in binary variables only\.$"):
        solver().solve(model)

    model = pyo.ConcreteModel()
    model.z = pyo.Var(domain=pyo.Binary, bounds=(0.2, 0.8))
    model.obj = pyo.Objective(expr=model.z)
    with pytest.raises(NoSolutionError, match=r"The bounds 0\.2 ≤ z ≤ 0\.8 allow neither 0 nor 1\."):
        solver().solve(model)
    results = solver().solve(model, load_solutions=False)
    assert results.termination_condition == TerminationCondition.provenInfeasible
    assert results.solution_status == SolutionStatus.noSolution


def test_models_outside_the_problem_class():
    """Continuous variables, fixings to 0.5, constraints, cubic terms and missing objectives."""
    model = pyo.ConcreteModel()
    model.x = pyo.Var(domain=pyo.Binary)
    model.y = pyo.Var()  # continuous
    model.obj = pyo.Objective(expr=model.x * model.y - model.x)
    with pytest.raises(IncompatibleModelError, match=r"^y is neither binary nor fixed"):
        solver().solve(model)

    model = binary_model(2)
    model.x[0].set_value(0.5, skip_validation=True)
    model.x[0].fixed = True
    model.obj = pyo.Objective(expr=model.x[0] * model.x[1])
    with pytest.raises(IncompatibleModelError, match=r"^x\[0\] is fixed to 0\.5; MFAOA accepts variables fixed to 0 or 1 only\.$"):
        solver().solve(model)

    model = binary_model(2)
    model.obj = pyo.Objective(expr=model.x[0])
    model.c = pyo.Constraint(expr=model.x[0] + model.x[1] <= 1)
    with pytest.raises(IncompatibleModelError, match=r"the constraint c; MFAOA solves problems without constraints"):
        solver().solve(model)
    model.c.deactivate()  # an inactive constraint is not part of the problem
    solver().solve(model)

    model = binary_model(2)
    model.obj = pyo.Objective(expr=model.x[0] ** 3)
    with pytest.raises(IncompatibleModelError, match="not linear or quadratic"):
        solver().solve(model)

    model = binary_model(2)
    with pytest.raises(IncompatibleModelError, match="0 active objectives"):
        solver().solve(model)


def ring_solution(weight):
    """A ring of 12 variables with alternating couplings and small biases."""
    model = binary_model(12)
    x = model.x
    model.obj = pyo.Objective(
        expr=weight * (sum((-1) ** i * x[i] * x[(i + 1) % 12] for i in range(12)) - sum(0.1 * (i + 1) * x[i] for i in range(12)))
    )
    results = solver().solve(model)
    return values(x), results.incumbent_objective


def test_units_of_the_objective_do_not_matter():
    """Scaling the objective by 1000 scales the value and keeps the solution."""
    x1, v1 = ring_solution(1)
    x1000, v1000 = ring_solution(1000)
    assert x1000 == x1
    assert v1000 == pytest.approx(1000 * v1)


def test_held_variables_are_substituted_as_brute_force_says():
    """With variables fixed and held by a bound, the result matches brute force."""
    model = binary_model(8)
    x = model.x
    model.obj = pyo.Objective(expr=cut(x, [(i, i + 1) for i in range(7)]) + 0.5 * x[7], sense=pyo.maximize)
    x[0].fix(1)
    x[4].setlb(1)
    results = solver().solve(model)
    best = max(
        cut(y, [(i, i + 1) for i in range(7)]) + 0.5 * y[7]
        for rest in itertools.product([0, 1], repeat=6)
        for y in [(1, *rest[:3], 1, *rest[3:])]
    )
    assert results.incumbent_objective == best == pyo.value(model.obj)
    assert pyo.value(x[0]) == 1 and pyo.value(x[4]) == 1


def test_options():
    """The options p, tau, beta and gamma, and the ones that are rejected."""
    model = binary_model(2)
    model.obj = pyo.Objective(expr=model.x[0] * model.x[1] - model.x[0])
    results = solver().solve(model, solver_options={"p": 200, "tau": 0.4})
    assert values(model.x) == [1, 0]
    assert "200 steps" in results.solver_log

    schedule = mfaoa.linear_schedule(50)
    with pytest.raises(ValueError, match="both beta and gamma"):
        solver().solve(model, solver_options={"beta": schedule.beta})
    results = solver().solve(model, solver_options={"beta": schedule.beta, "gamma": schedule.gamma})
    assert "50 steps" in results.solver_log

    with pytest.raises(ValueError, match="need at least one step"):
        solver().solve(model, solver_options={"p": 0})
    with pytest.raises(ValueError, match="tau must be positive"):
        solver().solve(model, solver_options={"tau": -1.0})
    with pytest.raises(ValueError, match="MFAOA has no option 'layers'"):
        solver().solve(model, solver_options={"layers": 3})
    with pytest.raises(ValueError, match="no time limit"):
        solver().solve(model, time_limit=10)


def test_results_without_loading_and_the_strict_flag():
    """load_solutions=False, the solution loader, and raise_exception_on_nonoptimal_result."""
    model = binary_model(3)
    model.obj = pyo.Objective(expr=three_variable_objective(model.x))
    results = solver().solve(model, load_solutions=False)
    assert all(v.value is None for v in model.x.values())
    assert [results.solution_loader.get_vars()[model.x[i]] for i in range(3)] == [0, 1, 1]
    with pytest.raises(NoDualsError):
        results.solution_loader.get_duals()
    with pytest.raises(NoOptimalSolutionError):
        solver().solve(model, raise_exception_on_nonoptimal_result=True)


def test_tee_prints_the_log():
    """tee writes the one-line log."""
    model = binary_model(2)
    model.obj = pyo.Objective(expr=model.x[0] * model.x[1] - model.x[0])
    stream = io.StringIO()
    solver().solve(model, tee=stream, solver_options={"p": 30})
    assert stream.getvalue() == "MFAOA ran 30 steps on 2 free variables.\n"


def test_sparse_and_dense_storage():
    """Couplings stay sparse below a quarter fill and become dense above."""
    import scipy.sparse as sp
    from mfaoa._problems import coupling_storage

    sparse_Q = sp.csr_array(([1.0, -1.0], ([0, 1], [1, 2])), shape=(10, 10))
    assert sp.issparse(coupling_storage(sparse_Q))
    dense_Q = sp.csr_array(np.triu(np.ones((10, 10)), 1))
    assert isinstance(coupling_storage(dense_Q), np.ndarray)
