# Pyomo

Importing `mfaoa.pyomo` makes MFAOA known to Pyomo as the solver `"mfaoa"`.
It comes with `pip install "mfaoa[pyomo]"`.

## Supported models

A model may contain

- binary variables (`domain=pyo.Binary`),
- integer variables whose bounds allow only 0 and 1,
- one objective, linear or quadratic, to be minimized or maximized,
- variables fixed to 0 or 1 with their `fix` method.

A bound that allows only one of the values 0 and 1, such as a lower bound of
1 on a binary variable, holds the variable at that value, as `fix` does.

As an example, choose which of four features to build. Each feature has a
value. Some pairs of features interact: features 1 and 3 overlap, so
building both loses 5, while features 2 and 4 together are worth 2 more.

```python exec="on" source="above" result="text" session="pyomo"
import pyomo.environ as pyo
import mfaoa.pyomo  # makes the solver "mfaoa" known to Pyomo

model = pyo.ConcreteModel()
model.build = pyo.Var([1, 2, 3, 4], domain=pyo.Binary)  # build[f] = 1: feature f is built
b = model.build
model.value = pyo.Objective(
    expr=3 * b[1] + 2 * b[2] + 4 * b[3] + b[4]
    + b[1] * b[2] - 5 * b[1] * b[3] + 2 * b[2] * b[4] - b[3] * b[4],
    sense=pyo.maximize,
)
pyo.SolverFactory("mfaoa").solve(model)

print("build:", [int(pyo.value(b[f])) for f in model.build])
print("value:", pyo.value(model.value))
```

The best choice builds features 1, 2 and 4, for a value of 9. If feature 3
must be built, fix its variable to 1 and solve again:

```python exec="on" source="above" result="text" session="pyomo"
b[3].fix(1)
pyo.SolverFactory("mfaoa").solve(model)

print("build:", [int(pyo.value(b[f])) for f in model.build])
print("value:", pyo.value(model.value))
```

`b[3].unfix()` removes the fixing again.

## Unsupported models

MFAOA solves problems without constraints. For a model with a constraint,
`solve` raises Pyomo's `IncompatibleModelError`, and the message names the
constraint:

```python exec="on" source="above" result="text" session="pyomo"
from pyomo.contrib.solver.common.util import IncompatibleModelError

pair = pyo.ConcreteModel()
pair.x = pyo.Var([1, 2], domain=pyo.Binary)
pair.f = pyo.Objective(expr=pair.x[1] - pair.x[2])
pair.at_most_one = pyo.Constraint(expr=pair.x[1] + pair.x[2] <= 1)
try:
    pyo.SolverFactory("mfaoa").solve(pair)
except IncompatibleModelError as error:
    print(error)
```

A continuous variable, an integer variable that may take other values than
0 and 1, and a variable fixed to another value than 0 or 1 raise the same
error, with a message that names the variable.

An objective that does not change when every variable x is replaced by
1 − x raises a `FlipSymmetricError` instead. Max-Cut has such an objective,
and [Symmetric problems and fixing variables](symmetry.md) explains what to
do about it.

## Statuses and results

Pyomo has two solver interfaces. `pyo.SolverFactory`, used above, is the
familiar one. The newer one, `SolverFactory` from
`pyomo.contrib.solver.common.factory`, is still marked as a preview by
Pyomo. `mfaoa` works with both.

The newer interface reports what happened in two fields of the results:

| `termination_condition` | `solution_status` | Meaning |
|:--- |:--- |:--- |
| `convergenceCriteriaSatisfied` | `feasible` | MFAOA returned a solution, without a proof that no better one exists. |
| `convergenceCriteriaSatisfied` | `optimal` | Every variable is fixed or held by its bounds, so there was nothing to search. |
| `provenInfeasible` | `noSolution` | The bounds of a variable allow neither 0 nor 1. |

In the last case, `solve` raises Pyomo's `NoSolutionError` unless it was
called with `load_solutions=False`. `incumbent_objective` gives the
objective value of the solution:

```python exec="on" source="above" result="text" session="pyomo"
from pyomo.contrib.solver.common.factory import SolverFactory

b[3].unfix()
results = SolverFactory("mfaoa").solve(model)

print(results.termination_condition)
print(results.solution_status)
print(results.incumbent_objective)
```

The familiar interface translates these conditions into its own, older
ones. A solution without a proof becomes `optimal`, and
`pyo.check_optimal_termination(results)` returns True, even though the
solution is not proven optimal. The termination message keeps the original
condition:

```python exec="on" source="above" result="text" session="pyomo"
results = pyo.SolverFactory("mfaoa").solve(model)

print(results.solver.termination_condition)
print(results.solver.termination_message)
```

## Options

| Option | Meaning | Default |
|:--- |:--- |:--- |
| `"p"` | the number of steps that MFAOA runs | 1000 |
| `"tau"` | the step size τ of the linear schedule | 0.5 |
| `"beta"`, `"gamma"` | a schedule of your own, as two sequences of equal length; used instead of the linear schedule when both are set | not set |

[Schedules and the field scale](schedules.md) explains what these settings
do. The familiar interface takes them as `options`, the newer one as
`solver_options`. More steps usually give a better solution and take
proportionally longer. The newer interface keeps a summary of the run in
`solver_log`, and `tee=True` prints it in both interfaces:

```python exec="on" source="above" result="text" session="pyomo"
results = SolverFactory("mfaoa").solve(model, solver_options={"p": 200})
print(results.solver_log)
```

MFAOA runs a fixed number of steps, so it has no time limit. Setting
Pyomo's `time_limit` raises a `ValueError`; set `p` instead.

## Storage of the objective

MFAOA stores the quadratic part of the objective as a sparse matrix, unless
at least a quarter of all pairs of variables appear in it together. Then it
stores a dense matrix. [Performance](performance.md) explains why this
matters for the run time.
