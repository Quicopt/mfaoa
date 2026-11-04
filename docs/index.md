# mfaoa

`mfaoa` solves binary quadratic problems without constraints. Every variable
is 0 or 1, and the objective is a quadratic function of the variables,

$$
f(x) = x^\top Q x + c^\top x + \text{constant}, \qquad x \in \{0, 1\}^n,
$$

to be minimized or maximized. QUBO models, Max-Cut and many selection
problems have this form.

The package runs the mean-field approximate optimization algorithm (MFAOA)
for a fixed number of steps, 1000 by default. The result is a good solution,
without a proof that no better one exists. A problem can come from a Pyomo
model, from a binary quadratic model of D-Wave's dimod package, or as NumPy
arrays and SciPy sparse matrices. The algorithm itself is a small C library,
which C programs can also use directly.

## Installation

```sh
pip install "mfaoa[pyomo]"    # with the Pyomo solver
pip install "mfaoa[dimod]"    # with the dimod sampler
pip install mfaoa             # NumPy and SciPy only
```

The package needs Python 3.10 or later. PyPI has wheels for Linux, macOS and
Windows.

## Example

The problem below has three variables:

$$
\text{minimize} \quad 2 x_1 x_2 - 3 x_2 x_3 - x_1 - x_2 + x_3 .
$$

Each tab states it in one of the three front-ends.

=== "Pyomo"

    ```python exec="on" source="above" result="text"
    import pyomo.environ as pyo
    import mfaoa.pyomo  # makes the solver "mfaoa" known to Pyomo

    model = pyo.ConcreteModel()
    model.x = pyo.Var([1, 2, 3], domain=pyo.Binary)
    model.f = pyo.Objective(
        expr=2 * model.x[1] * model.x[2] - 3 * model.x[2] * model.x[3]
        - model.x[1] - model.x[2] + model.x[3],
        sense=pyo.minimize,
    )
    pyo.SolverFactory("mfaoa").solve(model)
    print([int(pyo.value(model.x[i])) for i in (1, 2, 3)], pyo.value(model.f))
    ```

=== "dimod"

    ```python exec="on" source="above" result="text"
    import dimod
    from mfaoa.dimod import MFAOASampler

    bqm = dimod.BinaryQuadraticModel(
        {1: -1, 2: -1, 3: 1},       # -x1 - x2 + x3
        {(1, 2): 2, (2, 3): -3},    # 2 x1 x2 - 3 x2 x3
        0.0,
        dimod.BINARY,
    )
    print(MFAOASampler().sample(bqm))
    ```

=== "NumPy"

    ```python exec="on" source="above" result="text"
    import numpy as np
    import mfaoa

    Q = np.array([[0, 2, 0],     # 2 x1 x2
                  [0, 0, -3],    # -3 x2 x3
                  [0, 0, 0]])
    problem = mfaoa.QUBOProblem(Q, c=[-1, -1, 1])    # -x1 - x2 + x3
    result = mfaoa.solve(problem)
    print(result.x, problem.objective(result.x))
    ```

All three find x = (0, 1, 1) with the objective value −3, the smallest of
the eight assignments. dimod calls the objective value the energy.

## Contents

- [Max-Cut on the Gset graphs](tutorials/maxcut-gset.md): a tutorial on a
  standard benchmark, in Pyomo and with SciPy sparse matrices.
- [Problem forms](manual/problems.md): the QUBO form with 0/1 variables, the
  Ising form with ±1 variables, and the result of a solve.
- [Pyomo](manual/pyomo.md): what a model may contain, the options and the
  statuses.
- [dimod](manual/dimod.md): the sampler, its parameters and the sample set it
  returns.
- [NumPy and SciPy](manual/numpy.md): dense and sparse coupling matrices.
- [Schedules and the field scale](manual/schedules.md): the settings that
  trade run time for the quality of the solution.
- [Symmetric problems and fixing variables](manual/symmetry.md): some
  objectives, such as Max-Cut's, need a fixed variable.
- [Performance](manual/performance.md): the cost of a solve, sparse and dense
  matrices, and the BLAS.
- [The C library](c-library.md): building it and calling it from C.
- [API reference](api.md): every function and class.

## Reference

MFAOA is the mean-field approximate optimization algorithm of

> A. Misra-Spieldenner, T. Bode, P. K. Schuhmacher, T. Stollenwerk, D. Bagrets
> and F. K. Wilhelm, *Mean-field approximate optimization algorithm*,
> PRX Quantum **4**, 030335 (2023),
> [doi:10.1103/PRXQuantum.4.030335](https://doi.org/10.1103/PRXQuantum.4.030335).
