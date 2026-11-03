<a href="https://quicopt.com">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Quicopt/mfaoa/main/.github/images/quicopt-logo-dark.svg">
    <img alt="Quicopt" src="https://raw.githubusercontent.com/Quicopt/mfaoa/main/.github/images/quicopt-logo.svg" width="240">
  </picture>
</a>

# mfaoa

The `mfaoa` package solves binary quadratic problems without constraints. In
such a problem, every variable is 0 or 1, and the objective is a quadratic
function of the variables, to be minimized or maximized. Max-Cut, QUBO models
and many selection problems have this form.

The package runs the mean-field approximate optimization algorithm (MFAOA)
of Misra-Spieldenner et al. (2023) for a fixed number of steps. The result is
a good solution, without a proof that no better one exists. The package
solves Pyomo models, binary quadratic models from D-Wave's dimod package,
and problems given as NumPy arrays or SciPy sparse matrices. Its core is a
small C library, which C programs can also use directly.

[Quicopt](https://quicopt.com) develops and maintains `mfaoa`.

## Installation

```sh
pip install "mfaoa[pyomo]"    # with the Pyomo solver
pip install "mfaoa[dimod]"    # with the dimod sampler
pip install mfaoa             # NumPy and SciPy only
```

The package needs Python 3.10 or later. PyPI has wheels for Linux, macOS and
Windows.

## Solving a Pyomo model

A company can take on five projects, each with its own profit. Some pairs of
projects need the same staff, and taking on both projects of such a pair
costs extra:

```python
import pyomo.environ as pyo
import mfaoa.pyomo  # makes the solver "mfaoa" known to Pyomo

value = {"A": 6, "B": 5, "C": 4, "D": 3, "E": 3}
overlap = {("A", "B"): 4, ("A", "C"): 3, ("B", "D"): 2, ("C", "E"): 2, ("D", "E"): 1}

model = pyo.ConcreteModel()
model.choose = pyo.Var(value.keys(), domain=pyo.Binary)
model.profit = pyo.Objective(
    expr=sum(value[p] * model.choose[p] for p in value)
    - sum(cost * model.choose[p] * model.choose[q] for (p, q), cost in overlap.items()),
    sense=pyo.maximize,
)
pyo.SolverFactory("mfaoa").solve(model)
print([p for p in value if pyo.value(model.choose[p]) == 1], pyo.value(model.profit))
```

```
['A', 'D', 'E'] 11.0
```

The best choice is projects A, D and E. Their profit is 6 + 3 + 3, minus 1
for the staff that D and E share.

The model may contain binary variables and one objective, linear or
quadratic, in either sense. Constraints are not allowed; the
[Quicopt service](#quicopt) solves models with constraints. To hold a variable
at 0 or 1, fix it, as in `model.choose["A"].fix(1)`, or give it bounds that
allow only that value. Integer variables with bounds 0 and 1 count as
binary. For any other model, `solve` raises Pyomo's `IncompatibleModelError`,
and the message names the component that is not allowed.

The option `p` sets the number of steps, 1000 by default:
`pyo.SolverFactory("mfaoa").solve(model, options={"p": 5000})`. More steps usually give a better
solution, and the time grows in proportion to `p`.

The results report `TerminationCondition.convergenceCriteriaSatisfied` with
`SolutionStatus.feasible`, Pyomo's status for a solution that is not proven
optimal. `pyo.check_optimal_termination(results)` returns True even so,
since Pyomo's older interface maps that status to "optimal".

## Objectives that need a fixed variable

Max-Cut is the most common example. Each vertex `i` of a graph gets a
variable `x[i]`, and the objective, the cut, is the total weight of the
edges whose two vertices have different values:

```python
weight = {(0, 1): 2, (0, 2): 5, (1, 2): 5, (1, 3): 2, (2, 3): 3, (3, 4): 5, (2, 4): 4}

model = pyo.ConcreteModel()
model.x = pyo.Var(range(5), domain=pyo.Binary)
model.cut = pyo.Objective(
    expr=sum(w * (model.x[i] + model.x[j] - 2 * model.x[i] * model.x[j]) for (i, j), w in weight.items()),
    sense=pyo.maximize,
)
pyo.SolverFactory("mfaoa").solve(model)
```

```
mfaoa.FlipSymmetricError: The objective does not change when every variable is flipped (x → 1 - x).
MFAOA starts all variables at 1/2, where such an objective has zero slope in every variable, so no
variable would ever move. Fix one variable to 0 or 1 with its fix method, for example x[0]. This loses
nothing, since flipping every variable does not change the objective value.
```

Replacing every `x[i]` by `1 - x[i]` swaps the two sets of vertices and
leaves every cut as it is. After fixing one variable, the solve succeeds:

```python
model.x[0].fix(0)
pyo.SolverFactory("mfaoa").solve(model)
print([int(pyo.value(model.x[i])) for i in range(5)], pyo.value(model.cut))
```

```
[0, 0, 1, 1, 0] 21.0
```

The largest cut, 21, separates the vertices 0, 1 and 4 from 2 and 3.

A group of variables can have the same property: no term of the objective
links the group to the other variables, and flipping the whole group leaves
the objective unchanged. Each such group needs one fixed variable, and the
message names one per group. In the dimod sampler and in
`mfaoa.solve`, the error is the same `mfaoa.FlipSymmetricError`, and its
attribute `groups` lists the variables of each group.

## Solving a dimod model

The `mfaoa.dimod` module offers `MFAOASampler`, a sampler for binary
quadratic models of D-Wave's dimod package. It takes the same models as
D-Wave's own samplers and returns the same kind of `SampleSet`, so a script
written for one of them runs with the other. The project problem from above,
as a binary quadratic model:

```python
import dimod
from mfaoa.dimod import MFAOASampler

value = {"A": 6, "B": 5, "C": 4, "D": 3, "E": 3}
overlap = {("A", "B"): 4, ("A", "C"): 3, ("B", "D"): 2, ("C", "E"): 2, ("D", "E"): 1}

# dimod minimizes, so the energy is the negative profit
bqm = dimod.BinaryQuadraticModel({p: -v for p, v in value.items()}, overlap, 0.0, dimod.BINARY)
print(MFAOASampler().sample(bqm))
```

```
   A  B  C  D  E energy num_oc.
0  1  0  0  1  1  -11.0       1
['BINARY', 1 rows, 1 samples, 5 variables]
```

The sample set holds a single sample, because the algorithm is
deterministic: a second run returns the same one. `sample` takes the
options `p` and `tau` of the linear schedule, as `mfaoa.solve` does.

## Solving with NumPy and SciPy

Without a modelling package, a problem is a few arrays. In QUBO form, with
variables `x[i]` that are 0 or 1, the objective is

```math
f(x) = x^T Q x + c^T x + \text{constant}
```

`Q` is used exactly as given: a symmetric matrix counts each pair of
variables twice, an upper-triangular one once. Diagonal entries act as
linear terms, since `x[i]**2 = x[i]`. The project problem in this form:

```python
import numpy as np
import mfaoa

c = np.array([6, 5, 4, 3, 3])   # the profit of projects A to E
Q = np.zeros((5, 5))            # the extra cost of each pair, upper triangle
Q[0, 1], Q[0, 2], Q[1, 3], Q[2, 4], Q[3, 4] = -4, -3, -2, -2, -1
result = mfaoa.solve(mfaoa.QUBOProblem(Q, c=c, sense="max"))
print(result.x, result.objective)
```

```
[1 0 0 1 1] 11.0
```

In Ising form, the variables `σ[i]` are -1 or +1, and the objective is

```math
E(\sigma) = \sum_i h_i \sigma_i + \sum_{i \lt j} J_{ij} \sigma_i \sigma_j + \text{offset}
```

A problem in this form is `mfaoa.IsingProblem(h, J, offset=0.0, sense="min")`.
`J` must be symmetric and zero on the diagonal. A `QUBOProblem` is converted to this
form before the solve.

`Q` and `J` can be NumPy arrays or SciPy sparse matrices. Each step of the
algorithm multiplies `J` by a vector, which costs one multiplication per
entry of a dense matrix and one per stored entry of a sparse one. For a
problem in which most pairs of variables do not interact, a sparse matrix
is therefore much faster.

The result has the solution both as `result.x`, with entries 0 and 1, and
as `result.sigma`, with entries -1 and +1. `result.objective` is the
objective value of the solution. `result.objective_mf` is the objective at
the values between -1 and +1 that the algorithm ends with, before they are
rounded. A large gap between the two means that some variables were still
undecided at the end, and more steps usually close it:
`mfaoa.solve(problem, p=5000)`. A schedule of your own, as two arrays of
angles, goes in as `mfaoa.solve(problem, mfaoa.Schedule(beta, gamma))`; the
docstring of `mfaoa.Schedule` explains the angles.

## The C library

The Python package calls a C library, which C programs can also use
directly. It has one function that solves a problem, and it allocates no
memory: the caller passes every array, including the arrays that receive
the solution.

### The problem

The library minimizes or maximizes

```math
E(\sigma) = \sum_i h_i \sigma_i + \sum_{i \lt j} J_{ij} \sigma_i \sigma_j
```

over variables $\sigma_i$ that are -1 or +1, where $h$ is a vector and $J$ a
symmetric matrix that is zero on the diagonal. A problem in variables $x_i$
that are 0 or 1,

```math
f(x) = \sum_i c_i x_i + \sum_{i \lt j} q_{ij} x_i x_j
```

takes this form with $x_i = (1 + \sigma_i)/2$ and

```math
h_i = \frac{c_i}{2} + \frac{1}{4} \sum_{j \ne i} q_{ij}
```

```math
J_{ij} = J_{ji} = \frac{q_{ij}}{4}
```

```math
f(x) = E(\sigma) + \sum_i \frac{c_i}{2} + \sum_{i \lt j} \frac{q_{ij}}{4}
```

### Building

You need CMake 3.21 or later, a C compiler, and a BLAS with its C interface,
CBLAS. On macOS, Accelerate is part of the system. On Linux, OpenBLAS is the
usual choice (`libopenblas-dev` on Debian and Ubuntu).

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build                    # the reference runs and the example
cmake --install build --prefix <dir>
```

The installation holds `include/mfaoa.h`, the library in `lib/`, and the
files with which other builds find both: `lib/cmake/mfaoa/` for CMake and
`lib/pkgconfig/mfaoa.pc` for pkg-config.

CMake picks the BLAS: Accelerate on macOS, otherwise the first one it finds.
To choose one, pass `-DBLA_VENDOR`, and for an installation outside the
default paths also `-DCMAKE_PREFIX_PATH`. For Homebrew's OpenBLAS on macOS:

```sh
cmake -S . -B build -DBLA_VENDOR=OpenBLAS -DCMAKE_PREFIX_PATH=$(brew --prefix openblas)
```

`-DBUILD_SHARED_LIBS=OFF` builds a static library. A program that links it
must link the same BLAS as well; CMake and pkg-config add it.

`-DMFAOA_BLAS=OFF` builds the library without a BLAS. A program must then
pass a `dgemv` routine in every problem with dense couplings (see below).

The library has been built and tested on macOS with Accelerate and with
OpenBLAS. `mfaoa.h` declares the exports for a Windows DLL, but the build
has not been tried on Windows.

### Using the library

In a CMake project:

```cmake
find_package(mfaoa 0.1 REQUIRED)
target_link_libraries(your_program PRIVATE mfaoa::mfaoa)
```

With pkg-config:

```sh
cc your_program.c $(pkg-config --cflags --libs mfaoa)
```

`examples/solve.c` is a complete program. It minimizes a problem in four
variables and prints

```
sigma = (-1, 1, 1, 1), E = -4.25
```

A solve takes three structs, all declared in `mfaoa.h`:

- `mfaoa_problem` holds $h$ and $J$, and whether to minimize or maximize.
  $J$ is either a dense matrix in column-major order, or a sparse one in
  compressed sparse columns (CSC) with 0-based indices. A sparse $J$ stores
  both triangles, with the rows of each column in increasing order. The
  field `dgemv` can point to a BLAS `dgemv` of your own, in the Fortran
  calling convention, for the product with a dense $J$. NULL, the default,
  uses the BLAS the library is linked to.
- `mfaoa_schedule` holds the angles $\beta_k$ and $\gamma_k$ of the $p$
  steps. `mfaoa_linear_schedule` fills them with the linear schedule of the
  paper. 1000 steps with $\tau = 0.5$ are a good start. More steps usually
  give a better solution, and the time grows in proportion to $p$.
- `mfaoa_workspace` holds the arrays that receive the final state and the
  solution `sigma`.

Each step multiplies $J$ by a vector. A dense $J$ costs one multiplication
per entry, through the BLAS. A sparse $J$ costs one per stored entry.

`mfaoa_solve` returns `MFAOA_OK`, `MFAOA_INVALID_ARGUMENT`, or
`MFAOA_FLIP_SYMMETRIC`. The last one means that flipping a group of
variables, $\sigma_i \to -\sigma_i$ for every $i$ in the group, leaves $E$
unchanged. This happens when $J$ links the group to no other variable and
$h$ is zero on all of the group, as in Max-Cut. The algorithm starts every
variable halfway between $-1$ and $+1$, and there $E$ has zero slope in
these variables, so they would never move. The workspace then names one
variable per group. Fix it to either value: to fix $\sigma_i = s$, remove
variable $i$, add $s J_{ji}$ to $h_j$ for every other variable $j$, and add
$s h_i$ to the constant. Nothing is lost, because flipping the group turns
every solution into one with the same value.

`MFAOA_VERSION` is the version of the header, and `mfaoa_version()` returns
the version of the library that is loaded. While the major version is 0, a
new minor version may change the structs and functions.

## Quicopt

The Quicopt optimization service solves the models that `mfaoa` cannot:
models with constraints, and with integer or continuous variables. The
[`quicopt`](https://pypi.org/project/quicopt/) package sends a model written
with Pyomo, OR-Tools MathOpt or PuLP to the service and returns the answer.
The free tier needs no account. The
[documentation of `quicopt`](https://quicopt.github.io/quicopt-python/) has
the details, and [quicopt.com](https://quicopt.com) has more about Quicopt.

## Reference

A. Misra-Spieldenner, T. Bode, P. K. Schuhmacher, T. Stollenwerk, D. Bagrets
and F. K. Wilhelm, *Mean-Field Approximate Optimization Algorithm*, PRX
Quantum **4**, 030335 (2023),
[doi:10.1103/PRXQuantum.4.030335](https://doi.org/10.1103/PRXQuantum.4.030335).

## Licence

Apache License 2.0, see `LICENSE`.
