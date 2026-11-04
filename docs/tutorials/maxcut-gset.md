# Max-Cut on the Gset graphs

Max-Cut asks for a split of a graph's vertices into two sets, such that the
edges between the two sets carry as much total weight as possible. With one
binary variable per vertex, Max-Cut is a quadratic problem in binary
variables without constraints, the problem class that `mfaoa` solves.

This tutorial solves Max-Cut on graphs from Gset, a collection widely used to
compare Max-Cut methods, and compares the cuts with the best ones published.
It states the problem in Pyomo, with SciPy sparse matrices and in dimod, so
it needs `pip install "mfaoa[pyomo,dimod]"`.

```python exec="on" session="gset"
# Not shown on the page: the build downloads the graphs, as a reader would.
import pathlib
import urllib.request

pathlib.Path("gset").mkdir(exist_ok=True)
for name in ("G1", "G11", "G22", "G43", "G55"):
    target = pathlib.Path("gset") / name
    if not target.exists():
        urllib.request.urlretrieve(f"https://web.stanford.edu/~yyye/yyye/Gset/{name}", target)
```

## Reading a Gset graph

Yinyu Ye distributes the Gset graphs at
[web.stanford.edu/~yyye/yyye/Gset](https://web.stanford.edu/~yyye/yyye/Gset/).
To follow along, download the graphs G1, G11, G22, G43 and G55 from there
into a folder `gset` in your working directory. Each graph is a text file.
Its first line holds the number of vertices and the number of edges. Every
further line describes one edge by its two end vertices, numbered from 1,
and its weight.

```python exec="on" source="above" result="text" session="gset"
from pathlib import Path


def read_gset(name):
    """The number of vertices and the edges, as (i, j, weight) tuples, of the
    Gset graph `name`, such as "G1", from the folder gset."""
    lines = (Path("gset") / name).read_text().splitlines()
    n_vertices, n_edges = (int(field) for field in lines[0].split())
    edges = [tuple(int(field) for field in line.split()) for line in lines[1:n_edges + 1]]
    return n_vertices, edges


n_vertices, edges = read_gset("G1")
print("vertices:", n_vertices)
print("edges:   ", len(edges))
```

## Max-Cut in Pyomo

Give each vertex v a binary variable xᵥ that says which of the two sets it
is in. An edge between vertices i and j is cut when xᵢ and xⱼ differ. The
expression xᵢ + xⱼ − 2xᵢxⱼ is 1 in that case and 0 otherwise, so the value
of the cut is its weighted sum over all edges:

```python exec="on" source="above" result="text" session="gset"
import pyomo.environ as pyo
import mfaoa
import mfaoa.pyomo  # makes the solver "mfaoa" known to Pyomo

model = pyo.ConcreteModel()
model.x = pyo.Var(range(1, n_vertices + 1), domain=pyo.Binary)  # x[v]: the set, 0 or 1, of vertex v
model.cut = pyo.Objective(
    expr=sum(weight * (model.x[i] + model.x[j] - 2 * model.x[i] * model.x[j])
             for i, j, weight in edges),
    sense=pyo.maximize,
)
try:
    pyo.SolverFactory("mfaoa").solve(model)
except mfaoa.FlipSymmetricError as error:
    print(error)
```

In Max-Cut, exchanging the two sets cuts the same edges, so replacing every
xᵥ by 1 − xᵥ does not change the objective. Putting vertex 1 in set 0, as
the message suggests, loses no cut: of a cut and its exchanged copy, one has
vertex 1 in set 0, and both have the same value.

```python exec="on" source="above" result="text" session="gset"
best_known_cut = 11624             # Benlic and Hao (2013), Table 2

model.x[1].fix(0)
pyo.SolverFactory("mfaoa").solve(model)
print("cut found:     ", pyo.value(model.cut))
print("best known cut:", best_known_cut)
```

The cut is one edge short of the best cut of G1 that has been published.
[Symmetric problems and fixing variables](../manual/symmetry.md) explains
when variables must be fixed and how.

## The same problem with SciPy

`mfaoa` also accepts a problem directly, as a vector and a matrix. The form
it works in has variables σᵢ that are −1 or +1 instead of 0 or 1, and it
minimizes

$$
h^\top \sigma + \sum_{i<j} J_{ij}\, \sigma_i \sigma_j ,
$$

where h is a vector and J is a symmetric matrix that is zero on the
diagonal. This form is called the Ising form, and
`mfaoa.IsingProblem(h, J)` states a problem in it.

Max-Cut is easy to state in this form. Let σᵢ say which set vertex i is in.
The product σᵢσⱼ is −1 for an edge that is cut and +1 for one that is not,
so (1 − σᵢσⱼ)/2 is 1 if the edge is cut and 0 otherwise. With wᵢⱼ the weight
of the edge between i and j, and W the total weight of all edges, the value
of the cut is

$$
\operatorname{cut}(\sigma)
  = \sum_{\text{edges } ij} w_{ij}\, \frac{1 - \sigma_i \sigma_j}{2}
  = \frac{W - f(\sigma)}{2},
\qquad
f(\sigma) = \sum_{\text{edges } ij} w_{ij}\, \sigma_i \sigma_j .
$$

The largest cut therefore has the smallest f, and f is in Ising form:
Jᵢⱼ = wᵢⱼ for every edge, and h = 0.

As in Pyomo, one vertex must be fixed. Put vertex 1 in the set σ₁ = +1.
Each edge from vertex 1 to a vertex j then contributes w₁ⱼσⱼ, a linear term:

$$
f(\sigma) = \sum_{j \ne 1} w_{1j}\, \sigma_j
  + \sum_{\text{edges } ij,\; i, j \ne 1} w_{ij}\, \sigma_i \sigma_j .
$$

The remaining problem is in the vertices 2 to n. Its h holds the weights of
the edges to vertex 1, and its J the edges among the remaining vertices.
Most pairs of vertices have no edge between them, so J is best stored as a
sparse matrix:

```python exec="on" source="above" result="text" session="gset"
import numpy as np
import scipy.sparse as sp


def maxcut_matrices(n_vertices, edges):
    """Max-Cut of a graph in Ising form, with vertex 1 fixed at σ = +1.

    Returns h and J for the vertices 2 to n_vertices, and the total weight W
    of all edges.
    """
    i, j, weight = (np.array(column) for column in zip(*edges))
    # the weight of each edge, stored at (i, j) and at (j, i), counted from 0
    adjacency = sp.csc_array(
        (np.concatenate([weight, weight]).astype(float),
         (np.concatenate([i, j]) - 1, np.concatenate([j, i]) - 1)),
        shape=(n_vertices, n_vertices),
    )
    h = adjacency[1:, [0]].toarray().ravel()   # the weight of the edge to vertex 1, or 0
    J = adjacency[1:, 1:]                      # the edges among the vertices 2 to n_vertices
    return h, J, float(weight.sum())


h, J, total_weight = maxcut_matrices(n_vertices, edges)
result = mfaoa.solve(mfaoa.IsingProblem(h, J))
print("cut found:", (total_weight - result.objective) / 2)
```

Pyomo found the same cut.

## The same problem in dimod

A binary quadratic model in dimod's `SPIN` vartype is the Ising form, so f
goes straight into `from_ising`. `fix_variable` puts vertex 1 in the set
σ₁ = +1, and the sampler minimizes the energy f:

```python exec="on" source="above" result="text" session="gset"
import dimod
from mfaoa.dimod import MFAOASampler

bqm = dimod.BinaryQuadraticModel.from_ising({}, {(i, j): weight for i, j, weight in edges})
bqm.fix_variable(1, 1)
energy = MFAOASampler().sample(bqm).first.energy
print("cut found:", (total_weight - energy) / 2)
```

## Sparse coupling matrices

MFAOA runs a fixed number of steps, 1000 by default. Each step multiplies J
by a vector. For a dense matrix that takes one multiplication per entry, for
a sparse matrix one per stored entry. Graph G55 shows the difference,
because it has many vertices but few edges:

```python exec="on" source="above" result="text" session="gset"
n_vertices, edges = read_gset("G55")
print("vertices:", n_vertices)
print("edges:   ", len(edges))
```

Solve it once with J stored as a sparse matrix and once with J stored as a
dense one:

```python exec="on" source="above" result="text" session="gset"
import time

h, J, total_weight = maxcut_matrices(n_vertices, edges)
dense_J = J.toarray()
sparse_problem = mfaoa.IsingProblem(h, J)
dense_problem = mfaoa.IsingProblem(h, dense_J)

start = time.perf_counter()
dense_result = mfaoa.solve(dense_problem)
dense_time = time.perf_counter() - start
start = time.perf_counter()
sparse_result = mfaoa.solve(sparse_problem)
sparse_time = time.perf_counter() - start

sparse_bytes = J.data.nbytes + J.indices.nbytes + J.indptr.nbytes
print(f"{'':24} {'dense':>12} {'sparse':>10}")
print(f"{'entries of J:':24} {dense_J.size:12d} {J.nnz:10d}")
print(f"{'memory for J (MB):':24} {dense_J.nbytes / 1e6:12.1f} {sparse_bytes / 1e6:10.2f}")
print(f"{'solve time (s):':24} {dense_time:12.3f} {sparse_time:10.3f}")
print(f"{'cut found:':24} {(total_weight - dense_result.objective) / 2:12.0f} "
      f"{(total_weight - sparse_result.objective) / 2:10.0f}")
```

Both find the same cut. The sparse matrix stores a thousandth of the
entries, needs 0.44 MB of memory instead of 200 MB, and makes the solve many
times faster.

The Pyomo interface and the dimod sampler make this choice for you: they
store the objective as a sparse matrix unless at least a quarter of all
pairs of variables appear in it together. An `IsingProblem` uses the matrix
you give it as it is.

## Comparison with the best known cuts

The loop below solves four Gset graphs with the default settings and
compares each cut with the best one published. G1, G22 and G43 are random
graphs: every pair of vertices is joined by an edge with the same
probability. G11 is a grid of 8 by 100 vertices that wraps around in both
directions, like the surface of a doughnut, and its edges have weights +1
and −1.

```python exec="on" source="above" result="text" session="gset"
def solve_gset(name):
    """Solve Max-Cut on the Gset graph `name` with the default settings.

    Returns the number of vertices and the cut found.
    """
    n_vertices, edges = read_gset(name)
    h, J, total_weight = maxcut_matrices(n_vertices, edges)
    result = mfaoa.solve(mfaoa.IsingProblem(h, J))
    return n_vertices, (total_weight - result.objective) / 2


# the best published cuts, from Benlic and Hao (2013), Table 2
best_known_cuts = {"G1": 11624, "G22": 13359, "G43": 6660, "G11": 564}

print(f"{'graph':<5} {'vertices':>9} {'cut':>8} {'best known':>11} {'gap':>7}")
for name, best_known_cut in best_known_cuts.items():
    vertices, cut = solve_gset(name)
    gap = 100 * (best_known_cut - cut) / best_known_cut    # in percent
    print(f"{name:<5} {vertices:9d} {cut:8.0f} {best_known_cut:11d} {gap:6.2f}%")
```

On the three random graphs, the cut found is within 0.7 % of the best
known. The grid G11 has the largest gap, about 2.5 %. The settings that
control a run, `p` and `tau`, are the subject of
[Schedules and the field scale](../manual/schedules.md).

## References

- C. Helmberg and F. Rendl, *A spectral bundle method for semidefinite
  programming*, SIAM Journal on Optimization **10**, 673–696 (2000),
  [doi:10.1137/S1052623497328987](https://doi.org/10.1137/S1052623497328987).
- U. Benlic and J.-K. Hao, *Breakout local search for the max-cut problem*,
  Engineering Applications of Artificial Intelligence **26**, 1162–1173 (2013),
  [doi:10.1016/j.engappai.2012.09.001](https://doi.org/10.1016/j.engappai.2012.09.001).
