# NumPy and SciPy

With NumPy and SciPy, you state a problem as arrays, in one of the two
[problem forms](problems.md), and pass it to `mfaoa.solve`. The vectors `c`
and `h` can be anything that NumPy turns into a one-dimensional array. The
matrices `Q` and `J` can be NumPy arrays or SciPy sparse matrices.

## Dense and sparse matrices

A sparse matrix stores only its nonzero entries, and the solve then only
visits those. When most pairs of variables have no term in the objective, a
sparse matrix makes the solve much faster; [Performance](performance.md)
gives numbers. Every SciPy sparse format is accepted. `mfaoa` converts it to
compressed sparse columns (CSC), the format of the C library.

The two storages give the same solution. The problem below has 200
variables, and 2 % of the entries of `Q` are nonzero:

```python exec="on" source="above" result="text" session="numpy"
import numpy as np
import scipy.sparse as sp
import mfaoa

rng = np.random.default_rng(1)
n = 200
Q_sparse = sp.random_array((n, n), density=0.02, rng=rng, format="csr")
c = rng.normal(size=n)

result_sparse = mfaoa.solve(mfaoa.QUBOProblem(Q_sparse, c=c))
result_dense = mfaoa.solve(mfaoa.QUBOProblem(Q_sparse.toarray(), c=c))

print("same solution:", np.array_equal(result_sparse.x, result_dense.x))
print("objective:    ", result_sparse.objective)
```

## Changing arrays after building a problem

A problem may keep the arrays it is given, without copying them, when they
are already NumPy arrays of 64-bit floats. A change to such an array after
the problem was built can therefore reach the problem, or not, depending on
the array. Build a new problem after changing an array. Building one is
cheap: it checks the arrays and copies only what it needs.

## Checks

A problem is checked when it is built. An array that does not fit the form
is rejected with a message that names the entry:

```python exec="on" source="above" result="text" session="numpy"
J = np.array([[1.0, 0.0],
              [0.0, 0.0]])
try:
    mfaoa.IsingProblem([1.0, 0.0], J)
except ValueError as error:
    print(error)
```
