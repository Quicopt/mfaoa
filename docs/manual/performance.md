# Performance

## Cost of a solve

Each step multiplies the coupling matrix J of the Ising form by a vector,
and then does a fixed amount of work per variable. For a dense matrix, the
product takes one multiplication per entry, n² in all; for a sparse matrix,
one per stored entry. The time of a solve grows in proportion to the number
of steps.

## Sparse and dense matrices

Most of the time goes into the product with J, so a sparse J pays off when
most of its entries are zero. The table compares the two for 100 steps on
random problems with 2000 variables, in which a growing share of all pairs
of variables is coupled:

```python exec="on" source="above" result="text" session="performance"
import time

import numpy as np
import scipy.sparse as sp
import mfaoa


def random_couplings(n, share):
    """h and J of a random Ising problem with n variables, in which about the
    given share of all pairs is coupled, with Gaussian weights."""
    rng = np.random.default_rng(1)
    upper = sp.triu(sp.random_array((n, n), density=share, rng=rng,
                                    data_sampler=rng.standard_normal), k=1)
    return rng.standard_normal(n), (upper + upper.T).tocsc()


def best_time(problem, p):
    """The best of three timings of a solve with p steps, in seconds."""
    times = []
    for _ in range(3):
        start = time.perf_counter()
        mfaoa.solve(problem, p=p)
        times.append(time.perf_counter() - start)
    return min(times)


print(f"{'pairs coupled':>14} {'stored':>10} {'dense (s)':>10} {'sparse (s)':>11}")
for share in (0.001, 0.01, 0.1, 0.3):
    h, J = random_couplings(2000, share)
    sparse_problem = mfaoa.IsingProblem(h, J)
    dense_problem = mfaoa.IsingProblem(h, J.toarray())
    print(f"{100 * share:13.1f}% {J.nnz:10d} "
          f"{best_time(dense_problem, 100):10.4f} {best_time(sparse_problem, 100):11.4f}")
```

The share at which sparse and dense take equally long depends on the
computer; the table shows it for the computer that built this page. The
Pyomo interface and the dimod sampler store the objective as a dense matrix
once at least a quarter of all pairs of variables appear in it, and as a
sparse one below that. An `IsingProblem` or `QUBOProblem` uses the matrix
you give it.

## The BLAS

The product with a dense matrix runs through the BLAS library that SciPy
uses, the routine `dgemv`. SciPy's wheels bring their own BLAS: OpenBLAS on
most platforms, and Apple's Accelerate on Macs with Apple silicon. On the
computer that built this page, SciPy uses:

```python exec="on" source="above" result="text" session="performance"
import scipy

print(scipy.show_config(mode="dicts")["Build Dependencies"]["blas"]["name"])
```

For dense problems that fit into the processor's cache, Accelerate is
several times faster than OpenBLAS. A SciPy built with another BLAS, such as
MKL, speeds up dense problems in the same way, without changes to your
code. Sparse problems do not use the BLAS.
