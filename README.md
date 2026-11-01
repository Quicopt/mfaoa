# mfaoa

mfaoa solves binary quadratic problems without constraints with the
mean-field approximate optimization algorithm (mean-field AOA) of
Misra-Spieldenner et al. (2023). It is a C library with one function that
solves a problem. The library allocates no memory: the caller passes every
array, including the one that receives the solution.

## The problem

mfaoa minimizes or maximizes

$$E(\sigma) = \sum_i h_i \sigma_i + \sum_{i<j} J_{ij}\, \sigma_i \sigma_j
\qquad \text{over } \sigma_i \in \{-1, +1\},$$

where $h$ is a vector and $J$ a symmetric matrix that is zero on the
diagonal. A problem in variables $x_i \in \{0, 1\}$,

$$f(x) = \sum_i c_i x_i + \sum_{i<j} q_{ij}\, x_i x_j ,$$

takes this form with $x_i = (1 + \sigma_i)/2$:

$$h_i = \frac{c_i}{2} + \frac{1}{4} \sum_{j \ne i} q_{ij}, \qquad
J_{ij} = J_{ji} = \frac{q_{ij}}{4}, \qquad
f(x) = E(\sigma) + \sum_i \frac{c_i}{2} + \sum_{i<j} \frac{q_{ij}}{4}.$$

The algorithm runs a fixed number of steps $p$ and returns a good solution,
without a proof that no better one exists.

## Building

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

## Using the library

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

## Reference

A. Misra-Spieldenner, T. Bode, P. K. Schuhmacher, T. Stollenwerk, D. Bagrets
and F. K. Wilhelm, *Mean-Field Approximate Optimization Algorithm*, PRX
Quantum **4**, 030335 (2023),
[doi:10.1103/PRXQuantum.4.030335](https://doi.org/10.1103/PRXQuantum.4.030335).

## Licence

Apache License 2.0, see `LICENSE`.
