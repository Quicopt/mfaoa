# The C library

The Python package calls a C library, which C programs can also use
directly. It has one function that solves a problem, and it allocates no
memory: the caller passes every array, including the arrays that receive
the solution. The library is in the
[source repository](https://github.com/Quicopt/mfaoa), in `src/` and
`include/`.

## The problem

The library minimizes or maximizes

$$
E(\sigma) = \sum_i h_i \sigma_i + \sum_{i < j} J_{ij}\, \sigma_i \sigma_j
$$

over variables σᵢ that are -1 or +1, where h is a vector and J a symmetric
matrix that is zero on the diagonal. A problem in variables xᵢ that are 0
or 1,

$$
f(x) = \sum_i c_i x_i + \sum_{i < j} q_{ij}\, x_i x_j ,
$$

takes this form with xᵢ = (1 + σᵢ) / 2 and

$$
h_i = \frac{c_i}{2} + \frac{1}{4} \sum_{j \ne i} q_{ij}, \qquad
J_{ij} = J_{ji} = \frac{q_{ij}}{4}, \qquad
f(x) = E(\sigma) + \sum_i \frac{c_i}{2} + \sum_{i < j} \frac{q_{ij}}{4} .
$$

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

The library is built and tested on Linux with OpenBLAS, on macOS with
Accelerate and with OpenBLAS, and on Windows without a BLAS.

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

This complete program, `examples/solve.c`, minimizes a problem in four
variables:

```c
--8<-- "examples/solve.c"
```

It prints

```
sigma = (-1, 1, 1, 1), E = -4.25
```

A solve takes three structs, all declared in `mfaoa.h`:

- `mfaoa_problem` holds h and J, and whether to minimize or maximize. J is
  either a dense matrix in column-major order, or a sparse one in
  compressed sparse columns (CSC) with 0-based indices. A sparse J stores
  both triangles, with the rows of each column in increasing order. The
  field `dgemv` can point to a BLAS `dgemv` of your own, in the Fortran
  calling convention, for the product with a dense J. NULL, the default,
  uses the BLAS the library is linked to.
- `mfaoa_schedule` holds the angles βₖ and γₖ of the p steps.
  `mfaoa_linear_schedule` fills them with the
  [linear schedule](manual/schedules.md#the-linear-schedule). 1000 steps
  with τ = 0.5 are a good start. More steps usually give a better solution,
  and the time grows in proportion to p.
- `mfaoa_workspace` holds the arrays that receive the final state and the
  solution `sigma`.

Each step multiplies J by a vector. A dense J costs one multiplication per
entry, through the BLAS. A sparse J costs one per stored entry.

## Return codes

`mfaoa_solve` returns `MFAOA_OK`, `MFAOA_INVALID_ARGUMENT`, or
`MFAOA_FLIP_SYMMETRIC`. The last one means that flipping a group of
variables, σᵢ → −σᵢ for every i in the group, leaves E unchanged. This
happens when J links the group to no other variable and h is zero on all of
the group, as in Max-Cut. The algorithm starts every variable halfway
between −1 and +1, and there E has zero slope in these variables, so they
would never move. The workspace then names one variable per group. Fix it
to either value: to fix σᵢ = s, remove variable i, add s Jⱼᵢ to hⱼ for every
other variable j, and add s hᵢ to the constant. Nothing is lost, because
flipping the group turns every solution into one with the same value.
[Symmetric problems and fixing variables](manual/symmetry.md) shows this in
Python.

## Versions

`MFAOA_VERSION` is the version of the header, and `mfaoa_version()` returns
the version of the library that is loaded. While the major version is 0, a
new minor version may change the structs and functions.
