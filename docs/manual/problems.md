# Problem forms

`mfaoa` states every problem in one of two forms: the QUBO form, in variables
that are 0 or 1, and the Ising form, in variables that are -1 or +1. The
Pyomo interface and the dimod sampler build these forms from a model. With
NumPy and SciPy, you state a form yourself and pass it to `mfaoa.solve`.

## The QUBO form

`QUBOProblem(Q, c=c, constant=constant, sense=sense)` states the problem

$$
\text{minimize} \quad x^\top Q x + c^\top x + \text{constant}
\qquad \text{over } x \in \{0, 1\}^n .
$$

The objective 2x₁x₂ − 3x₂x₃ − x₁ − x₂ + x₃ from the
[example on the home page](../index.md#example) has two pair terms, which go
into `Q`, and three linear terms, which go into `c`. Python counts from 0, so
x₁ is `x[0]`:

```python exec="on" source="above" result="text" session="problems"
import numpy as np
import mfaoa

Q = np.array([[0, 2, 0],      # 2 x1 x2
              [0, 0, -3],     # -3 x2 x3
              [0, 0, 0]])
c = np.array([-1, -1, 1])     # -x1 - x2 + x3
qubo = mfaoa.QUBOProblem(Q, c=c)
result = mfaoa.solve(qubo)

print("x:        ", result.x)
print("objective:", result.objective)
```

The product `x @ Q @ x` counts both `Q[i, j]` and `Q[j, i]` for a pair of
variables. A pair term can therefore go into one triangle of `Q`, as above,
or be split between the two triangles, as in this symmetric `Q`:

```python exec="on" source="above" result="text" session="problems"
Q_symmetric = np.array([[0, 1, 0],
                        [1, 0, -1.5],
                        [0, -1.5, 0]])
print(mfaoa.QUBOProblem(Q_symmetric, c=c).objective([0, 1, 1]))
```

Since x² = x for a binary variable, a diagonal entry `Q[i, i]` acts like a
linear term. To maximize instead, pass `sense="max"`. `objective` then
reports values as stated, without a change of sign.

## The Ising form

`IsingProblem(h, J, offset=offset, sense=sense)` states a problem in
variables σᵢ that are -1 or +1:

$$
\text{minimize} \quad h^\top \sigma + \sum_{i<j} J_{ij}\, \sigma_i \sigma_j + \text{offset}
\qquad \text{over } \sigma \in \{-1, +1\}^n .
$$

`J` must be symmetric and zero on the diagonal. Each pair appears twice in
`J` but counts once in the sum. Some problems are easier to state in this
form; [Max-Cut](../tutorials/maxcut-gset.md) is one.

Every QUBO has an Ising form, with x = (1 + σ) / 2. `QUBOProblem.to_ising`
computes it, and the objective values agree:

```python exec="on" source="above" result="text" session="problems"
ising = qubo.to_ising()
print("h:     ", ising.h)
print("J:")
print(ising.J)
print("offset:", ising.offset)
print("objective at σ = (-1, 1, 1):", ising.objective([-1, 1, 1]))
```

MFAOA works in the Ising form, and `solve` converts a QUBO first. The result
gives the solution in both forms.

## The result

`solve` returns a `Result` with these fields:

- `x`: the solution in the QUBO form, entries 0 and 1;
- `sigma`: the same solution in the Ising form, entries -1 and +1;
- `objective`: its objective value, in the problem's sense and including
  the constant;
- `objective_mf`: the objective at the values just before they are rounded
  to the solution; [Schedules and the field scale](schedules.md) explains how
  to read it;
- `state`: the final state of the algorithm.

For the QUBO above:

```python exec="on" source="above" result="text" session="problems"
print(result)
print("sigma:       ", result.sigma)
print("objective_mf:", result.objective_mf)
```
