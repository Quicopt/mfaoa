# Symmetric problems and fixing variables

## Max-Cut and other symmetric objectives

Max-Cut splits the vertices of a graph into two sets, such that the edges
between the two sets carry as much total weight as possible. In the
triangle below, the edges 1–2, 2–3 and 1–3 weigh 1, 2 and 3. The largest
cut separates vertex 3 from the other two and cuts a weight of 5. With one
variable per vertex, MFAOA rejects this problem:

=== "Pyomo"

    ```python exec="on" source="above" result="text" session="symmetry-pyomo"
    import pyomo.environ as pyo
    import mfaoa
    import mfaoa.pyomo

    weight = {(1, 2): 1, (2, 3): 2, (1, 3): 3}
    model = pyo.ConcreteModel()
    model.x = pyo.Var([1, 2, 3], domain=pyo.Binary)  # x[v]: the set, 0 or 1, of vertex v
    model.cut = pyo.Objective(
        expr=sum(w * (model.x[i] + model.x[j] - 2 * model.x[i] * model.x[j])
                 for (i, j), w in weight.items()),
        sense=pyo.maximize,
    )
    try:
        pyo.SolverFactory("mfaoa").solve(model)
    except mfaoa.FlipSymmetricError as error:
        print(error)
    ```

=== "dimod"

    ```python exec="on" source="above" result="text" session="symmetry-dimod"
    import dimod
    import mfaoa
    from mfaoa.dimod import MFAOASampler

    weight = {(1, 2): 1, (2, 3): 2, (1, 3): 3}
    bqm = dimod.BinaryQuadraticModel(dimod.BINARY)   # energy: minus the cut
    for (i, j), w in weight.items():
        bqm.add_linear(i, -w)
        bqm.add_linear(j, -w)
        bqm.add_quadratic(i, j, 2 * w)
    try:
        MFAOASampler().sample(bqm)
    except mfaoa.FlipSymmetricError as error:
        print(error)
    ```

=== "NumPy"

    ```python exec="on" source="above" result="text" session="symmetry-numpy"
    import numpy as np
    import mfaoa

    weight = {(0, 1): 1, (1, 2): 2, (0, 2): 3}   # vertices 0, 1, 2
    Q = np.zeros((3, 3))
    c = np.zeros(3)
    for (i, j), w in weight.items():   # w (x_i + x_j - 2 x_i x_j)
        Q[i, j] -= 2 * w
        c[i] += w
        c[j] += w
    qubo = mfaoa.QUBOProblem(Q, c=c, sense="max")
    try:
        mfaoa.solve(qubo)
    except mfaoa.FlipSymmetricError as error:
        print(error)
    ```

Exchanging the two sets of a cut cuts the same edges, so the objective does
not change when every x is replaced by 1 − x. MFAOA starts every variable
at 1/2 and moves it according to the slope of the objective. The point
where all variables are 1/2 lies halfway between every x and its flipped
copy 1 − x. For an objective that takes the same value at both, the slope
there is zero, so no variable would ever move.

Fixing one variable solves this. Of each pair x and 1 − x, exactly one has
x₁ = 0, and both have the same objective value. Fixing x₁ to 0 therefore
keeps the best objective value:

=== "Pyomo"

    ```python exec="on" source="above" result="text" session="symmetry-pyomo"
    model.x[1].fix(0)
    pyo.SolverFactory("mfaoa").solve(model)

    print("x:  ", [int(pyo.value(model.x[v])) for v in model.x])
    print("cut:", pyo.value(model.cut))
    ```

=== "dimod"

    ```python exec="on" source="above" result="text" session="symmetry-dimod"
    bqm.fix_variable(1, 0)   # removes vertex 1 from the model
    print(MFAOASampler().sample(bqm))
    ```

=== "NumPy"

    The NumPy form has no fixing of its own: a fixed variable is removed from
    the problem, as [the last section](#fixing-variables-by-hand) shows.

The error's attribute `groups` lists the variables of each group that needs
a fixed variable.

## Objectives in separate parts

The same holds for a group of variables that no term of the objective links
to the other variables: if flipping the group does not change the
objective, one of its variables must be fixed. Max-Cut of a graph in two
separate pieces needs one fixed vertex in each piece. MFAOA finds such
groups itself and names a variable to fix in each:

```python exec="on" source="above" result="text" session="symmetry-parts"
import pyomo.environ as pyo
import mfaoa
import mfaoa.pyomo

edges = [(1, 2), (3, 4)]   # two pieces: vertices 1, 2 and vertices 3, 4
model = pyo.ConcreteModel()
model.x = pyo.Var([1, 2, 3, 4], domain=pyo.Binary)
model.cut = pyo.Objective(
    expr=sum(model.x[i] + model.x[j] - 2 * model.x[i] * model.x[j] for i, j in edges),
    sense=pyo.maximize,
)
model.x[1].fix(0)
try:
    pyo.SolverFactory("mfaoa").solve(model)
except mfaoa.FlipSymmetricError as error:
    print(error)
```

After fixing x₃ as well, both edges are cut:

```python exec="on" source="above" result="text" session="symmetry-parts"
model.x[3].fix(0)
pyo.SolverFactory("mfaoa").solve(model)

print("x:  ", [int(pyo.value(model.x[v])) for v in model.x])
print("cut:", pyo.value(model.cut))
```

A variable that appears in no term of the objective needs no fixing. Any
value is optimal for it, and MFAOA sets it to 1.

## Fixing variables by hand

In the Ising form, an objective made of pair terms σᵢσⱼ alone does not
change when every σᵢ changes sign. The symmetry shows as h = 0, everywhere
or on a group of variables that no coupling links to the others.
`is_flip_symmetric` tests for it.

To fix σᵢ to the value s, remove variable i from the problem. Its pair terms
with the other variables become linear terms, and its own linear term
becomes a constant. For the weighted triangle, the Ising objective
σ₁σ₂ + 2σ₂σ₃ + 3σ₁σ₃ is smallest for the largest cut:

```python exec="on" source="above" result="text" session="symmetry-ising"
import numpy as np
import mfaoa


def fix_variable(h, J, offset, i, s):
    """h, J and offset of the problem with σ[i] fixed to s, in the other variables."""
    others = np.arange(len(h)) != i
    h_others = h[others] + s * J[others, i]
    return h_others, J[np.ix_(others, others)], offset + s * h[i]


J = np.array([[0.0, 1.0, 3.0],
              [1.0, 0.0, 2.0],
              [3.0, 2.0, 0.0]])        # the weights of the triangle's edges
h = np.zeros(3)
triangle = mfaoa.IsingProblem(h, J)
print("flip-symmetric:", triangle.is_flip_symmetric())

h_rest, J_rest, offset = fix_variable(h, J, 0.0, 0, 1)   # σ of vertex 1 is +1
result = mfaoa.solve(mfaoa.IsingProblem(h_rest, J_rest, offset=offset))
print("σ of vertices 2 and 3:", result.sigma)
print("objective:            ", result.objective)
```

Vertex 3 ends up alone, on the other side. A product σᵢσⱼ is −1 for a cut
edge and +1 for the others, so the objective is the total weight, 6, minus
twice the cut: −4 for the cut of weight 5.

In the QUBO form, fixing xᵢ to the value v adds v (Qᵢⱼ + Qⱼᵢ) to cⱼ for every
other variable j, and v (Qᵢᵢ + cᵢ) to the constant.
