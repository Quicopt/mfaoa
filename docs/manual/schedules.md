# Schedules and the field scale

## The algorithm

MFAOA works on the [Ising form](problems.md#the-ising-form) of a problem, in
variables σᵢ that are -1 or +1. It represents each variable by an arrow of
length 1 in three dimensions. The height of the arrow, its z component, is
the variable's current value, between -1 and +1. Every arrow starts
horizontal, pointing along the x axis, so every value starts at 0.

MFAOA then runs a fixed number of steps. Step k turns every arrow twice:

1. about the vertical z axis, by an angle proportional to γₖ and to the
   slope of the objective in that variable;
2. about the x axis, by the angle 2βₖ.

After the last step, each variable is rounded to the sign of its arrow's
height. The two sequences of angles, β and γ, form the schedule.

## The linear schedule

Unless you pass a schedule of your own, MFAOA uses the linear schedule
`linear_schedule(p, tau)` with p = 1000 steps and step size τ = 0.5. Its
angles, numbered from k = 0 as in Python, are

$$
\gamma_k = \tau \, \frac{k + 1}{p}, \qquad
\beta_k = \tau \left(1 - \frac{k}{p}\right), \qquad k = 0, \dots, p - 1 .
$$

With four steps, for example, γ grows and β shrinks like this:

```python exec="on" source="above" result="text" session="schedules"
import mfaoa

schedule = mfaoa.linear_schedule(4, 0.5)
print("beta: ", schedule.beta)
print("gamma:", schedule.gamma)
```

`mfaoa.solve(problem, p=p, tau=tau)` sets the two numbers. The Pyomo
interface takes them as the options `"p"` and `"tau"`, and the dimod sampler
as the parameters `p` and `tau`. A step size between 0.25 and 0.5 suits most
problems.

## Number of steps

The time a solve takes grows in proportion to the number of steps, and the
solution usually improves with it. On a random problem with 2000
variables:

```python exec="on" source="above" result="text" session="schedules"
import time

import numpy as np
import scipy.sparse as sp

# A random problem in Ising form: 2000 variables, each coupled to about six
# others with Gaussian weights, and weak random linear terms.
rng = np.random.default_rng(1)
n = 2000
upper = sp.triu(sp.random_array((n, n), density=6 / n, rng=rng,
                                data_sampler=rng.standard_normal), k=1)
J = (upper + upper.T).tocsc()
h = 0.1 * rng.standard_normal(n)
problem = mfaoa.IsingProblem(h, J)

print(f"{'p':>6} {'objective':>12} {'objective_mf':>14} {'time (s)':>9}")
for p in (10, 100, 1000, 10000):
    start = time.perf_counter()
    result = mfaoa.solve(problem, p=p)
    seconds = time.perf_counter() - start
    print(f"{p:6d} {result.objective:12.2f} {result.objective_mf:14.2f} {seconds:9.4f}")
```

`objective_mf` is the objective at the values just before rounding. When it
is far from `objective`, many values were still far from -1 and +1 at the
end of the run, and more steps are likely to give a better solution.

## Field scale

In step k, the first turn of arrow i has the angle 2γₖ mᵢ / s. The number
mᵢ is the slope of the objective in variable i. The number s is the
problem's field scale, `problem.fieldscale`, the typical size of such a
slope: the root mean square of mᵢ over all variables and over random values
±1 of the others. Because the slope is divided by s, multiplying the
objective by a constant changes neither the run nor the solution:

```python exec="on" source="above" result="text" session="schedules"
scaled = mfaoa.IsingProblem(1000 * h, 1000 * J)
print("field scale:        ", problem.fieldscale)
print("field scale, scaled:", scaled.fieldscale)
print("same solution:      ",
      np.array_equal(mfaoa.solve(scaled).sigma, mfaoa.solve(problem).sigma))
```

One step size τ therefore suits problems of any size and in any units. To
measure the slope in the units of the objective itself, build the problem
with `scale=1`; the angles of the schedule then have to suit those units.

## Custom schedules

Any two sequences of equal length make a schedule. Pass them as
`mfaoa.Schedule(beta, gamma)` to `solve`, as the options `"beta"` and
`"gamma"` in Pyomo, or as the parameters `beta` and `gamma` of the dimod
sampler. In the following schedule, γ grows quadratically instead of
linearly:

```python exec="on" source="above" result="text" session="schedules"
p = 1000
k = np.arange(p)
beta = 0.5 * (1 - k / p)          # as in linear_schedule
gamma = 0.5 * ((k + 1) / p) ** 2
result = mfaoa.solve(problem, mfaoa.Schedule(beta, gamma))
print(f"objective: {result.objective:.2f}")
```

`IsingProblem` also accepts the keyword `delta`: one positive factor per
variable on the angles β. The default, 1 for every variable, rarely needs
changing.
