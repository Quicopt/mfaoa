# dimod

`mfaoa.dimod` provides `MFAOASampler`, a sampler for the binary quadratic
models (BQMs) of D-Wave's dimod package. It comes with
`pip install "mfaoa[dimod]"`.

## Sampling

The sampler takes a BQM in either vartype, `SPIN` (variables -1 and +1) or
`BINARY` (variables 0 and 1). It returns a sample set in the BQM's own
vartype and variable labels. Like every dimod sampler, it minimizes the
energy:

```python exec="on" source="above" result="text" session="dimod"
import dimod
from mfaoa.dimod import MFAOASampler

bqm = dimod.BinaryQuadraticModel.from_ising(
    {"a": 1, "b": -1},                     # a - b
    {("a", "b"): -0.5, ("b", "c"): 1},     # -0.5 a b + b c
)
sampleset = MFAOASampler().sample(bqm)
print(sampleset)
```

The lowest energy, -2.5, belongs to a = -1, b = +1, c = -1. The sample set
holds one sample: MFAOA is deterministic, so a second sample would repeat
the first. `sampleset.first.sample` and `sampleset.first.energy` give the
sample and its energy.

## Parameters

| Parameter | Meaning | Default |
|:--- |:--- |:--- |
| `p` | the number of steps that MFAOA runs | 1000 |
| `tau` | the step size τ of the linear schedule | 0.5 |
| `beta`, `gamma` | a schedule of your own, as two sequences of equal length; used instead of the linear schedule when both are set | not set |

[Schedules and the field scale](schedules.md) explains what these settings
do. More steps usually give a better solution and take proportionally
longer. The `info` of the sample set records the number of steps and the
time of the solve, in seconds:

```python exec="on" source="above" result="text" session="dimod"
sampleset = MFAOASampler().sample(bqm, p=200)
print(sampleset.info["steps"])
```

As dimod's conventions ask, the sampler ignores other keyword arguments,
with a warning.

## Symmetric models

A BQM whose energy does not change when every variable is flipped raises a
`FlipSymmetricError`; the message names a variable to fix with
`bqm.fix_variable`. Max-Cut is such a model, and
[Symmetric problems and fixing variables](symmetry.md) explains what to do
about it.

## Storage of the model

MFAOA stores the interactions of a BQM as a sparse matrix, unless at least a
quarter of all pairs of variables interact. Then it stores a dense matrix.
[Performance](performance.md) explains why this matters for the run time.
