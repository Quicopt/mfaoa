"""Binary quadratic optimization with the mean-field approximate optimization
algorithm (mean-field AOA) of Misra-Spieldenner et al., PRX Quantum 4,
030335 (2023).

State a problem as an `IsingProblem` (variables -1 and +1) or a
`QUBOProblem` (variables 0 and 1), then call `solve`:

    >>> import numpy as np, mfaoa
    >>> J = np.array([[0.0, 1.0], [1.0, 0.0]])
    >>> mfaoa.solve(mfaoa.IsingProblem([0.5, 0.0], J)).sigma
    array([-1,  1], dtype=int8)
"""

from ._lib import version as _version
from ._problems import IsingProblem, QUBOProblem, Schedule, linear_schedule
from ._solve import Result, SpinState, solve
from ._symmetry import FlipSymmetricError

__version__ = _version()

__all__ = [
    "FlipSymmetricError",
    "IsingProblem",
    "QUBOProblem",
    "Result",
    "Schedule",
    "SpinState",
    "linear_schedule",
    "solve",
]
