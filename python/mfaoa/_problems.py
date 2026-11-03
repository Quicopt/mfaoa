"""The two ways to state a problem, the conversion between them, and schedules.

Both problem types keep the arrays they are given, without copying them,
when those are already float64 NumPy arrays. A sparse J is stored in
compressed sparse columns (CSC), the format of the C library; its index
arrays are copied once, to the 64-bit integers the library takes.
"""

import operator

import numpy as np
import scipy.sparse as sp

from . import _lib

_SENSES = {"min": 1, "max": -1}


def _check_sense(sense):
    """Raise a ValueError unless sense is "min" or "max"."""
    if sense not in _SENSES:
        raise ValueError(f"sense must be 'min' or 'max', got {sense!r}")


def _matrix(A):
    """A as a float64 matrix: a SciPy sparse array in canonical CSC form
    (sorted indices, no duplicates), or a contiguous NumPy array."""
    if sp.issparse(A):
        A = sp.csc_array(A, dtype=np.float64)
        if not A.has_canonical_format:
            A = A.copy()
            A.sum_duplicates()
        return A
    A = np.asarray(A, dtype=np.float64)
    if not (A.flags.c_contiguous or A.flags.f_contiguous):
        A = np.ascontiguousarray(A)
    return A


def _vector(v, name, n=None):
    """v as a contiguous float64 vector.

    Args:
        v: anything NumPy turns into a one-dimensional array.
        name: the argument's name, for messages.
        n: the length v must have, or None.

    Raises:
        ValueError: v is not one-dimensional, or its length is not n.
    """
    v = np.ascontiguousarray(v, dtype=np.float64)
    if v.ndim != 1:
        raise ValueError(f"{name} must be a vector, got an array of shape {v.shape}")
    if n is not None and v.shape[0] != n:
        raise ValueError(f"{name} must have one entry per variable, got {v.shape[0]} for n = {n}")
    return v


def _check_couplings(J, n):
    """Raise a ValueError unless J is n × n, symmetric and zero on the diagonal."""
    if J.shape != (n, n):
        raise ValueError(f"J must be {n}×{n} to match h, got {J.shape}")
    diagonal = J.diagonal()
    nonzero = np.flatnonzero(diagonal)
    if nonzero.size:
        i = int(nonzero[0])
        raise ValueError(
            f"J must be zero on the diagonal, but J[{i}, {i}] = {float(diagonal[i])}. "
            "Since σ[i]**2 = 1, a diagonal entry only adds a constant: add it to offset instead."
        )
    if sp.issparse(J):
        differ = (J != J.T).tocoo()
        pairs = list(zip(differ.row, differ.col))
    else:
        pairs = np.argwhere(J != J.T)
    if len(pairs):
        i, j = (int(k) for k in pairs[0])
        raise ValueError(
            f"J must be symmetric, but J[{i}, {j}] = {float(J[i, j])} "
            f"and J[{j}, {i}] = {float(J[j, i])}"
        )


def _sum_of_squares(A):
    """The sum of the squared entries of a dense or sparse matrix."""
    values = A.data if sp.issparse(A) else A.ravel(order="K")
    return float(np.dot(values, values))


class IsingProblem:
    """A binary quadratic problem in Ising form.

    The variables σ[i] are -1 or +1, and the objective, to be minimized (or,
    with ``sense="max"``, maximized), is

        h @ σ + Σ_{i<j} J[i, j] σ[i] σ[j] + offset.

    J must be symmetric with a zero diagonal. Since σ[i]**2 = 1, a diagonal
    entry would only add a constant; put it into `offset` instead. J can be
    a NumPy array or a SciPy sparse matrix. A sparse J is much faster when
    most entries are zero.

    Two keyword arguments tune the algorithm and rarely need to be set:

    - `delta`: one positive factor per variable on the angles `beta` of the
      schedule (see `Schedule`). The default, None, means 1 for every
      variable and takes a faster code path.
    - `scale`: the unit in which the schedule measures the slope of the
      objective (see `fieldscale`). The default makes the solution
      independent of the units of h and J; ``scale=1`` uses those units
      themselves.
    """

    def __init__(self, h, J, *, delta=None, offset=0.0, sense="min", scale=None):
        """Check and store the problem; see the class docstring.

        Raises:
            ValueError: an argument that does not describe a problem in Ising
                form, with a message that says which and why.
        """
        _check_sense(sense)
        h = _vector(h, "h")
        n = h.shape[0]
        if n < 1:
            raise ValueError("the problem needs at least one variable")
        J = _matrix(J)
        _check_couplings(J, n)
        if delta is not None:
            delta = _vector(delta, "delta", n)
            if not np.all(delta > 0):
                raise ValueError("all driver strengths delta[i] must be positive")
        if scale is None:
            s = np.sqrt((float(np.dot(h, h)) + _sum_of_squares(J)) / n)
            scale = 1.0 if s == 0 else float(s)  # 1 for an all-zero objective
        elif not scale > 0:
            raise ValueError(f"scale must be positive, got {scale}")
        self.n = n
        self.h = h
        self.J = J
        self.delta = delta
        self.offset = float(offset)
        self.sense = sense
        self._scale = float(scale)
        if sp.issparse(J):
            self._colptr = J.indptr.astype(np.int64)
            self._rowval = J.indices.astype(np.int64)

    @property
    def fieldscale(self):
        """The unit in which the schedule measures the slope of the objective.

        The algorithm moves each variable according to the slope of the
        objective in that variable, m[i] = h[i] + Σ_j J[i, j] z[j], where
        z[j], between -1 and +1, is the current value of variable j. In step
        k, the angle that this slope sets is 2 gamma[k] m[i] / s (see
        `Schedule`). Unless `scale` was given when the problem was built, s
        is the root mean square of m[i] over all variables i and over random
        values ±1 of z:

            s = sqrt((Σ_i h[i]**2 + Σ_ij J[i, j]**2) / n).

        Multiplying h and J by a constant multiplies s by the same constant,
        so the algorithm, and the solution, do not depend on the units of the
        objective.
        """
        return self._scale

    def objective(self, sigma):
        """The objective value at sigma, a vector of -1 and +1, including the offset."""
        s = np.asarray(sigma, dtype=np.float64)
        return float(self.h @ s + s @ (self.J @ s) / 2 + self.offset)

    def is_flip_symmetric(self):
        """Whether the problem has a group of variables that MFAOA cannot solve for.

        Flipping all variables of such a group at once (σ → -σ) leaves the
        objective unchanged. No term of the objective links a variable of the
        group to one outside it, and h is zero on every variable of the
        group, up to rounding errors relative to the largest coupling. When h
        is zero everywhere, as in Max-Cut, the whole problem is such a
        group. A single variable that appears in no term of the objective
        does not count: any value is optimal for it, and the solution sets it
        to +1.

        The algorithm starts every variable at 0, halfway between -1 and +1.
        There the objective has zero slope in the variables of such a group,
        so they would never move. `solve` raises a `FlipSymmetricError`
        instead, which names the groups. Fix one variable of each group to
        either value and solve for the others. This loses nothing: flipping
        the group turns every solution into one with the same objective
        value, so of each such pair, one solution remains.

        To fix σ[i] = s, remove variable i from the problem, add s * J[j, i]
        to h[j] for every other variable j, and add s * h[i] to the offset.
        """
        code, _ = self._run(np.zeros(1), np.zeros(1))
        return code == _lib.FLIP_SYMMETRIC

    def _run(self, beta, gamma):
        """mfaoa_solve on this problem; returns the code and the workspace."""
        sparse = sp.issparse(self.J)
        return _lib.solve(
            self.h,
            None if sparse else self.J,
            self._colptr if sparse else None,
            self._rowval if sparse else None,
            self.J.data if sparse else None,
            self.delta,
            self._scale,
            _SENSES[self.sense],
            beta,
            gamma,
        )

    def __repr__(self):
        """The size, storage and sense of the problem."""
        storage = "sparse" if sp.issparse(self.J) else "dense"
        return f"IsingProblem: {self.n} variables, {storage} couplings, sense = {self.sense!r}"


class QUBOProblem:
    """A binary quadratic problem in QUBO form.

    The variables x[i] are 0 or 1, and the objective, to be minimized (or,
    with ``sense="max"``, maximized), is

        x @ Q @ x + c @ x + constant.

    Q is used exactly as given: a symmetric matrix counts each off-diagonal
    pair twice, an upper-triangular matrix once. Diagonal entries are
    allowed; since x[i]**2 = x[i], they act as linear terms. Q can be a NumPy
    array or a SciPy sparse matrix. `solve` converts the problem to Ising
    form with x = (1 + σ) / 2 (see `to_ising`) and reports the solution both
    as x and as σ.
    """

    def __init__(self, Q, *, c=None, constant=0.0, sense="min"):
        """Check and store the problem; see the class docstring.

        Raises:
            ValueError: Q is not square or empty, c has the wrong length, or
                sense is not "min" or "max".
        """
        _check_sense(sense)
        Q = _matrix(Q)
        if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
            raise ValueError(f"Q must be square, got shape {Q.shape}")
        n = Q.shape[0]
        if n < 1:
            raise ValueError("the problem needs at least one variable")
        self.n = n
        self.Q = Q
        self.c = np.zeros(n) if c is None else _vector(c, "c", n)
        self.constant = float(constant)
        self.sense = sense

    def objective(self, x):
        """The objective value at x, a vector of 0 and 1, including the constant."""
        v = np.asarray(x, dtype=np.float64)
        return float(v @ (self.Q @ v) + self.c @ v + self.constant)

    def to_ising(self, *, delta=None, scale=None):
        """The same problem in Ising form, with x = (1 + σ) / 2.

        A pair term Q[i, j] x[i] x[j] + Q[j, i] x[j] x[i] becomes the
        coupling J[i, j] = (Q[i, j] + Q[j, i]) / 4 and contributes the same
        amount to both fields and to the offset. The weight of x[i] alone,
        c[i] + Q[i, i], goes half into h[i] and half into the offset. J is
        sparse if Q is.
        """
        n, Q = self.n, self.Q
        if sp.issparse(Q):
            Q = Q.tocoo()
            off = Q.row != Q.col
            i, j, v = Q.row[off], Q.col[off], Q.data[off] / 4
            # Both orientations of every entry; duplicates add up, which
            # gives (Q[i, j] + Q[j, i]) / 4 at (i, j) and at (j, i).
            J = sp.coo_array(
                (np.concatenate([v, v]), (np.concatenate([i, j]), np.concatenate([j, i]))),
                shape=(n, n),
            ).tocsc()
            J.sum_duplicates()
            J.eliminate_zeros()
        else:
            J = (Q + Q.T) / 4
            np.fill_diagonal(J, 0.0)
        linear = self.c + Q.diagonal()  # the weight of x[i] alone
        coupling_sums = J @ np.ones(n)  # Σ_j J[i, j]
        h = linear / 2 + coupling_sums
        offset = self.constant + linear.sum() / 2 + coupling_sums.sum() / 2
        return IsingProblem(h, J, delta=delta, offset=offset, sense=self.sense, scale=scale)

    def is_flip_symmetric(self):
        """Whether the problem has a group of variables that MFAOA cannot solve for.

        Flipping all variables of such a group at once (x → 1 - x) leaves the
        objective unchanged. See `IsingProblem.is_flip_symmetric` for why
        the algorithm cannot solve such a problem, and why fixing one variable
        of each group loses nothing.

        To fix x[i] = v, remove variable i from the problem, add
        v * (Q[i, j] + Q[j, i]) to c[j] for every other variable j, and add
        v * (Q[i, i] + c[i]) to the constant.
        """
        return self.to_ising().is_flip_symmetric()

    def __repr__(self):
        """The size, storage and sense of the problem."""
        storage = "sparse" if sp.issparse(self.Q) else "dense"
        return f"QUBOProblem: {self.n} variables, {storage} Q, sense = {self.sense!r}"


class Schedule:
    """The angles that drive the algorithm, one pair beta[k], gamma[k] per step k.

    The algorithm represents each variable σ[i] by a unit vector in three
    dimensions. The vector's z component, between -1 and +1, is the
    variable's current value. Every vector starts at (1, 0, 0), so every
    variable starts at 0, halfway between -1 and +1. Each step k turns every
    vector i twice:

    1. about the z axis by the angle 2 gamma[k] m[i] / s, where m[i] is the
       slope of the objective in variable i at the current values, and s is
       the problem's `fieldscale` (the opposite angle for a maximization);
    2. about the x axis by the angle 2 beta[k] delta[i], where delta[i] is 1
       unless the problem sets it.

    After the last step, σ[i] is the sign of the z component. beta and gamma
    must have the same length, the number of steps p. `linear_schedule`
    builds the usual choice; any other pair of vectors can be passed here.
    """

    def __init__(self, beta, gamma):
        """Check and store the angles.

        Raises:
            ValueError: beta and gamma are not vectors of the same, nonzero length.
        """
        beta = _vector(beta, "beta")
        gamma = _vector(gamma, "gamma")
        if beta.shape != gamma.shape:
            raise ValueError(
                f"beta and gamma must have one entry per step; got {beta.shape[0]} and {gamma.shape[0]}"
            )
        if beta.shape[0] == 0:
            raise ValueError("a schedule needs at least one step")
        self.beta = beta
        self.gamma = gamma

    def __len__(self):
        """The number of steps p."""
        return self.beta.shape[0]

    def __repr__(self):
        """The number of steps."""
        return f"Schedule: {len(self)} steps"


def linear_schedule(p, tau=0.5):
    """The usual schedule: over p steps, the angles gamma grow and the angles
    beta shrink linearly,

        gamma[k] = tau (k + 1) / p,    beta[k] = tau (1 - k / p),    k = 0, …, p - 1.

    This is Eq. 13 of Misra-Spieldenner et al., PRX Quantum 4, 030335 (2023).
    tau sets the size of the angles. Because gamma is measured in units of
    the problem's `fieldscale`, the same tau suits problems of any size and
    units; values between 0.25 and 0.5 work well on typical instances. With
    more steps, the angles change more slowly from one step to the next.
    That usually gives a better solution, and the time of a solve grows in
    proportion to p.
    """
    p = operator.index(p)
    if p < 1:
        raise ValueError(f"need at least one step, got p = {p}")
    return Schedule(*_lib.linear_schedule(p, float(tau)))


# The sparse product beats a dense one below about 25-30 % fill, measured for
# MFAOA.jl on an Apple M4 with OpenBLAS for n = 500 to 2000.
DENSE_FROM = 0.25


def coupling_storage(A):
    """Choose dense or sparse storage for a matrix of pair weights.

    The Pyomo interface and the dimod sampler collect the terms of an
    objective into a sparse matrix. The solve is faster with a dense one once
    the pairs that A couples, the pattern of A + A.T, fill a quarter of it.

    Args:
        A: a square SciPy sparse matrix.

    Returns:
        A as a dense NumPy array when that pattern fills at least DENSE_FROM
        of the matrix, and otherwise A itself.
    """
    n = A.shape[0]
    pattern = abs(A) + abs(A.T)
    return A.toarray() if pattern.nnz >= DENSE_FROM * n * n else A


def schedule_from(p=None, tau=None, beta=None, gamma=None):
    """The schedule that a front end's options ask for.

    The Pyomo interface and the dimod sampler take the same four options as
    `solve`.

    Args:
        p: the number of steps of the linear schedule (default 1000).
        tau: its step size (default 0.5).
        beta, gamma: an explicit schedule, used instead of the linear one.

    Returns:
        A `Schedule`: from beta and gamma when both are given, else the
        linear schedule with p and tau.

    Raises:
        ValueError: only one of beta and gamma, or a tau that is not positive.
    """
    if (beta is None) != (gamma is None):
        raise ValueError("Set both beta and gamma for a schedule of your own, or neither.")
    if beta is not None:
        return Schedule(beta, gamma)
    tau = 0.5 if tau is None else tau
    if not tau > 0:
        raise ValueError(f"tau must be positive, got {tau}")
    return linear_schedule(1000 if p is None else p, tau)
