"""The C library through the binding, on the reference runs of test/golden.txt."""

import re

import mfaoa
import numpy as np
import pytest
import scipy.sparse as sp
from golden import HEADER, read_cases
from mfaoa import _lib

CASES = read_cases()
IDS = [c["name"] for c in CASES]


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_binding_reproduces_reference_runs(case):
    """The binding, called with the raw arrays, reproduces every reference run."""
    dense = "J" in case
    code, ws = _lib.solve(
        case["h"],
        case["J"] if dense else None,
        None if dense else case["colptr"],
        None if dense else case["rowval"],
        None if dense else case["nzval"],
        case.get("delta"),
        case["scale"],
        case["sense"],
        case["beta"],
        case["gamma"],
    )
    assert code == case["expect"]
    if code == _lib.OK:
        np.testing.assert_allclose(ws["z"], case["z"], rtol=0, atol=1e-8)
        np.testing.assert_array_equal(ws["sigma"], case["sigma"])
    elif code == _lib.FLIP_SYMMETRIC:
        np.testing.assert_array_equal(ws["group"], case["group"])
        np.testing.assert_array_equal(ws["sigma"], case["sigma"])


def _problem(case):
    """The case as an IsingProblem, built through the public API."""
    n = case["n"]
    if "J" in case:
        J = case["J"]
    else:
        J = sp.csc_array((case["nzval"], case["rowval"], case["colptr"]), shape=(n, n))
    sense = "min" if case["sense"] == 1 else "max"
    scale = case["scale"] if case["scale"] > 0 else None
    return mfaoa.IsingProblem(case["h"], J, delta=case.get("delta"), sense=sense, scale=scale)


@pytest.mark.parametrize("case", [c for c in CASES if c["expect"] == _lib.OK], ids=lambda c: c["name"])
def test_solve_reproduces_reference_runs(case):
    """solve reproduces the reference runs that end with a solution."""
    result = mfaoa.solve(_problem(case), mfaoa.Schedule(case["beta"], case["gamma"]))
    np.testing.assert_allclose(result.state.z, case["z"], rtol=0, atol=1e-8)
    np.testing.assert_array_equal(result.sigma, case["sigma"])
    np.testing.assert_array_equal(result.x, (case["sigma"] + 1) // 2)


@pytest.mark.parametrize("case", [c for c in CASES if c["expect"] == _lib.FLIP_SYMMETRIC], ids=lambda c: c["name"])
def test_solve_reports_reference_groups(case):
    """solve reports the groups of the flip-symmetric reference runs."""
    problem = _problem(case)
    assert problem.is_flip_symmetric()
    with pytest.raises(mfaoa.FlipSymmetricError) as error:
        mfaoa.solve(problem, mfaoa.Schedule(case["beta"], case["gamma"]))
    for group in error.value.groups:
        assert case["sigma"][group[0]] == 1
        np.testing.assert_array_equal(case["group"][group], group[0])
    flagged = np.flatnonzero(case["sigma"] == 1)
    assert [g[0] for g in error.value.groups] == flagged.tolist()


@pytest.mark.parametrize("case", [c for c in CASES if c["expect"] == _lib.INVALID_ARGUMENT], ids=lambda c: c["name"])
def test_python_rejects_what_the_library_rejects(case):
    """Problems that the C library rejects are rejected by Python first."""
    with pytest.raises(ValueError):
        _problem(case)


def test_linear_schedule_matches_reference():
    """linear_schedule reproduces the reference schedule exactly."""
    case = CASES[0]  # the linear schedule with p = 40 and tau = 0.5
    schedule = mfaoa.linear_schedule(len(case["beta"]), 0.5)
    np.testing.assert_array_equal(schedule.beta, case["beta"])
    np.testing.assert_array_equal(schedule.gamma, case["gamma"])


def test_versions_agree():
    """The header, the loaded library and the package report the same version."""
    header = re.search(r'#define MFAOA_VERSION "([^"]+)"', HEADER.read_text()).group(1)
    assert _lib.version() == header == _lib.HEADER_VERSION == mfaoa.__version__
