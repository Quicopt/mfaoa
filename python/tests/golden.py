"""Reading the reference runs in test/golden.txt, which test/check.c also uses.

Each case holds a problem, a schedule and the expected result, computed with
MFAOA.jl 0.1: the return code, and either the final z components and the
solution sigma, or, for a flip-symmetric problem, the groups and flags the
workspace reports.
"""

from pathlib import Path

import numpy as np

GOLDEN = Path(__file__).resolve().parents[2] / "test" / "golden.txt"
HEADER = Path(__file__).resolve().parents[2] / "include" / "mfaoa.h"

_INTEGERS = {"colptr", "rowval", "group", "sigma"}


def read_cases(path=GOLDEN):
    """The cases of the reference file, each as a dict of its fields and arrays."""
    tokens = path.read_text(encoding="utf-8").split()
    cases, k = [], 0
    while k < len(tokens):
        assert tokens[k] == "case"
        case = {"name": tokens[k + 1]}
        k += 2
        while tokens[k] in ("n", "format", "sense", "scale", "delta", "expect"):
            case[tokens[k]] = tokens[k + 1]
            k += 2
        for key in ("n", "sense", "delta", "expect"):
            case[key] = int(case[key])
        case["scale"] = float(case["scale"])
        case["has_delta"] = case.pop("delta") == 1
        while True:
            label, length = tokens[k], int(tokens[k + 1])
            values = tokens[k + 2 : k + 2 + length]
            k += 2 + length
            dtype = np.int64 if label in _INTEGERS else np.float64
            case[label] = np.array(values, dtype=dtype)
            if label == "sigma":
                case["sigma"] = case["sigma"].astype(np.int8)
                break
        n = case["n"]
        if "J" in case:  # column-major n × n
            case["J"] = np.ascontiguousarray(case["J"].reshape(n, n, order="F"))
        cases.append(case)
    return cases
