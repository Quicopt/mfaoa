"""The C library: loading it, and calling mfaoa_solve through ctypes.

The structs below mirror include/mfaoa.h of the version HEADER_VERSION.
Nothing checks the two against each other when the package is built, so the
import compares the loaded library's version with HEADER_VERSION, and the
reference runs in the tests exercise every field.

The library in the wheels is built without a BLAS. For dense couplings it
calls SciPy's dgemv, whose address this module passes in every problem.
"""

import ctypes
import functools
import re
import sys
from pathlib import Path

import numpy as np

HEADER_VERSION = "0.1.0"

# The return codes of mfaoa_solve.
OK = 0
FLIP_SYMMETRIC = 1
INVALID_ARGUMENT = 2

_int64 = ctypes.c_int64
_ptr = ctypes.c_void_p


class _Problem(ctypes.Structure):
    """mfaoa_problem of mfaoa.h; pointers are passed as plain addresses."""
    _fields_ = [
        ("n", _int64),
        ("h", _ptr),
        ("J", _ptr),
        ("colptr", _ptr),
        ("rowval", _ptr),
        ("nzval", _ptr),
        ("delta", _ptr),
        ("scale", ctypes.c_double),
        ("sense", ctypes.c_int32),
        ("dgemv", _ptr),
    ]


class _Schedule(ctypes.Structure):
    """mfaoa_schedule of mfaoa.h."""
    _fields_ = [("p", _int64), ("beta", _ptr), ("gamma", _ptr)]


class _Workspace(ctypes.Structure):
    """mfaoa_workspace of mfaoa.h."""
    _fields_ = [(name, _ptr) for name in ("x", "y", "z", "m", "sigma", "group")]


def _library_path():
    """The path of the C library inside the installed package, named as CMake names it on this platform."""
    names = {"win32": "mfaoa.dll", "darwin": "libmfaoa.dylib"}
    return Path(__file__).with_name(names.get(sys.platform, "libmfaoa.so"))


def _load():
    """Load the C library and declare its functions.

    Returns:
        The ctypes library.

    Raises:
        ImportError: the library's major and minor version differ from
            HEADER_VERSION, so the structs above may not match it.
    """
    lib = ctypes.CDLL(str(_library_path()))
    lib.mfaoa_solve.restype = ctypes.c_int32
    lib.mfaoa_solve.argtypes = [
        ctypes.POINTER(_Problem),
        ctypes.POINTER(_Schedule),
        ctypes.POINTER(_Workspace),
    ]
    lib.mfaoa_linear_schedule.restype = ctypes.c_int32
    lib.mfaoa_linear_schedule.argtypes = [_int64, ctypes.c_double, _ptr, _ptr]
    lib.mfaoa_version.restype = ctypes.c_char_p
    lib.mfaoa_version.argtypes = []
    loaded = lib.mfaoa_version().decode()
    if loaded.split(".")[:2] != HEADER_VERSION.split(".")[:2]:
        raise ImportError(
            f"mfaoa loaded the C library version {loaded} from {_library_path()}, "
            f"but its Python code was written for version {HEADER_VERSION}. "
            "Reinstall the package: pip install --force-reinstall mfaoa"
        )
    return lib


_lib = _load()


def version():
    """The version of the loaded C library, such as "0.1.0"."""
    return _lib.mfaoa_version().decode()


# SciPy's dgemv takes 32-bit integers. A SciPy built with 64-bit BLAS
# integers would show another type than "int" in this signature.
_DGEMV_SIGNATURE = (
    "void (char *, int *, int *, double *, double *, int *, "
    "double *, int *, double *, double *, int *)"
)


@functools.cache
def scipy_dgemv():
    """The address of SciPy's dgemv, in the form mfaoa_problem.dgemv takes."""
    from scipy.linalg import cython_blas

    capsule = cython_blas.__pyx_capi__["dgemv"]
    get_name = ctypes.pythonapi.PyCapsule_GetName
    get_name.restype = ctypes.c_char_p
    get_name.argtypes = [ctypes.py_object]
    get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
    get_pointer.restype = ctypes.c_void_p
    get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    name = get_name(capsule)
    # Cython writes SciPy's own name for double, such as
    # __pyx_t_5scipy_6linalg_11cython_blas_d.
    signature = re.sub(r"__pyx_t_\w+_d\b", "double", name.decode())
    if signature != _DGEMV_SIGNATURE:
        raise RuntimeError(
            "mfaoa needs SciPy's BLAS with 32-bit integers for dense couplings, "
            f"but this SciPy's dgemv has the signature {signature}. "
            "Pass J as a scipy.sparse matrix, or install SciPy from PyPI."
        )
    return get_pointer(capsule, name)


def _address(a):
    """The address of an array's data, or None (NULL) for None."""
    return None if a is None else a.ctypes.data


def solve(h, J, colptr, rowval, nzval, delta, scale, sense, beta, gamma):
    """Run mfaoa_solve on arrays already in the form mfaoa.h describes.

    J is a dense matrix (C- or Fortran-ordered; it must be symmetric) or
    None, in which case colptr, rowval and nzval hold the couplings in
    compressed sparse columns. scale <= 0 lets the library compute the
    field scale. Returns the code and the workspace arrays.
    """
    n = h.shape[0]
    problem = _Problem(
        n=n,
        h=_address(h),
        J=_address(J),
        colptr=_address(colptr),
        rowval=_address(rowval),
        nzval=_address(nzval),
        delta=_address(delta),
        scale=scale,
        sense=sense,
        dgemv=scipy_dgemv() if J is not None else None,
    )
    schedule = _Schedule(p=beta.shape[0], beta=_address(beta), gamma=_address(gamma))
    ws = {name: np.empty(n) for name in ("x", "y", "z", "m")}
    ws["sigma"] = np.empty(n, dtype=np.int8)
    ws["group"] = np.empty(n, dtype=np.int64)
    workspace = _Workspace(**{name: _address(a) for name, a in ws.items()})
    code = _lib.mfaoa_solve(ctypes.byref(problem), ctypes.byref(schedule), ctypes.byref(workspace))
    return code, ws


def linear_schedule(p, tau):
    """beta and gamma of the linear schedule, from mfaoa_linear_schedule."""
    beta = np.empty(p)
    gamma = np.empty(p)
    code = _lib.mfaoa_linear_schedule(p, tau, _address(beta), _address(gamma))
    assert code == OK
    return beta, gamma
