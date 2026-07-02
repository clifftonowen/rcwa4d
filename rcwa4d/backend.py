"""Optional GPU (CuPy, single-precision) offload for the solver's heavy linear algebra.

WHY (fork adapter — keeps the solver otherwise numpy/CPU)
---------------------------------------------------------
Benchmarks on an RTX 3060 Ti: complex128 (double) on a consumer GPU is a wash
(FP64 is throttled), but complex64 (single) is 25-45x faster for the ops that
dominate RCWA -- general `eig`, `inv`, and `solve` on the (2N+1)^4-sized coupled
matrices. So the win requires SINGLE precision, and the risk is accuracy.

This module offloads ONLY the heavy calls to the GPU in complex64 and returns
complex128, so the rest of the solver is unchanged (arrays stay numpy/CPU). Small
matrices stay on the CPU (transfer overhead would dominate). `eig` is the biggest
cost, so this captures most of the speedup while isolating the precision question
to a few well-defined ops (checked via energy conservation + CD-vs-CPU agreement).

Control
-------
`RCWA4D_DEVICE` env var ('cpu' default, or 'gpu'), or call `set_device('gpu')`.
`RCWA4D_GPU_MIN` env var: min matrix dimension to offload (default 256).
Default is CPU -> byte-for-byte the original behaviour.
"""

import os

import numpy as np

DEVICE = os.environ.get("RCWA4D_DEVICE", "cpu")          # 'cpu' | 'gpu'
GPU_MIN = int(os.environ.get("RCWA4D_GPU_MIN", "256"))   # offload if min(shape) >= this
_GPU_DTYPE = None  # set to cupy.complex64 lazily

_cp = None


def _cupy():
    global _cp, _GPU_DTYPE
    if _cp is None:
        import cupy as cp
        _cp = cp
        _GPU_DTYPE = cp.complex64
    return _cp


def set_device(dev: str) -> None:
    """'cpu' or 'gpu'. Checked per call, so it can be toggled at runtime."""
    global DEVICE
    assert dev in ("cpu", "gpu"), dev
    DEVICE = dev


def _offload(A) -> bool:
    return DEVICE == "gpu" and min(A.shape) >= GPU_MIN


def ginv(A):
    """Complex matrix inverse; GPU/complex64 for large A, else numpy/complex128."""
    if _offload(A):
        cp = _cupy()
        r = cp.linalg.inv(cp.asarray(A, dtype=_GPU_DTYPE))
        return cp.asnumpy(r).astype(np.complex128)
    return np.linalg.inv(A)


def gsolve(A, B):
    """Solve A x = B; GPU/complex64 for large A, else numpy/complex128."""
    if _offload(A):
        cp = _cupy()
        r = cp.linalg.solve(cp.asarray(A, dtype=_GPU_DTYPE), cp.asarray(B, dtype=_GPU_DTYPE))
        return cp.asnumpy(r).astype(np.complex128)
    return np.linalg.solve(A, B)


def geig(A):
    """General eigendecomposition; GPU/complex64 for large A, else numpy/complex128."""
    if _offload(A):
        cp = _cupy()
        w, V = cp.linalg.eig(cp.asarray(A, dtype=_GPU_DTYPE))
        return cp.asnumpy(w).astype(np.complex128), cp.asnumpy(V).astype(np.complex128)
    return np.linalg.eig(A)
