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


def gmatmul(A, B):
    """Matrix product A @ B; GPU/complex64 for large A, else numpy/complex128.

    Added alongside ginv/gsolve/geig after profiling showed `RedhefferStar`'s
    plain `@` chains (not routed through this module) are ~47% of N=3 wall
    time -- the ops here (inv/solve/eig) were only ~43%. Isolated benchmark:
    matmul is ~20.7x faster on GPU/complex64 at the ~4800x4800 sizes RCWA
    hits, similar to solve's ~9-15x. Kept as a standalone offload for spot
    use / benchmarking; `redheffer_star_gpu` below is the preferred path for
    RedhefferStar itself since it batches transfer across the whole call.
    """
    if _offload(A):
        cp = _cupy()
        r = cp.matmul(cp.asarray(A, dtype=_GPU_DTYPE), cp.asarray(B, dtype=_GPU_DTYPE))
        return cp.asnumpy(r).astype(np.complex128)
    return A @ B


def redheffer_offload(N) -> bool:
    """Whether RedhefferStar on NxN blocks should dispatch to the GPU path."""
    return DEVICE == "gpu" and N >= GPU_MIN


def redheffer_star_gpu(SA, SB):
    """GPU-resident Redheffer star product.

    Per-op offload (ginv/gsolve/gmatmul individually) pays a host<->device
    transfer for each of the ~10 large matmuls/solves inside one Redheffer
    star call, which is what capped the earlier offload-only port at ~1.7-2.6x
    end-to-end (Amdahl's law: only ~43% of wall time was in offloaded ops).
    This helper instead moves the 8 S-blocks to the GPU ONCE in complex64,
    does the whole computation on-device (matches `RedhefferStar` in utils.py
    exactly, just in cupy), and moves the 4 result blocks back ONCE.
    RedhefferStar is called only ~8x/spectrum, so batching the transfer this
    way matters far more than it would for a call made thousands of times.
    """
    cp = _cupy()
    dt = _GPU_DTYPE

    SA_11 = cp.asarray(SA['S11'], dtype=dt)
    SA_12 = cp.asarray(SA['S12'], dtype=dt)
    SA_21 = cp.asarray(SA['S21'], dtype=dt)
    SA_22 = cp.asarray(SA['S22'], dtype=dt)
    SB_11 = cp.asarray(SB['S11'], dtype=dt)
    SB_12 = cp.asarray(SB['S12'], dtype=dt)
    SB_21 = cp.asarray(SB['S21'], dtype=dt)
    SB_22 = cp.asarray(SB['S22'], dtype=dt)

    N = SA_11.shape[0]
    I = cp.eye(N, dtype=dt)
    D = I - SB_11 @ SA_22
    F = I - SA_22 @ SB_11

    SAB_11 = SA_11 + SA_12 @ cp.linalg.solve(D, SB_11) @ SA_21
    SAB_12 = SA_12 @ cp.linalg.solve(D, SB_12)
    SAB_21 = SB_21 @ cp.linalg.solve(F, SA_21)
    SAB_22 = SB_22 + SB_21 @ cp.linalg.solve(F, SA_22) @ SB_12

    # cp.block is avoided (cupy support for it is inconsistent across
    # versions); cp.concatenate is unambiguous and always available.
    SAB = cp.concatenate(
        [cp.concatenate([SAB_11, SAB_12], axis=1),
         cp.concatenate([SAB_21, SAB_22], axis=1)],
        axis=0,
    )

    def _to_host(X):
        return cp.asnumpy(X).astype(np.complex128)

    SAB_host = _to_host(SAB)
    SAB_dict = {
        'S11': _to_host(SAB_11), 'S22': _to_host(SAB_22),
        'S12': _to_host(SAB_12), 'S21': _to_host(SAB_21),
    }
    return SAB_host, SAB_dict
