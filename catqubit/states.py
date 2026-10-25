"""Logical states of the cat qubit."""

import dynamiqs as dq
from dynamiqs import QArray

MIN_DIM = 2


def cat_state(dim: int, alpha: complex, parity: int) -> QArray:
    """Cat state (|alpha> + parity * |-alpha>) / N, with parity +1 (even) or -1 (odd).

    Built from truncated coherent states and renormalized, so the parity is exact in
    the truncated space even when the analytic normalization would not be.
    """
    if dim < MIN_DIM:
        raise ValueError(f"dim must be at least {MIN_DIM}, got {dim}")
    if parity not in (1, -1) or isinstance(parity, bool):
        raise ValueError(f"parity must be +1 or -1, got {parity!r}")
    return (dq.coherent(dim, alpha) + parity * dq.coherent(dim, -alpha)).unit()
