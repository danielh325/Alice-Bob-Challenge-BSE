"""Convergence-study and decay-fit helpers (Tasks 1.2 and 1.3)."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

MIN_POINTS = 2


def smallest_converged(sizes: Sequence[int], values: Sequence[float], rtol: float) -> int:
    """Smallest truncation whose value, and that of every larger one, is within rtol
    of the value at the largest truncation (taken as the reference)."""
    if len(sizes) != len(values):
        raise ValueError("sizes and values must have the same length")
    if len(sizes) < MIN_POINTS:
        raise ValueError(f"need at least two sizes, got {len(sizes)}")
    if any(b <= a for a, b in zip(sizes, sizes[1:], strict=False)):
        raise ValueError(f"sizes must be strictly increasing, got {list(sizes)}")
    if rtol <= 0:
        raise ValueError(f"rtol must be positive, got {rtol}")
    reference = values[-1]
    if reference == 0:
        raise ValueError("reference value (largest size) must be nonzero for a relative tolerance")

    converged = sizes[-1]
    for size, value in zip(reversed(sizes), reversed(values), strict=True):
        if abs(value - reference) > rtol * abs(reference):
            break
        converged = size
    return converged


def fit_decay_rate(t: ArrayLike, y: ArrayLike) -> float:
    """Rate gamma of a least-squares fit y = A exp(-gamma t), done linearly on log(y)."""
    t, y = np.asarray(t, dtype=float), np.asarray(y, dtype=float)
    if t.size < MIN_POINTS or t.shape != y.shape:
        raise ValueError("need at least two (t, y) points of matching shape")
    if np.any(y <= 0):
        raise ValueError("y must be strictly positive to fit an exponential on log(y)")
    slope, _ = np.polyfit(t, np.log(y), 1)
    return float(-slope)
