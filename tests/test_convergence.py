import math

import numpy as np
import pytest

from catqubit.convergence import fit_decay_rate, smallest_converged


def test_smallest_converged_finds_first_size_matching_all_larger_sizes():
    assert smallest_converged([16, 20, 24, 32], [4.0, 4.3, 4.40, 4.401], rtol=0.01) == 24


def test_smallest_converged_returns_first_size_when_all_agree():
    assert smallest_converged([4, 6, 8], [1.0, 1.0005, 1.0], rtol=0.01) == 4


def test_smallest_converged_requires_every_larger_size_to_agree():
    # 20 matches the reference but 24 does not, so 20 is not converged.
    assert smallest_converged([16, 20, 24, 32], [3.0, 4.4, 5.0, 4.4], rtol=0.01) == 32


@pytest.mark.parametrize(
    ("sizes", "values", "rtol", "match"),
    [
        ([16, 20], [1.0], 0.01, "same length"),
        ([16], [1.0], 0.01, "at least two"),
        ([20, 16], [1.0, 1.0], 0.01, "increasing"),
        ([16, 20], [1.0, 1.0], 0.0, "rtol"),
        ([16, 20], [1.0, 0.0], 0.01, "reference"),
    ],
)
def test_smallest_converged_rejects_bad_input(sizes, values, rtol, match):
    with pytest.raises(ValueError, match=match):
        smallest_converged(sizes, values, rtol=rtol)


def test_fit_decay_rate_recovers_exponential_rate():
    t = np.linspace(0.0, 5.0, 50)
    assert fit_decay_rate(t, 0.7 * np.exp(-0.515 * t)) == pytest.approx(0.515, rel=1e-10)


def test_fit_decay_rate_is_insensitive_to_amplitude():
    t = np.linspace(1.0, 3.0, 20)
    y = np.exp(-2.0 * t)
    assert fit_decay_rate(t, 5 * y) == pytest.approx(fit_decay_rate(t, y))


def test_fit_decay_rate_rejects_non_positive_values():
    with pytest.raises(ValueError, match="positive"):
        fit_decay_rate([0.0, 1.0, 2.0], [1.0, 0.0, -0.1])


def test_fit_decay_rate_rejects_too_few_points():
    with pytest.raises(ValueError, match="two"):
        fit_decay_rate([0.0], [1.0])


def test_fit_decay_rate_of_growth_is_negative():
    t = np.linspace(0.0, 1.0, 5)
    assert fit_decay_rate(t, np.exp(math.log(3) * t)) == pytest.approx(-math.log(3))
