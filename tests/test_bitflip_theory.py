"""Closed-form bit-flip theory for the effective one-mode cat (Task 2.1)."""

import math

import numpy as np
import pytest

from catqubit import bitflip_theory as bt
from catqubit.params import CatParams

KAPPA1 = 1.0
KAPPA2 = 100.0


def test_chin_is_the_entire_part_of_the_hyperbolic_cosine_integral():
    # Chin(z) = sum_k z^(2k) / (2k (2k)!) = z^2/4 + z^4/96 + ...
    z = 0.3
    series = z**2 / 4 + z**4 / 96 + z**6 / 4320
    assert bt.chin(z) == pytest.approx(series, rel=1e-8)


def test_first_order_rate_matches_its_large_alpha_asymptote():
    alpha2 = 4.0
    asymptote = KAPPA1 * alpha2 * math.exp(-4 * alpha2)
    assert bt.first_order_jump_rate(KAPPA1, alpha2) == pytest.approx(asymptote, rel=1e-10)


def test_first_order_rate_is_exact_coth_form_at_small_alpha():
    alpha2 = 0.5
    exact = KAPPA1 * alpha2 * (1 / math.tanh(2 * alpha2) - 1) / 2
    assert bt.first_order_jump_rate(KAPPA1, alpha2) == pytest.approx(exact, rel=1e-12)


def test_second_order_rate_approaches_its_asymptote_from_above():
    ratios = [
        bt.second_order_jump_rate(KAPPA1, KAPPA2, a2)
        / (KAPPA1**2 / (2 * KAPPA2) * math.exp(-2 * a2))
        for a2 in (5.0, 20.0, 50.0)
    ]
    assert ratios[0] > ratios[1] > ratios[2] > 1.0
    assert ratios[2] == pytest.approx(1.0, abs=0.02)


def test_asymptotic_jump_rate_is_the_sum_of_both_exponentials():
    a2 = 3.0
    expected = KAPPA1 * a2 * math.exp(-4 * a2) + KAPPA1**2 / (2 * KAPPA2) * math.exp(-2 * a2)
    assert bt.asymptotic_jump_rate(KAPPA1, KAPPA2, a2) == pytest.approx(expected, rel=1e-12)


def test_observable_bit_flip_rate_is_twice_the_jump_rate():
    a2 = 2.0
    jump = bt.first_order_jump_rate(KAPPA1, a2) + bt.second_order_jump_rate(KAPPA1, KAPPA2, a2)
    assert bt.dubovitskii_bit_flip_rate(KAPPA1, KAPPA2, a2) == pytest.approx(2 * jump, rel=1e-12)
    asym = bt.asymptotic_jump_rate(KAPPA1, KAPPA2, a2)
    assert bt.asymptotic_bit_flip_rate(KAPPA1, KAPPA2, a2) == pytest.approx(2 * asym, rel=1e-12)


def test_rates_accept_arrays():
    a2 = np.array([1.0, 2.0, 3.0])
    rates = bt.dubovitskii_bit_flip_rate(KAPPA1, KAPPA2, a2)
    assert rates.shape == (3,)
    assert np.all(np.diff(rates) < 0)


def test_leading_log_crossover_is_half_log_of_the_rate_ratio():
    assert bt.crossover_alpha2_leading_log(KAPPA1, KAPPA2) == pytest.approx(math.log(100) / 2)


def test_asymptotic_crossover_equates_the_two_asymptotic_terms():
    a2 = bt.crossover_alpha2_asymptotic(KAPPA1, KAPPA2)
    first = KAPPA1 * a2 * math.exp(-4 * a2)
    second = KAPPA1**2 / (2 * KAPPA2) * math.exp(-2 * a2)
    assert first == pytest.approx(second, rel=1e-10)
    assert a2 > bt.crossover_alpha2_leading_log(KAPPA1, KAPPA2)


def test_exact_crossover_equates_the_two_exact_orders():
    a2 = bt.crossover_alpha2_exact(KAPPA1, KAPPA2)
    first = bt.first_order_jump_rate(KAPPA1, a2)
    second = bt.second_order_jump_rate(KAPPA1, KAPPA2, a2)
    assert first == pytest.approx(second, rel=1e-8)


def test_exact_crossover_for_the_challenge_parameters_lies_near_alpha_1p75(params):
    alpha_c = math.sqrt(bt.crossover_alpha2_exact(params.kappa_a, params.kappa2))
    assert 1.6 < alpha_c < 1.8


@pytest.mark.parametrize("func", [bt.crossover_alpha2_leading_log, bt.crossover_alpha2_asymptotic])
def test_crossover_needs_weak_single_photon_loss(func):
    with pytest.raises(ValueError, match="kappa1"):
        func(1.0, 1.0)


def test_thermal_estimate_is_kappa1_nth_times_e_minus_2_alpha2():
    assert bt.thermal_bit_flip_estimate(2.0, 0.1, 1.5) == pytest.approx(0.2 * math.exp(-3.0))
    assert bt.thermal_bit_flip_estimate(2.0, 0.0, 1.5) == 0.0
    with pytest.raises(ValueError, match="n_th"):
        bt.thermal_bit_flip_estimate(2.0, -0.1, 1.5)


@pytest.mark.parametrize(
    ("bad", "match"),
    [({"kappa1": -1.0}, "kappa1"), ({"alpha2": 0.0}, "alpha2"), ({"kappa2": 0.0}, "kappa2")],
)
def test_rates_reject_unphysical_inputs(bad, match):
    args = {"kappa1": 1.0, "kappa2": 10.0, "alpha2": 1.0} | bad
    with pytest.raises(ValueError, match=match):
        bt.dubovitskii_bit_flip_rate(**args)


@pytest.mark.parametrize(("alpha", "rtol"), [(0.8, 2e-3), (1.0, 2e-3), (1.4, 2e-3), (2.1, 2.5e-2)])
def test_dubovitskii_formula_matches_the_effective_model_gap(params, alpha, rtol):
    # Second-order perturbation theory in kappa1/kappa2 = 0.0104; the higher orders grow
    # with alpha^2 kappa1/kappa2, so the tolerance is looser at large alpha.
    p = params.with_alpha(alpha)
    gap = bt.effective_bit_flip_rate(p, na=bt.effective_memory_dim(alpha))
    theory = bt.dubovitskii_bit_flip_rate(p.kappa_a, p.kappa2, alpha**2)
    assert gap == pytest.approx(theory, rel=rtol)


def test_effective_gap_is_converged_in_the_memory_truncation(params):
    p = params.with_alpha(2.1)
    na = bt.effective_memory_dim(2.1)
    assert bt.effective_bit_flip_rate(p, na) == pytest.approx(
        bt.effective_bit_flip_rate(p, na + 10), rel=1e-5
    )


def test_thermal_photons_raise_the_effective_bit_flip_rate(params):
    p = params.with_alpha(1.6)
    na = bt.effective_memory_dim(1.6)
    cold = bt.effective_bit_flip_rate(p, na)
    hot = bt.effective_bit_flip_rate(p, na, n_th=0.1)
    assert hot > 1.5 * cold


def test_effective_gap_rejects_negative_thermal_population(params):
    with pytest.raises(ValueError, match="n_th"):
        bt.effective_bit_flip_rate(params.with_alpha(1.0), 12, n_th=-0.1)


def test_effective_gap_without_single_photon_loss_is_zero():
    p = CatParams(g2=1.0, kappa_b=4.0, kappa_a=0.0, alpha=1.2)
    # Exact zero mode; shift-invert round-off leaves ~1e-9 (kappa2 = 1 here).
    assert bt.effective_bit_flip_rate(p, 16) == pytest.approx(0.0, abs=1e-8)


def test_local_exponent_of_a_pure_exponential_is_its_rate():
    a2 = np.linspace(1.0, 4.0, 7)
    np.testing.assert_allclose(bt.local_exponent(a2, 3.0 * np.exp(-2.5 * a2)), 2.5, rtol=1e-10)


def test_local_exponent_includes_the_prefactor_slope():
    a2 = np.linspace(2.0, 6.0, 401)
    exps = bt.local_exponent(a2, a2 * np.exp(-2.0 * a2))
    np.testing.assert_allclose(exps[1:-1], 2.0 - 1.0 / a2[1:-1], rtol=1e-4)


def test_local_exponent_rejects_bad_samples():
    with pytest.raises(ValueError, match="increasing"):
        bt.local_exponent([2.0, 1.0, 3.0], [1.0, 0.5, 0.2])
    with pytest.raises(ValueError, match="positive"):
        bt.local_exponent([1.0, 2.0, 3.0], [1.0, 0.0, 0.2])
    with pytest.raises(ValueError, match="three"):
        bt.local_exponent([1.0, 2.0], [1.0, 0.5])


@pytest.mark.parametrize("power", [0, 1])
def test_fit_exponent_recovers_gamma_in_both_forms(power):
    a2 = np.linspace(0.6, 2.0, 8)
    rates = 0.7 * a2**power * np.exp(-2.8 * a2)
    fit = bt.fit_exponent(a2, rates, prefactor_power=power)
    assert fit.gamma == pytest.approx(2.8, rel=1e-10)
    assert fit.log_prefactor == pytest.approx(math.log(0.7), rel=1e-10)
    assert fit.rms_residual == pytest.approx(0.0, abs=1e-10)
    assert fit.prefactor_power == power


def test_fit_exponent_needs_two_points():
    with pytest.raises(ValueError, match="two"):
        bt.fit_exponent([1.0], [0.1])
