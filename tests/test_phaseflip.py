"""Phase-flip rate from parity moments, analytic references and fits (Task 2.2)."""

import math
from dataclasses import replace

import dynamiqs as dq
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp

from catqubit.liouvillian import FockTruncation, lindblad_superoperator, vec
from catqubit.phaseflip import (
    cat_manifold_offset,
    cat_manifold_parity,
    cat_manifold_rate,
    even_cat_rate,
    evolve_parity,
    fit_line,
    fit_parity_decay,
    instantaneous_phase_flip_rate,
    large_alpha_rate,
    parity_derivative,
    parity_operators,
    steady_state_parity,
    thermal_rate,
)
from catqubit.rates import phase_flip_rate
from catqubit.simulate import inflate
from catqubit.states import cat_state

KAPPA = 0.3  # an arbitrary rate for the closed-form checks


# --- parity operators -------------------------------------------------------------------


def test_parity_operators_obey_their_algebra():
    ops = parity_operators(6, 3)
    p = np.asarray(ops.parity.to_jax())
    n_p = np.asarray(ops.n_parity.to_jax())
    m_p = np.asarray(ops.m_parity.to_jax())
    assert np.allclose(p @ p, np.eye(18))
    # a a^dag = a^dag a + 1 except on the top memory level, which the truncation cuts.
    top = np.kron(np.diag([0, 0, 0, 0, 0, 1.0]), np.eye(3))
    assert np.allclose(m_p, n_p + p - top @ (n_p + p))


def test_single_mode_parity_operators():
    ops = parity_operators(5)
    assert np.allclose(np.diag(np.asarray(ops.parity.to_jax())), [1, -1, 1, -1, 1])
    assert np.allclose(np.diag(np.asarray(ops.n_parity.to_jax())), [0, -1, 2, -3, 4])


# --- the exact identity d<P>/dt = -2 kappa_a [(1 + n) <a^dag a P> + n <a a^dag P>] -------


def _finite_difference_check(t, parity, derivative):
    fd = np.gradient(parity, t, edge_order=2)
    inner = slice(2, -2)
    scale = np.max(np.abs(derivative))
    return np.max(np.abs(fd[inner] - derivative[inner])) / scale


def test_identity_matches_finite_difference_of_inflate(params):
    p = params.with_alpha(1.0)
    na, nb = 12, 4
    ops = parity_operators(na, nb)
    tsave = jnp.linspace(0.0, 2.0, 801)
    res = inflate(
        na,
        nb,
        p,
        tsave,
        memory0=cat_state(na, p.alpha, +1),
        extra_exp_ops=[ops.parity, ops.n_parity],
        save_states=False,
    )
    parity, n_parity = (np.real(np.asarray(res.expects[k])) for k in (2, 3))
    derivative = parity_derivative(p.kappa_a, n_parity, np.zeros_like(n_parity))
    assert _finite_difference_check(np.asarray(tsave), parity, derivative) < 1e-4


def test_thermal_identity_needs_the_gain_term(params):
    # Raise kappa_a so the memory-loss term dominates the finite-difference error budget.
    p = replace(params.with_alpha(1.0), kappa_a=1.0)
    n_th = 0.1
    trace = evolve_parity(p, 12, 4, jnp.linspace(0.0, 1.0, 801), n_th_a=n_th, n_th_b=0.011)
    t = np.asarray(trace.t)
    exact = parity_derivative(p.kappa_a, trace.n_parity, trace.m_parity, n_th)
    loss_only = parity_derivative(p.kappa_a * (1 + n_th), trace.n_parity, trace.m_parity)
    assert _finite_difference_check(t, trace.parity, exact) < 1e-4
    assert _finite_difference_check(t, trace.parity, loss_only) > 1e-2


def test_evolve_parity_matches_inflate_without_thermal_photons(params):
    p = params.with_alpha(1.0)
    na, nb = 10, 4
    tsave = jnp.linspace(0.0, 1.0, 11)
    trace = evolve_parity(p, na, nb, tsave)
    res = inflate(
        na,
        nb,
        p,
        tsave,
        memory0=cat_state(na, p.alpha, +1),
        extra_exp_ops=[parity_operators(na, nb).parity],
        save_states=False,
    )
    assert np.allclose(trace.parity, np.real(np.asarray(res.expects[2])), atol=1e-8)


def test_instantaneous_rate_is_minus_log_derivative():
    parity = np.array([1.0, 0.5])
    n_parity = np.array([2.0, 1.0])
    m_parity = np.array([3.0, 1.5])
    rate = instantaneous_phase_flip_rate(KAPPA, parity, n_parity, m_parity, n_th=0.1)
    expected = 2 * KAPPA * (1.1 * n_parity + 0.1 * m_parity) / parity
    assert np.allclose(rate, expected)


def test_identity_rejects_negative_thermal_population():
    with pytest.raises(ValueError, match="n_th"):
        parity_derivative(KAPPA, np.ones(2), np.ones(2), n_th=-0.1)


# --- analytic references ------------------------------------------------------------------


def test_finite_alpha_forms_reach_two_kappa_alpha_squared_at_large_alpha():
    alpha = 3.0
    reference = large_alpha_rate(KAPPA, alpha)
    assert reference == pytest.approx(2 * KAPPA * 9.0)
    assert even_cat_rate(KAPPA, alpha) == pytest.approx(reference, rel=1e-7)
    assert cat_manifold_rate(KAPPA, alpha) == pytest.approx(reference, rel=1e-7)
    assert cat_manifold_offset(alpha) < 1e-7


def test_finite_alpha_forms_at_small_alpha():
    alpha = 1e-3
    # coth(2 alpha^2) ~ 1 / (2 alpha^2): the cat manifold rate tends to kappa_a.
    assert cat_manifold_rate(KAPPA, alpha) == pytest.approx(KAPPA, rel=1e-6)
    # The even cat tends to |0> + O(alpha^2) |2>: n = alpha^2 tanh(alpha^2) ~ alpha^4.
    assert even_cat_rate(KAPPA, alpha) == pytest.approx(2 * KAPPA * alpha**4, rel=1e-6)
    assert cat_manifold_offset(alpha) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("alpha", [0.7, 1.0, 1.5, 2.1])
def test_initial_slope_of_manifold_parity_is_the_even_cat_rate(alpha):
    # -dP/dt at t = 0 is (r_+ + r_-)(1 - P_inf) = 2 r_+ = 2 kappa alpha^2 tanh(alpha^2).
    initial = cat_manifold_rate(KAPPA, alpha) * (1 - cat_manifold_offset(alpha))
    assert initial == pytest.approx(even_cat_rate(KAPPA, alpha), rel=1e-12)
    assert cat_manifold_parity(0.0, KAPPA, alpha) == pytest.approx(1.0)
    assert cat_manifold_parity(1e6, KAPPA, alpha) == pytest.approx(cat_manifold_offset(alpha))


def test_even_cat_rate_is_twice_kappa_times_photon_number_of_the_even_cat():
    alpha, dim = 1.3, 30
    n_even = float(np.real(dq.expect(dq.number(dim), cat_state(dim, alpha, +1))))
    assert even_cat_rate(KAPPA, alpha) == pytest.approx(2 * KAPPA * n_even, rel=1e-10)


def test_thermal_rate_counts_loss_and_gain_jumps():
    alpha, n = 2.1, 0.1
    assert thermal_rate(KAPPA, alpha, 0.0) == pytest.approx(large_alpha_rate(KAPPA, alpha))
    assert thermal_rate(KAPPA, alpha, n) == pytest.approx(2 * KAPPA * (alpha**2 * 1.2 + 0.1))
    with pytest.raises(ValueError, match="n_th"):
        thermal_rate(KAPPA, alpha, -0.1)


def _effective_parity_rate(alpha: float, kappa_a: float, kappa2: float, dim: int = 24) -> float:
    """Slowest nonzero even-sector rate of kappa2 D[a^2 - alpha^2] + kappa_a D[a] (dense)."""
    a = sp.diags(np.sqrt(np.arange(1, dim)), 1, format="csr")
    two_photon = math.sqrt(kappa2) * (a @ a - alpha**2 * sp.identity(dim))
    zero_h = sp.csr_matrix((dim, dim), dtype=complex)
    big_l = lindblad_superoperator(zero_h, [two_photon, math.sqrt(kappa_a) * a]).toarray()
    n = np.arange(dim)
    even = vec((n[:, None] + n[None, :]) % 2) == 0
    vals = np.linalg.eigvals(big_l[np.ix_(even, even)])
    return float(-np.sort(vals.real)[-2])


@pytest.mark.parametrize("alpha", [1.0, 1.5])
def test_cat_manifold_rate_is_the_adiabatic_limit_of_the_effective_model(alpha):
    # a^dag a rho leaves the cat manifold, so the coth form holds only for kappa_a << kappa2;
    # the correction is linear in kappa_a / kappa2.
    kappa2 = 1.0
    errors = []
    for ratio in (1e-2, 1e-4):
        rate = _effective_parity_rate(alpha, ratio * kappa2, kappa2)
        errors.append(rate / cat_manifold_rate(ratio * kappa2, alpha) - 1)
    assert abs(errors[1]) < 1e-4
    assert errors[0] / errors[1] == pytest.approx(100, rel=0.02)


def test_large_alpha_form_misses_the_finite_alpha_correction_at_alpha_one():
    rate = _effective_parity_rate(1.0, 1e-4, 1.0)
    assert rate / large_alpha_rate(1e-4, 1.0) - 1 > 0.03  # coth(2) = 1.037


# --- fits ---------------------------------------------------------------------------------


def test_fit_parity_decay_recovers_rate_amplitude_and_offset():
    t = np.linspace(1.0, 10.0, 60)
    y = 0.05 + 0.9 * np.exp(-0.4 * t)
    fit = fit_parity_decay(t, y)
    assert fit.rate == pytest.approx(0.4, rel=1e-8)
    assert fit.amplitude == pytest.approx(0.9, rel=1e-8)
    assert fit.offset == pytest.approx(0.05, abs=1e-9)
    assert fit.rate_stderr < 1e-8


def test_fit_parity_decay_rejects_too_few_points():
    with pytest.raises(ValueError, match="points"):
        fit_parity_decay(np.array([0.0, 1.0]), np.array([1.0, 0.5]))


def test_fit_line_gives_slope_intercept_and_error():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    fit = fit_line(x, 2.0 * x + 0.5)
    assert fit.slope == pytest.approx(2.0)
    assert fit.intercept == pytest.approx(0.5)
    assert fit.slope_stderr == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ValueError, match="points"):
        fit_line(x[:2], x[:2])


@pytest.mark.slow
def test_offset_fit_of_two_mode_parity_matches_the_spectral_rate(params):
    p = params.with_alpha(1.2)
    na, nb = 18, 5
    gamma = phase_flip_rate(p, FockTruncation(na=na, nb=nb))
    trace = evolve_parity(p, na, nb, jnp.linspace(0.0, 3.0 / gamma, 121))
    late = np.asarray(trace.t) > 1.0
    fit = fit_parity_decay(np.asarray(trace.t)[late], trace.parity[late])
    assert fit.rate == pytest.approx(gamma, rel=1e-4)
    assert math.isclose(fit.offset, cat_manifold_offset(p.alpha), rel_tol=0.05)
    # Once the fast modes are gone the identity gives Gamma_Z (1 - P_inf / P(t)).
    rate = instantaneous_phase_flip_rate(p.kappa_a, trace.parity, trace.n_parity, trace.m_parity)
    plateau = gamma * (1 - fit.offset / trace.parity)
    assert np.allclose(rate[late], plateau[late], rtol=1e-3)


def test_steady_state_parity_tends_to_the_cat_manifold_offset(params):
    p = params.with_alpha(1.0)
    p_inf = steady_state_parity(p, FockTruncation(na=14, nb=5))
    # Adiabatic corrections are O(kappa_a / kappa_2) ~ 1e-2 here.
    assert p_inf == pytest.approx(cat_manifold_offset(p.alpha), rel=0.03)
    assert steady_state_parity(p, FockTruncation(na=14, nb=5), n_th_a=0.1) < p_inf


@pytest.mark.slow
def test_steady_state_parity_matches_the_long_time_limit(params):
    p = params.with_alpha(1.0)
    na, nb = 14, 5
    gamma = phase_flip_rate(p, FockTruncation(na=na, nb=nb))
    trace = evolve_parity(p, na, nb, jnp.linspace(0.0, 12.0 / gamma, 7))
    assert trace.parity[-1] == pytest.approx(
        steady_state_parity(p, FockTruncation(na=na, nb=nb)), abs=1e-4
    )
