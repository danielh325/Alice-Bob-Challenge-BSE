"""Buffer readout dynamics <b>(t) (Task 2.3): linear oscillator, spectral modes, time domain."""

import math

import numpy as np
import pytest
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from catqubit.liouvillian import FockTruncation, two_mode_liouvillian, two_mode_operators, vec
from catqubit.params import CatParams
from catqubit.readout_dynamics import (
    LinearBufferResponse,
    LiouvillianModes,
    SpectralBufferResponse,
    buffer_spectrum,
    codeword_density,
    hermitian_conjugate_modes,
    liouvillian_modes,
    modal_weights,
    settling_time_from_curve,
    simulate_buffer_response,
)

# ---------------------------------------------------------------- (a) linear oscillator


def _oscillator(zeta: float, nu: float = 20.0, eps_z: float = 0.3, sign: int = 1):
    return LinearBufferResponse(nu=nu, zeta=zeta, eps_z=eps_z, sign=sign)


def test_linear_response_from_reglade_parameters(params):
    lin = LinearBufferResponse.from_params(params, eps_z=0.05 * params.g2, sign=1)
    assert lin.nu == pytest.approx(2 * params.alpha * params.g2)
    assert lin.zeta == pytest.approx(params.kappa_b / (8 * params.g2 * params.alpha))
    assert lin.zeta == pytest.approx(0.20, abs=0.01)
    assert lin.damping_rate == pytest.approx(params.kappa_b / 4)
    assert lin.ringing_frequency_mhz == pytest.approx(3.1, abs=0.05)
    assert lin.envelope_settling_time() == pytest.approx(1.1, abs=0.05)


@pytest.mark.parametrize("sign", [1, -1])
def test_linear_steady_state_is_minus_sign_eps_z_over_two_alpha_g2(params, sign):
    eps_z = 0.1 * params.g2
    lin = LinearBufferResponse.from_params(params, eps_z=eps_z, sign=sign)
    assert lin.steady_state == pytest.approx(-sign * eps_z / (2 * params.alpha * params.g2))
    assert lin.mean_b(np.array([50.0]))[0] == pytest.approx(lin.steady_state, rel=1e-12)


def test_linear_response_rejects_invalid_arguments():
    with pytest.raises(ValueError, match="sign"):
        _oscillator(0.2, sign=0)
    with pytest.raises(ValueError, match="nu"):
        _oscillator(0.2, nu=-1.0)
    with pytest.raises(ValueError, match="zeta"):
        _oscillator(0.0)


@pytest.mark.parametrize("zeta", [0.2, 1.0, 2.5])
def test_linear_mean_b_solves_the_damped_oscillator_ode(zeta):
    """b'' + 2 zeta nu b' + nu^2 b = -sign nu eps_z, with b(0) = b'(0) = 0."""
    lin = _oscillator(zeta, sign=-1)
    h = 1e-4
    t = np.linspace(0.05, 1.0, 40)
    b, bp, bm = lin.mean_b(t), lin.mean_b(t + h), lin.mean_b(t - h)
    d1, d2 = (bp - bm) / (2 * h), (bp - 2 * b + bm) / h**2
    residual = d2 + 2 * zeta * lin.nu * d1 + lin.nu**2 * b + lin.sign * lin.nu * lin.eps_z
    assert np.max(np.abs(residual)) < 1e-5 * lin.nu**2 * abs(lin.steady_state)
    b0, b1 = lin.mean_b(np.array([0.0, h]))
    assert b0 == pytest.approx(0.0, abs=1e-15)
    assert abs(b1 - b0) / h < 1e-3 * lin.nu * abs(lin.steady_state)


@pytest.mark.parametrize("zeta", [0.2, 1.0, 2.5])
def test_linear_eigenvalues_are_roots_of_the_characteristic_polynomial(zeta):
    lin = _oscillator(zeta)
    lam = lin.eigenvalues
    assert lam.shape == (2,)
    assert np.allclose(lam**2 + 2 * zeta * lin.nu * lam + lin.nu**2, 0.0, atol=1e-9 * lin.nu**2)
    assert lam[0].real >= lam[1].real


def test_ringing_frequency_is_zero_without_oscillation():
    assert _oscillator(0.2).ringing_frequency == pytest.approx(20.0 * math.sqrt(1 - 0.04))
    assert _oscillator(0.2).is_underdamped
    assert _oscillator(1.0).ringing_frequency == 0.0
    assert _oscillator(2.5).ringing_frequency == 0.0
    assert not _oscillator(2.5).is_underdamped


@pytest.mark.parametrize("zeta", [0.2, 1.0, 2.5])
def test_settling_time_is_the_last_exit_from_the_tolerance_band(zeta):
    lin = _oscillator(zeta)
    tol = 0.01
    t_s = lin.settling_time(tol)
    dev = lambda t: np.abs(lin.mean_b(t) - lin.steady_state) / abs(lin.steady_state)  # noqa: E731
    assert dev(np.array([t_s]))[0] == pytest.approx(tol, rel=1e-6)
    later = np.linspace(t_s + 1e-6, t_s + 20 / lin.nu, 4000)
    assert np.all(dev(later) <= tol)
    t = np.linspace(0.0, 2 * t_s, 200001)
    from_curve = settling_time_from_curve(t, lin.mean_b(t), lin.steady_state, tol)
    assert from_curve == pytest.approx(t_s, abs=2 * (t[1] - t[0]))


def test_exact_settling_time_does_not_exceed_the_envelope_estimate(params):
    lin = LinearBufferResponse.from_params(params, eps_z=0.05 * params.g2, sign=1)
    assert lin.settling_time() <= lin.envelope_settling_time()
    assert lin.settling_time() > 0.8 * lin.envelope_settling_time()


def test_envelope_estimate_needs_an_underdamped_oscillator():
    with pytest.raises(ValueError, match="underdamped"):
        _oscillator(2.5).envelope_settling_time()


def test_settling_time_needs_a_nonzero_signal():
    with pytest.raises(ValueError, match="nonzero"):
        _oscillator(0.2, eps_z=0.0).settling_time()


def test_settling_time_is_zero_for_a_band_wider_than_the_signal():
    assert _oscillator(0.2).settling_time(tol=2.0) == 0.0


def test_settling_time_from_curve_edge_cases():
    t = np.linspace(0.0, 1.0, 11)
    assert settling_time_from_curve(t, np.ones(11), 1.0) == 0.0
    y = np.where(t < 0.45, 2.0, 1.0)
    assert settling_time_from_curve(t, y, 1.0) == pytest.approx(0.4 + 0.1 * 0.99)
    with pytest.raises(ValueError, match="not settled"):
        settling_time_from_curve(t, 1.0 + (t > 0.95), 1.0)
    with pytest.raises(ValueError, match="nonzero"):
        settling_time_from_curve(t, y, 0.0)


# ---------------------------------------------------------------- (b) Liouvillian modes


def _known_spectrum_matrix(values: np.ndarray, seed: int = 3) -> sp.csc_matrix:
    """Non-normal P diag(values) P^-1 with a well-conditioned random P."""
    rng = np.random.default_rng(seed)
    n = len(values)
    p = np.eye(n) + 0.3 * (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) / math.sqrt(n)
    return sp.csc_matrix(p @ np.diag(values) @ np.linalg.inv(p))


def _check_modes(modes: LiouvillianModes, big_l: sp.spmatrix, tol: float) -> None:
    k = len(modes.values)
    assert modes.right.shape == modes.left.shape == (big_l.shape[0], k)
    assert np.allclose(modes.left.conj().T @ modes.right, np.eye(k), atol=tol)
    assert modes.residual < tol
    assert np.all(np.diff(modes.values.real) <= 1e-12)
    lr = big_l @ modes.right - modes.right * modes.values
    assert np.max(np.linalg.norm(lr, axis=0) / np.linalg.norm(modes.right, axis=0)) < tol * 10


def test_liouvillian_modes_on_a_non_hermitian_matrix_with_known_spectrum():
    rng = np.random.default_rng(7)
    values = -rng.uniform(0.5, 5.0, 60) + 1j * rng.uniform(-10, 10, 60)
    big_l = _known_spectrum_matrix(values)
    sigma = -1.0 + 2.0j
    modes = liouvillian_modes(big_l, sigma=sigma, k=6)
    expected = values[np.argsort(np.abs(values - sigma))[:6]]
    assert np.allclose(np.sort_complex(modes.values), np.sort_complex(expected), atol=1e-10)
    _check_modes(modes, big_l, tol=1e-9)


def test_liouvillian_modes_resolves_a_near_degenerate_slow_pair():
    """Steady state 0 and a bit-flip-like mode -1e-7, as in the full model at alpha = 2.1."""
    rng = np.random.default_rng(11)
    others = -rng.uniform(0.5, 5.0, 40) + 1j * rng.uniform(-5, 5, 40)
    values = np.concatenate([[0.0, -1e-7], others])
    big_l = _known_spectrum_matrix(values)
    modes = liouvillian_modes(big_l, sigma=-1e-5, k=4)
    assert modes.values[0] == pytest.approx(0.0, abs=1e-13)
    assert modes.values[1] == pytest.approx(-1e-7, abs=1e-13)
    _check_modes(modes, big_l, tol=1e-8)


def test_liouvillian_modes_of_a_small_two_mode_liouvillian_match_dense_eig(params):
    p = params.with_alpha(1.0)
    trunc = FockTruncation(na=6, nb=3)
    big_l = two_mode_liouvillian(p, trunc, eps_z=0.2 * p.g2)
    assert big_l.shape[0] < 2000
    sigma = -2.0 + 6.0j
    modes = liouvillian_modes(big_l, sigma=sigma, k=8)
    dense = np.linalg.eigvals(big_l.toarray())
    expected = dense[np.argsort(np.abs(dense - sigma))[:8]]
    assert np.allclose(np.sort_complex(modes.values), np.sort_complex(expected), atol=1e-9)
    _check_modes(modes, big_l, tol=1e-9)


def test_liouvillian_modes_refuses_an_ambiguous_pairing():
    """An exactly degenerate pair has no unique left/right pairing."""
    rng = np.random.default_rng(5)
    others = -rng.uniform(1.0, 5.0, 30) + 1j * rng.uniform(-5, 5, 30)
    big_l = _known_spectrum_matrix(np.concatenate([[-0.1, -0.1], others]))
    with pytest.raises(ValueError, match="ambiguous"):
        liouvillian_modes(big_l, sigma=0.0, k=3)


def test_liouvillian_modes_refuses_an_unpartnered_eigenvalue(monkeypatch):
    import catqubit.readout_dynamics as rd

    big_l = _known_spectrum_matrix(-np.arange(1.0, 31.0) + 0.5j)
    monkeypatch.setattr(rd, "PAIR_RTOL", 0.0)
    with pytest.raises(ValueError, match="no left partner"):
        liouvillian_modes(big_l, sigma=-0.5, k=3)


def test_left_run_requests_only_k_modes_when_the_pairing_succeeds(monkeypatch):
    """Extra left modes are costly (poorly separated in shift-invert): only on a retry."""
    import catqubit.readout_dynamics as rd

    requested = []
    eigs = rd.sla.eigs

    def spy(*args, **kwargs):
        requested.append(kwargs["k"])
        return eigs(*args, **kwargs)

    monkeypatch.setattr(rd.sla, "eigs", spy)
    big_l = _known_spectrum_matrix(-np.arange(1.0, 31.0) + 0.5j)
    liouvillian_modes(big_l, sigma=-0.5, k=3)
    assert requested == [3, 3]


@pytest.mark.parametrize("k", [2, 3])
def test_liouvillian_modes_pairs_a_conjugate_pair_split_at_the_kth_mode(k):
    """Real sigma: -1 +- 1j are equidistant, so the k-th right and left modes may differ."""
    rng = np.random.default_rng(13)
    others = -rng.uniform(3.0, 6.0, 30) + 1j * rng.uniform(-5, 5, 30)
    values = np.concatenate([[-0.5, -0.8, -1.0 + 1.0j, -1.0 - 1.0j], others])
    big_l = _known_spectrum_matrix(values)
    modes = liouvillian_modes(big_l, sigma=0.0, k=k + 1)
    _check_modes(modes, big_l, tol=1e-9)


def test_pairing_refuses_two_right_modes_on_one_left_mode():
    from catqubit.readout_dynamics import _pair_left_to_right

    with pytest.raises(ValueError, match="matches two right modes"):
        _pair_left_to_right(np.array([-1.0, -1.0 - 1e-12]), np.array([-1.0, -9.0]), 0.5)


def test_liouvillian_modes_raises_without_retry_when_no_extra_modes_fit():
    """For n = 8 and k = 6, k + EXTRA_LEFT_MODES exceeds n - 2: the first failure is final."""
    big_l = _known_spectrum_matrix(np.array([-0.1, -0.1, -1, -2, -3, -4, -5, -6.0]))
    with pytest.raises(ValueError, match="ambiguous"):
        liouvillian_modes(big_l, sigma=0.0, k=6)


def _dense_modes(big_l: np.ndarray) -> LiouvillianModes:
    values, left, right = sla.eig(big_l, left=True, right=True)
    left = left / np.conj(np.sum(left.conj() * right, axis=0))
    order = np.argsort(-values.real)
    return LiouvillianModes(values[order], right[:, order], left[:, order], residual=0.0)


def test_modal_weights_reproduce_the_time_evolution_with_all_modes(params):
    p = params.with_alpha(1.0)
    trunc = FockTruncation(na=5, nb=2)
    big_l = two_mode_liouvillian(p, trunc, eps_z=0.3 * p.g2).toarray()
    _, b = two_mode_operators(trunc)
    row = vec(b.toarray().T)
    rng = np.random.default_rng(2)
    psi = rng.normal(size=trunc.dim) + 1j * rng.normal(size=trunc.dim)
    rho0 = vec(np.outer(psi, psi.conj()) / np.vdot(psi, psi))
    weights = modal_weights(_dense_modes(big_l), row, rho0)
    modes = _dense_modes(big_l)
    for t in (0.0, 0.1, 0.7):
        exact = row @ sla.expm(big_l * t) @ rho0
        assert np.sum(weights * np.exp(modes.values * t)) == pytest.approx(exact, abs=1e-10)


# ---------------------------------------------------------------- (c), (d) full model

# Cheap point for the fast suite. alpha = 1.6 is still well underdamped (zeta = 0.27) and
# FockTruncation(18, 4) gives the same ringing eigenvalue as (20, 4) to 2e-5 rad/us.
# (alpha = 1.4 is not used: its ringing is spread over three bands, 11.5-14.4 rad/us.)
CHEAP_ALPHA = 1.6
CHEAP_TRUNC = FockTruncation(na=18, nb=4)
SMALL_EPS_Z = 0.05  # in units of g2
T_WINDOW = np.linspace(0.0, 3.0, 601)


def _expm_trace(p: CatParams, trunc: FockTruncation, eps_z: float, sign: int) -> np.ndarray:
    """Time-domain <b>(t) on T_WINDOW by sparse exp(L t) rho0 (independent of the modes)."""
    big_l = two_mode_liouvillian(p, trunc, eps_z=eps_z).tocsc()
    _, b = two_mode_operators(trunc)
    rho0 = codeword_density(p, trunc, sign).astype(complex)
    states = spla.expm_multiply(big_l, rho0, start=0.0, stop=T_WINDOW[-1], num=len(T_WINDOW))
    return states @ vec(b.toarray().T)


@pytest.fixture(scope="module")
def cheap():
    """Spectrum and time traces at alpha = 1.6, eps_z = 0.05 g2 (computed once, ~15 s)."""
    p = CatParams.reglade().with_alpha(CHEAP_ALPHA)
    eps_z = SMALL_EPS_Z * p.g2
    spectrum = buffer_spectrum(p, CHEAP_TRUNC, eps_z)
    traces = {s: (T_WINDOW, _expm_trace(p, CHEAP_TRUNC, eps_z, s)) for s in (1, -1)}
    return p, eps_z, spectrum, traces


@pytest.mark.slow
def test_simulate_buffer_response_matches_the_sparse_propagator(params):
    """The inflate wrapper (dynamiqs, rectangular na x nb) agrees with exp(L t) rho0."""
    p = params.with_alpha(1.0)
    trunc = FockTruncation(na=10, nb=3)
    eps_z = 0.1 * p.g2
    tsave = np.linspace(0.0, 1.0, 11)
    big_l = two_mode_liouvillian(p, trunc, eps_z=eps_z).tocsc()
    _, b = two_mode_operators(trunc)
    rho0 = codeword_density(p, trunc, -1).astype(complex)
    states = spla.expm_multiply(big_l, rho0, start=0.0, stop=1.0, num=11)
    t, mean_b = simulate_buffer_response(p, trunc.na, trunc.nb, eps_z, -1, tsave)
    assert np.allclose(t, tsave)
    assert np.allclose(mean_b, states @ vec(b.toarray().T), atol=1e-7)


def test_hermitian_conjugate_modes_are_eigenmodes(params):
    p = params.with_alpha(1.0)
    trunc = FockTruncation(na=6, nb=3)
    big_l = two_mode_liouvillian(p, trunc, eps_z=0.2 * p.g2)
    modes = liouvillian_modes(big_l, sigma=-2.0 + 6.0j, k=4)
    partner = hermitian_conjugate_modes(modes, trunc.dim)
    assert np.allclose(partner.values, np.conj(modes.values))
    _check_modes(partner, big_l, tol=1e-9)


def test_codeword_density_is_a_normalized_coherent_state(params):
    trunc = FockTruncation(na=20, nb=3)
    p = params.with_alpha(1.6)
    rho = codeword_density(p, trunc, sign=-1).reshape(trunc.dim, trunc.dim, order="F")
    a, b = two_mode_operators(trunc)
    assert np.trace(rho) == pytest.approx(1.0)
    assert np.trace(a.toarray() @ rho) == pytest.approx(-1.6, abs=1e-6)
    assert np.trace(b.toarray() @ rho) == pytest.approx(0.0, abs=1e-14)


def test_spectral_modes_are_accurate_and_include_conjugate_partners(cheap):
    _, _, spectrum, _ = cheap
    modes = spectrum.modes
    assert modes.residual < 1e-8
    k = len(modes.values)
    assert np.allclose(modes.left.conj().T @ modes.right, np.eye(k), atol=1e-8)
    for lam in modes.values:
        assert np.min(np.abs(modes.values - np.conj(lam))) < 1e-8 * max(abs(lam), 1.0)


@pytest.mark.parametrize("sign", [1, -1])
def test_spectral_plateau_matches_the_linear_steady_state(cheap, sign):
    p, eps_z, spectrum, _ = cheap
    plateau = spectrum.response(sign).plateau
    linear = LinearBufferResponse.from_params(p, eps_z, sign).steady_state
    assert np.sign(plateau.real) == -sign
    assert plateau.real == pytest.approx(linear, rel=0.01)


def test_common_mode_is_codeword_independent_and_cancels_in_the_differential(cheap):
    p, eps_z, spectrum, _ = cheap
    plus, minus = spectrum.response(1).plateau, spectrum.response(-1).plateau
    assert plus.imag == pytest.approx(minus.imag, rel=1e-8)
    assert plus.imag > 0
    diff = spectrum.differential()
    assert diff.plateau == pytest.approx((plus - minus) / 2, rel=1e-12)
    assert abs(diff.plateau.imag) < 1e-8 * abs(diff.plateau)


def test_spectral_ringing_is_near_the_linear_oscillator_at_cheap_alpha(cheap):
    """At alpha = 1.6: frequency within 6% of nu sqrt(1 - zeta^2); the full model damps
    faster than kappa_b / 4 (measured +29%), so the decay is only bounded within 40%."""
    p, eps_z, spectrum, _ = cheap
    linear = LinearBufferResponse.from_params(p, eps_z, 1)
    ring = spectrum.response(1)
    assert ring.ringing_frequency == pytest.approx(linear.ringing_frequency, rel=0.06)
    assert ring.ringing_frequency_mhz == pytest.approx(ring.ringing_frequency / (2 * math.pi))
    assert linear.damping_rate < ring.decay_rate < 1.4 * linear.damping_rate
    assert ring.ringing_eigenvalue == pytest.approx(
        complex(-ring.decay_rate, ring.ringing_frequency)
    )
    assert abs(ring.ringing_weight) > 0.1 * abs(ring.plateau)


@pytest.mark.parametrize("sign", [1, -1])
def test_modal_sum_matches_the_time_domain_after_the_fast_transient(cheap, sign):
    """The handful of kept modes misses the fast ones: accurate to 1% of the plateau only
    from t = 0.3 us on (measured max 4.3e-3 there); at t = 0 the error is 13%."""
    _, _, spectrum, traces = cheap
    t, b = traces[sign]
    response = spectrum.response(sign)
    err = np.abs(response.mean_b(t) - b) / abs(response.plateau)
    assert np.max(err[t >= 0.3]) < 0.01
    assert err[0] < 0.15


def test_time_domain_signal_has_the_predicted_sign(cheap):
    p, eps_z, _, traces = cheap
    for sign, (_, b) in traces.items():
        assert np.sign(b[-1].real) == -sign
        assert b[-1].real == pytest.approx(-sign * eps_z / (2 * p.alpha * p.g2), rel=0.01)


def test_linear_oscillator_tracks_the_time_domain_differential_signal(cheap):
    """Measured max deviation 12.3% of |b_ss| at alpha = 1.6 (largest near the first peak)."""
    p, eps_z, _, traces = cheap
    t, plus = traces[1]
    _, minus = traces[-1]
    linear = LinearBufferResponse.from_params(p, eps_z, 1)
    diff = (plus - minus) / 2
    assert np.max(np.abs(diff.imag)) < 1e-6 * abs(linear.steady_state)
    assert np.max(np.abs(diff.real - linear.mean_b(t))) < 0.15 * abs(linear.steady_state)


def test_spectral_settling_time_matches_the_time_domain_curve(cheap):
    _, _, spectrum, traces = cheap
    t, b = traces[1]
    response = spectrum.response(1)
    from_curve = settling_time_from_curve(t, b, response.plateau)
    assert response.settling_time() == pytest.approx(from_curve, abs=0.02)


def test_spectral_response_validation(cheap):
    _, _, spectrum, _ = cheap
    with pytest.raises(ValueError, match="sign"):
        spectrum.response(0)
    flat = SpectralBufferResponse(
        values=np.array([-1.0 + 0j]),
        weights=np.array([0.0j]),
        plateau=0j,
        plateau_values=np.array([0j]),
        plateau_weights=np.array([0j]),
        residual=0.0,
        sign=1,
    )
    with pytest.raises(ValueError, match="nonzero"):
        flat.settling_time()


def test_spectral_settling_is_zero_when_already_inside_the_band():
    settled = SpectralBufferResponse(
        values=np.array([-1.0 + 2.0j]),
        weights=np.array([1e-5 + 0j]),
        plateau=1.0 + 0j,
        plateau_values=np.array([0j]),
        plateau_weights=np.array([1.0 + 0j]),
        residual=0.0,
        sign=1,
    )
    assert settled.settling_time() == 0.0


def test_spectral_settling_is_zero_when_cancelling_modes_stay_inside_the_band():
    """Sum |c_j| exceeds the band at t = 0 but the signal itself never leaves it."""
    response = SpectralBufferResponse(
        values=np.array([-1.0 + 0j, -2.0 + 0j]),
        weights=np.array([0.02 + 0j, -0.02 + 0j]),
        plateau=1.0 + 0j,
        plateau_values=np.array([0j]),
        plateau_weights=np.array([1.0 + 0j]),
        residual=0.0,
        sign=1,
    )
    assert response.settling_time() == 0.0


def test_simulate_buffer_response_rejects_a_bad_sign(params):
    with pytest.raises(ValueError, match="sign"):
        simulate_buffer_response(params, 10, 3, 0.1, 2, np.linspace(0, 0.1, 3))


# ---------------------------------------------------------------- physical point alpha = 2.1

# From the N_a in {24, 28, 31} x N_b in {4, 5, 6} sweep at eps_z = 0.05 g2: N_a >= 24 changes
# nothing (< 1e-4 rad/us); N_b 5 -> 6 moves the ringing eigenvalue by <= 0.015 rad/us, the
# plateau by < 1e-6 relative and inflate <b>(t) by 4e-4 |plateau|, far below the tolerances.
PHYSICAL_TRUNC = FockTruncation(na=24, nb=5)


@pytest.fixture(scope="module")
def physical():
    """alpha = 2.1, eps_z = 0.05 g2: spectrum and inflate traces (2-6 min, 0.5 GB)."""
    p = CatParams.reglade()
    eps_z = SMALL_EPS_Z * p.g2
    spectrum = buffer_spectrum(p, PHYSICAL_TRUNC, eps_z)
    na, nb = PHYSICAL_TRUNC.na, PHYSICAL_TRUNC.nb
    traces = {s: simulate_buffer_response(p, na, nb, eps_z, s, T_WINDOW) for s in (1, -1)}
    return p, eps_z, spectrum, traces


@pytest.mark.slow
def test_physical_ringing_matches_the_oscillator_frequency_but_decays_faster(physical):
    """Measured: Im lambda = 20.03 rad/us vs nu sqrt(1 - zeta^2) = 19.72 (+1.6%); decay
    4.71 rad/us vs kappa_b / 4 = 4.08 (+15%). Tolerances: 3% and [1, 1.25] kappa_b / 4."""
    p, eps_z, spectrum, _ = physical
    linear = LinearBufferResponse.from_params(p, eps_z, 1)
    ring = spectrum.response(1)
    assert ring.ringing_frequency == pytest.approx(linear.ringing_frequency, rel=0.03)
    assert linear.damping_rate < ring.decay_rate < 1.25 * linear.damping_rate
    assert spectrum.modes.residual < 1e-8


@pytest.mark.slow
@pytest.mark.parametrize("sign", [1, -1])
def test_physical_plateau_sign_and_value(physical, sign):
    p, eps_z, spectrum, traces = physical
    plateau = spectrum.response(sign).plateau
    expected = -sign * eps_z / (2 * p.alpha * p.g2)
    assert np.sign(plateau.real) == -sign
    assert plateau.real == pytest.approx(expected, rel=0.005)
    assert traces[sign][1][-1] == pytest.approx(plateau, rel=1e-3)


@pytest.mark.slow
@pytest.mark.parametrize("sign", [1, -1])
def test_physical_modal_sum_matches_inflate(physical, sign):
    """Max error 1% of the plateau for t >= 0.3 us; 1% settling times agree to 0.01 us."""
    _, _, spectrum, traces = physical
    t, b = traces[sign]
    response = spectrum.response(sign)
    err = np.abs(response.mean_b(t) - b) / abs(response.plateau)
    assert np.max(err[t >= 0.3]) < 0.01
    from_curve = settling_time_from_curve(t, b, response.plateau)
    assert response.settling_time() == pytest.approx(from_curve, abs=0.01)


@pytest.mark.slow
def test_physical_oscillator_tracks_the_inflate_differential_signal(physical):
    """Measured max deviation 8% of |b_ss| (first peak, t = 0.18 us); tolerance 12%."""
    p, eps_z, _, traces = physical
    t, plus = traces[1]
    _, minus = traces[-1]
    linear = LinearBufferResponse.from_params(p, eps_z, 1)
    diff = (plus - minus) / 2
    assert np.max(np.abs(diff.real - linear.mean_b(t))) < 0.12 * abs(linear.steady_state)
