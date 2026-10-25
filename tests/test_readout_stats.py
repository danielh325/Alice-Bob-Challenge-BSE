"""Exact readout-record statistics from the tilted Liouvillian (Task 2.3, plan R8)."""

import math

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.special import erfc

from catqubit.liouvillian import FockTruncation, lindblad_superoperator, two_mode_operators, vec
from catqubit.readout_stats import (
    BOXCAR,
    FourierGrid,
    Homodyne,
    PiecewiseFilter,
    SlowModes,
    TiltedGenerator,
    cat_readout_generator,
    chernoff_bound,
    codeword_state,
    fd_cumulants,
    gaussian_error,
    generating_function,
    homodyne_generator,
    matched_filter_error,
    readout_error,
    record_cumulants,
    tilt_superoperator,
    tilted_trace,
)

CAVITY_DIM = 26
CAVITY_KAPPA = 2.0
CAVITY_BETA = 0.9 + 0.3j


# ---------- helpers: a damped cavity whose homodyne record is exactly Gaussian ----------


def _destroy(n: int) -> sp.csr_matrix:
    return sp.diags(np.sqrt(np.arange(1, n, dtype=float)), 1, format="csr").astype(complex)


def _coherent_vec(n: int, beta: complex) -> np.ndarray:
    k = np.arange(n)
    log_fact = np.array([math.lgamma(x + 1) for x in k])
    amp = np.exp(k * np.log(complex(beta)) - 0.5 * log_fact - abs(beta) ** 2 / 2)
    return amp / np.linalg.norm(amp)


def _cavity(detector: Homodyne | None = None) -> tuple[TiltedGenerator, list[np.ndarray]]:
    """Free decay of a cavity, monitored on its output: coherent states stay coherent."""
    detector = Homodyne() if detector is None else detector
    a = _destroy(CAVITY_DIM)
    c = math.sqrt(CAVITY_KAPPA) * a
    big_l = lindblad_superoperator(sp.csr_matrix((CAVITY_DIM, CAVITY_DIM), dtype=complex), [c])
    rhos = []
    for beta in (CAVITY_BETA, -CAVITY_BETA):
        psi = _coherent_vec(CAVITY_DIM, beta)
        rhos.append(vec(np.outer(psi, psi.conj())))
    return homodyne_generator(big_l, c, detector), rhos


def _cavity_mean(times: np.ndarray, beta: complex, detector: Homodyne) -> np.ndarray:
    """2 sqrt(eta kappa) int_0^T Re(beta e^{-kappa t/2} e^{-i phi}) dt."""
    k = CAVITY_KAPPA
    area = 2 / k * (1 - np.exp(-k * times / 2))
    return 2 * math.sqrt(detector.eta * k) * (beta * np.exp(-1j * detector.phi)).real * area


@pytest.fixture(scope="module")
def small_cat():
    from catqubit.params import CatParams

    p = CatParams.reglade().with_alpha(1.0)
    trunc = FockTruncation(na=10, nb=3)
    gen = cat_readout_generator(p, trunc, eps_z=0.3 * p.g2)
    rhos = [codeword_state(p, trunc, +1), codeword_state(p, trunc, -1)]
    return p, trunc, gen, rhos


# ---------- tilt operator and generator ----------


def test_tilt_superoperator_acts_as_sqrt_eta_c_rho_plus_rho_c_dag():
    rng = np.random.default_rng(1)
    n = 5
    c = sp.csr_matrix(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
    rho = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    det = Homodyne(eta=0.6, phi=0.4)
    ce = c.toarray() * np.exp(-1j * det.phi)
    expected = math.sqrt(det.eta) * (ce @ rho + rho @ ce.conj().T)
    assert np.allclose(tilt_superoperator(c, det) @ vec(rho), vec(expected))


@pytest.mark.parametrize("eta", [0.0, -0.1, 1.5])
def test_homodyne_rejects_unphysical_efficiency(eta):
    with pytest.raises(ValueError, match="eta"):
        Homodyne(eta=eta)


def test_codeword_state_is_a_normalized_projector_with_the_right_sign(small_cat):
    p, trunc, _, (rho_p, rho_m) = small_cat
    a, _ = two_mode_operators(trunc)
    n = trunc.dim
    for rho, sign in ((rho_p, 1), (rho_m, -1)):
        mat = rho.reshape(n, n, order="F")
        assert np.trace(mat).real == pytest.approx(1.0)
        assert np.allclose(mat @ mat, mat)
        assert np.trace(a @ mat).real == pytest.approx(sign * p.alpha, rel=1e-5)


def test_codeword_state_rejects_bad_sign(small_cat):
    p, trunc, _, _ = small_cat
    with pytest.raises(ValueError, match="sign"):
        codeword_state(p, trunc, 0)


def test_cat_generator_uses_the_buffer_output_field(small_cat):
    p, trunc, gen, _ = small_cat
    _, b = two_mode_operators(trunc)
    expected = tilt_superoperator(math.sqrt(p.kappa_b) * b, Homodyne())
    assert abs(gen.tilt - expected).max() < 1e-14
    assert gen.liouvillian.shape == (trunc.dim**2, trunc.dim**2)


# ---------- filters ----------


def test_piecewise_filter_squared_integral_and_breakpoints():
    filt = PiecewiseFilter(edges=(0.0, 1.0, 3.0), weights=(2.0, 0.5, 1.0))
    times = np.array([0.5, 1.0, 2.0, 5.0])
    assert np.allclose(filt.squared_integral(times), [2.0, 4.0, 4.25, 4.5 + 2.0])
    assert filt.weight(0.2) == 2.0 and filt.weight(3.5) == 1.0
    assert BOXCAR.squared_integral(np.array([2.0]))[0] == pytest.approx(2.0)


@pytest.mark.parametrize(
    "edges, weights",
    [((0.5,), (1.0,)), ((0.0, 2.0, 1.0), (1.0, 1.0, 1.0)), ((0.0, 1.0), (1.0,))],
)
def test_piecewise_filter_rejects_bad_edges(edges, weights):
    with pytest.raises(ValueError):
        PiecewiseFilter(edges=edges, weights=weights)


# ---------- generating function ----------


def test_generating_function_is_one_at_s_zero(small_cat):
    _, _, gen, rhos = small_cat
    z = generating_function(gen, rhos, [0.0], np.array([0.5, 2.0]))
    assert np.allclose(z, 1.0, atol=1e-10)


def test_tilted_trace_rejects_unsorted_times(small_cat):
    _, _, gen, rhos = small_cat
    with pytest.raises(ValueError, match="times"):
        tilted_trace(gen, rhos, 0.1, np.array([2.0, 1.0]))


def test_cavity_record_is_exactly_gaussian_in_the_generating_function():
    det = Homodyne(eta=0.7, phi=0.3)
    gen, rhos = _cavity(det)
    times = np.array([0.4, 1.0, 2.5])
    s_values = [0.5, -0.8, 1.3j]
    z = generating_function(gen, rhos[:1], s_values, times)[0]
    mean = _cavity_mean(times, CAVITY_BETA, det)
    for i, s in enumerate(s_values):
        assert np.allclose(z[i], np.exp(s * mean + s**2 * times / 2), rtol=1e-8)


def test_exact_cumulants_match_finite_differences(small_cat):
    _, _, gen, rhos = small_cat
    times = np.array([0.5, 1.5, 3.0])
    filt = PiecewiseFilter(edges=(0.0, 1.0), weights=(0.5, 1.0))
    for rho in rhos:
        mean, var = record_cumulants(gen, rho, times, filt)
        fd_mean, fd_var = fd_cumulants(gen, [rho], times, filt)
        assert np.allclose(fd_mean[0], mean, rtol=1e-7, atol=1e-9)
        assert np.allclose(fd_var[0], var, rtol=1e-6)


def test_cat_record_variance_exceeds_shot_noise_and_means_have_opposite_signs(small_cat):
    _, _, gen, rhos = small_cat
    times = np.array([2.0])
    (m_p, v_p), (m_m, v_m) = (record_cumulants(gen, r, times) for r in rhos)
    assert m_p[0] < 0 < m_m[0]
    assert v_p[0] > times[0] and v_m[0] > times[0]


def test_slow_mode_continuation_matches_exact_propagation(small_cat):
    _, _, gen, rhos = small_cat
    times = np.array([1.0, 4.0, 5.0, 7.0])
    slow = SlowModes(t_switch=4.0, n_modes=8)
    for s in (0.7j, 2.0j, 0.4, -0.6):
        exact = tilted_trace(gen, rhos, s, times)
        spectral = tilted_trace(gen, rhos, s, times, slow=slow)
        assert np.allclose(spectral, exact, rtol=1e-7, atol=1e-12)


def test_slow_modes_from_the_initial_state_match_exact_propagation(small_cat):
    _, _, gen, rhos = small_cat
    times = np.array([2.0, 5.0, 7.0])
    slow = SlowModes(t_switch=4.0, n_modes=8, from_start=True)
    for s in (0.7j, 0.4):
        exact = tilted_trace(gen, rhos, s, times)
        spectral = tilted_trace(gen, rhos, s, times, slow=slow)
        assert np.allclose(spectral, exact, rtol=1e-6, atol=1e-10)


def test_from_start_needs_a_constant_filter_and_reports_no_switch_error(small_cat):
    _, _, gen, (rho_p, rho_m) = small_cat
    filt = PiecewiseFilter(edges=(0.0, 1.0), weights=(0.5, 1.0))
    slow = SlowModes(t_switch=4.0, from_start=True)
    with pytest.raises(ValueError, match="constant filter"):
        tilted_trace(gen, [rho_p], 0.5j, np.array([5.0]), filt, slow=slow)
    res = readout_error(gen, rho_p, rho_m, np.array([1.0, 5.0]), slow=slow)
    assert math.isnan(res.switch_error)


@pytest.mark.parametrize("kwargs", [dict(t_switch=0.0), dict(t_switch=1.0, n_modes=1)])
def test_slow_modes_validation(kwargs):
    with pytest.raises(ValueError):
        SlowModes(**kwargs)


def test_chernoff_rejects_nonpositive_s(small_cat):
    _, _, gen, (rho_p, rho_m) = small_cat
    with pytest.raises(ValueError, match="positive"):
        chernoff_bound(gen, rho_p, rho_m, np.array([1.0]), np.zeros(1), np.array([0.0, 0.1]))


def test_slow_mode_switch_must_follow_the_last_filter_edge(small_cat):
    _, _, gen, rhos = small_cat
    filt = PiecewiseFilter(edges=(0.0, 5.0), weights=(1.0, 0.5))
    with pytest.raises(ValueError, match="t_switch"):
        tilted_trace(gen, rhos, 0.5j, np.array([6.0]), filt, slow=SlowModes(t_switch=4.0))


def test_generating_function_parallel_matches_serial(small_cat):
    _, _, gen, rhos = small_cat
    times = np.array([1.0, 2.0])
    s_values = [0.3j, 0.9j, 0.2]
    serial = generating_function(gen, rhos, s_values, times)
    parallel = generating_function(gen, rhos, s_values, times, workers=2)
    assert np.allclose(parallel, serial, rtol=1e-12)


# ---------- Fourier grid and inversion ----------


def test_fourier_grid_covers_the_requested_window_and_bandwidth():
    grid = FourierGrid.covering(half_width=20.0, f2_min=0.5)
    assert grid.window >= 40.0
    assert math.exp(-(grid.k_max**2) * 0.5 / 2) <= 1e-16
    assert grid.ks[0] == 0.0 and np.allclose(np.diff(grid.ks), grid.dk)


def test_cavity_readout_error_equals_gaussian_erfc():
    det = Homodyne(eta=0.8)
    gen, (rho_p, rho_m) = _cavity(det)
    times = np.array([0.3, 1.0, 2.0])
    res = readout_error(gen, rho_p, rho_m, times)
    m = _cavity_mean(times, CAVITY_BETA, det)
    exact = 0.5 * erfc(np.abs(m) / np.sqrt(2 * times))
    assert np.allclose(res.error_threshold, exact, rtol=1e-8)
    assert np.allclose(res.error_bayes, exact, rtol=1e-3)
    assert np.allclose(res.thresholds, 0.0, atol=1e-6)
    assert np.allclose(res.error_gauss, exact, rtol=1e-6)
    for dist in res.distributions:
        dq_ = dist.q[1] - dist.q[0]
        assert np.sum(dist.p_plus) * dq_ == pytest.approx(1.0, abs=1e-10)


def test_cavity_density_matches_gaussian_pdf():
    gen, (rho_p, rho_m) = _cavity()
    times = np.array([1.5])
    res = readout_error(gen, rho_p, rho_m, times)
    dist = res.distributions[0]
    mean = _cavity_mean(times, CAVITY_BETA, Homodyne())[0]
    pdf = np.exp(-((dist.q - mean) ** 2) / (2 * 1.5)) / math.sqrt(2 * math.pi * 1.5)
    assert np.max(np.abs(dist.p_plus - pdf)) < 1e-12


def test_piecewise_filter_beats_boxcar_and_is_bounded_by_the_matched_filter():
    det = Homodyne()
    gen, (rho_p, rho_m) = _cavity(det)
    times = np.array([3.0])
    edges = tuple(np.arange(0.0, 3.0, 0.25))
    filt = PiecewiseFilter(edges=edges, weights=tuple(np.exp(-CAVITY_KAPPA * np.array(edges) / 2)))
    box = readout_error(gen, rho_p, rho_m, times).error_threshold[0]
    shaped = readout_error(gen, rho_p, rho_m, times, filt).error_threshold[0]
    t = np.linspace(0, 3.0, 3001)
    # homodyne at phi = 0 sees only Re<a>: the matched-filter formula needs that quadrature
    delta_b = 2 * CAVITY_BETA.real * np.exp(-CAVITY_KAPPA * t / 2)
    matched = matched_filter_error(delta_b, t, CAVITY_KAPPA, det.eta)[-1]
    assert matched < shaped < box
    assert shaped == pytest.approx(matched, rel=0.05)


def test_matched_filter_error_formula():
    t = np.linspace(0.0, 2.0, 2001)
    delta_b = np.full_like(t, 0.5, dtype=complex)
    err = matched_filter_error(delta_b, t, kappa_b=4.0, eta=0.5)
    assert err[0] == pytest.approx(0.5)
    assert err[-1] == pytest.approx(0.5 * erfc(math.sqrt(0.5 * 4.0 / 2 * 0.25 * 2.0)), rel=1e-9)


def test_gaussian_error_with_unequal_variances():
    err = gaussian_error(np.array([-1.0]), np.array([2.0]), np.array([1.0]), np.array([4.0]))
    assert err[0] == pytest.approx(0.5 * erfc(3.0 / (math.sqrt(2) * 3.0)))


def test_cat_distribution_is_normalized_and_nonnegative(small_cat):
    _, _, gen, (rho_p, rho_m) = small_cat
    res = readout_error(gen, rho_p, rho_m, np.array([1.0, 3.0]))
    for dist in res.distributions:
        dq_ = dist.q[1] - dist.q[0]
        for p in (dist.p_plus, dist.p_minus):
            assert np.sum(p) * dq_ == pytest.approx(1.0, abs=1e-9)
            assert p.min() > -1e-10
    assert np.all(res.error_bayes <= res.error_threshold + 1e-9)
    assert np.all(res.error_threshold < 0.5)


@pytest.mark.slow
def test_cat_long_window_with_slow_modes_matches_exact(small_cat):
    _, _, gen, (rho_p, rho_m) = small_cat
    times = np.array([1.0, 3.0, 6.0])
    exact = readout_error(gen, rho_p, rho_m, times)
    fast = readout_error(gen, rho_p, rho_m, times, slow=SlowModes(t_switch=3.0, n_modes=8))
    assert np.allclose(fast.error_threshold, exact.error_threshold, rtol=1e-6)
    # fast modes decay at about kappa_b / 4, so e^{-kappa_b t_switch / 4} ~ 5e-6 is dropped
    assert fast.switch_error < 1e-5


# ---------- Chernoff bound ----------


@pytest.mark.slow
def test_chernoff_bound_on_a_gaussian_record():
    gen, (rho_p, rho_m) = _cavity()
    times = np.array([1.0, 2.0])
    s_values = np.linspace(0.02, 3.0, 150)
    bound = chernoff_bound(gen, rho_p, rho_m, times, np.zeros(2), s_values)
    m = np.abs(_cavity_mean(times, CAVITY_BETA, Homodyne()))
    exact = 0.5 * erfc(m / np.sqrt(2 * times))
    assert np.all(bound >= exact)
    assert np.allclose(bound, np.exp(-(m**2) / (2 * times)), rtol=2e-3)


@pytest.mark.slow
def test_chernoff_bound_exceeds_exact_cat_error(small_cat):
    _, _, gen, (rho_p, rho_m) = small_cat
    times = np.array([1.0, 4.0])
    res = readout_error(gen, rho_p, rho_m, times)
    bound = chernoff_bound(gen, rho_p, rho_m, times, res.thresholds, np.linspace(0.05, 1.5, 15))
    assert np.all(bound >= res.error_threshold)
    assert np.all(bound < 1.0)


# ---------- stochastic trajectories (independent check) ----------


@pytest.mark.slow
def test_distribution_matches_stochastic_trajectories(small_cat):
    import dynamiqs as dq
    import jax
    from scipy import stats

    p, trunc, gen, (rho_p, rho_m) = small_cat
    a, b = two_mode_operators(trunc)
    from catqubit.liouvillian import two_mode_hamiltonian

    h = two_mode_hamiltonian(p, trunc, eps_z=0.3 * p.g2).toarray()
    jumps = [math.sqrt(p.kappa_b) * b.toarray(), math.sqrt(p.kappa_a) * a.toarray()]
    times = np.array([1.0, 2.0, 3.0])
    res = readout_error(gen, rho_p, rho_m, times)
    n_traj, bin_ = 1500, 0.5
    tsave = np.arange(0.0, times[-1] + 1e-9, bin_)
    for sign, rho in ((1, rho_p), (-1, rho_m)):
        n = trunc.dim
        w, v = np.linalg.eigh(rho.reshape(n, n, order="F"))
        psi0 = v[:, -1][:, None]
        keys = jax.random.split(jax.random.key(11 + sign), n_traj)
        out = dq.dssesolve(
            h, jumps, psi0, tsave, keys, method=dq.method.Rouchon1(dt=1e-3), save_states=False
        )
        q = np.cumsum(np.asarray(out.measurements)[:, 0, :] * bin_, axis=1)
        mean, var = record_cumulants(gen, rho, times)
        for j, t in enumerate(times):
            col = q[:, int(round(t / bin_)) - 1]
            assert abs(col.mean() - mean[j]) < 4 * math.sqrt(var[j] / n_traj)
            dist = res.distributions[j]
            pdf = dist.p_plus if sign == 1 else dist.p_minus
            cdf = np.cumsum(pdf) * (dist.q[1] - dist.q[0])
            assert stats.kstest(col, lambda x, q=dist.q, c=cdf: np.interp(x, q, c)).pvalue > 1e-3
