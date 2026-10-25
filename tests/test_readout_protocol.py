"""Tests for the telegraph-plus-noise readout surrogate, its filters and the SPRT (Task 2.4c)."""

import math

import numpy as np
import pytest

from catqubit.params import CatParams
from catqubit.readout_protocol import (
    FlipEvents,
    HmmTrace,
    RecordModel,
    boxcar_statistic,
    fixed_time_errors,
    gaussian_fixed_error,
    gaussian_fixed_time,
    gaussian_speedup,
    hmm_filter,
    illustrative_flip_rate,
    matched_filter_weights,
    matched_statistic,
    mean_record,
    protocol_stats,
    simulate_records,
    sprt,
    sprt_error,
    sprt_mean_time,
    sprt_stop_steps,
    sprt_threshold,
    state_at,
    step_response_integrals,
)

# A unit-SNR Gaussian latching readout: r = mu^2 = 1 per us, no flips, no ringing.
PURE = RecordModel(mu=1.0, flip_rate=0.0, dt=0.005)


# --------------------------------------------------------------------- model and physics


def test_from_physics_matches_linear_buffer_response(params: CatParams) -> None:
    eps_z = 0.2 * params.g2
    model = RecordModel.from_physics(params, eps_z=eps_z, eta=1.0, flip_rate=1e-6, dt=0.01)
    b_ss = eps_z / (2 * params.alpha * params.g2)
    # |+alpha> gives <b> = -eps_z / (2 alpha g2) < 0: the signed amplitude is negative.
    assert model.mu == pytest.approx(-2 * math.sqrt(params.kappa_b) * b_ss)
    assert model.nu == pytest.approx(2 * params.alpha * params.g2)
    assert model.damping == pytest.approx(params.kappa_b / 4)


def test_measurement_rate_equals_four_eps_sq_over_kappa2_alpha_sq(params: CatParams) -> None:
    eps_z = 0.3 * params.g2
    model = RecordModel.from_physics(params, eps_z=eps_z, eta=1.0, flip_rate=0.0, dt=0.01)
    expected = 4 * eps_z**2 / (params.kappa2 * params.alpha**2)
    assert model.measurement_rate == pytest.approx(expected)


def test_efficiency_and_heterodyne_scale_the_measurement_rate(params: CatParams) -> None:
    kw = {"eps_z": 0.2 * params.g2, "flip_rate": 0.0, "dt": 0.01}
    full = RecordModel.from_physics(params, eta=1.0, **kw)
    lossy = RecordModel.from_physics(params, eta=0.6, **kw)
    het = RecordModel.from_physics(params, eta=0.6, heterodyne=True, **kw)
    assert lossy.measurement_rate == pytest.approx(0.6 * full.measurement_rate)
    assert het.measurement_rate == pytest.approx(0.3 * full.measurement_rate)


def test_illustrative_flip_rate_adds_linear_drive_term(params: CatParams) -> None:
    eps_z = 0.5 * params.g2
    rate = illustrative_flip_rate(params, eps_z, gamma_x0=1e-7)
    drive = 4 * eps_z**2 / (params.kappa2 * params.alpha**2) * math.exp(-2 * params.alpha**2)
    assert rate == pytest.approx(1e-7 + drive)


def test_flip_step_probability_gives_exp_decay_of_mean_state() -> None:
    model = RecordModel(mu=1.0, flip_rate=0.5, dt=0.1)
    p = model.flip_step_probability
    assert (1 - 2 * p) ** 10 == pytest.approx(math.exp(-0.5 * 1.0))


def test_invalid_model_rejected() -> None:
    with pytest.raises(ValueError):
        RecordModel(mu=1.0, flip_rate=-1.0, dt=0.1)
    with pytest.raises(ValueError):
        RecordModel(mu=1.0, flip_rate=0.0, dt=0.0)
    with pytest.raises(ValueError):
        RecordModel(mu=1.0, flip_rate=0.0, dt=0.1, nu=1.0, damping=2.0)  # overdamped
    with pytest.raises(ValueError):
        RecordModel(mu=1.0, flip_rate=0.0, dt=0.1, nu=1.0)  # damping missing
    with pytest.raises(ValueError):
        RecordModel.from_physics(CatParams.reglade(), eps_z=0.1, eta=1.5, flip_rate=0.0, dt=0.1)


# --------------------------------------------------------------------- ringing kernel


def test_step_response_integrals_without_ringing_are_dt() -> None:
    np.testing.assert_allclose(step_response_integrals(PURE, 5), PURE.dt)


def test_step_response_integrals_match_quadrature() -> None:
    nu, gam, dt = 20.0, 4.0, 0.05
    model = RecordModel(mu=1.0, flip_rate=0.0, dt=dt, nu=nu, damping=gam)
    g_steps = step_response_integrals(model, 40)
    wd = math.sqrt(nu**2 - gam**2)
    t = np.linspace(0, 40 * dt, 400_001)
    g = 1 - np.exp(-gam * t) * (np.cos(wd * t) + gam / wd * np.sin(wd * t))
    cum = np.concatenate([[0.0], np.cumsum((g[1:] + g[:-1]) / 2 * np.diff(t))])
    edges = np.searchsorted(t, np.arange(41) * dt - 1e-12)
    np.testing.assert_allclose(g_steps, np.diff(cum[edges]), rtol=1e-6, atol=1e-12)
    # Underdamped: the response overshoots the settled value.
    assert g_steps.max() > dt * 1.2
    assert g_steps[-1] == pytest.approx(dt, rel=1e-2)


# --------------------------------------------------------------------- record generator


def test_mean_record_without_flips_is_s0_times_kernel() -> None:
    model = RecordModel(mu=0.7, flip_rate=0.0, dt=0.05, nu=20.0, damping=4.0)
    s0 = np.array([1.0, -1.0])
    mean = mean_record(model, s0, FlipEvents.empty(), 30)
    g_steps = step_response_integrals(model, 30)
    np.testing.assert_allclose(mean[:, 0], 0.7 * g_steps)
    np.testing.assert_allclose(mean[:, 1], -0.7 * g_steps)


def test_mean_record_flip_inverts_with_same_kernel() -> None:
    model = RecordModel(mu=1.0, flip_rate=0.0, dt=0.05, nu=20.0, damping=4.0)
    flips = FlipEvents(shot=np.array([0]), step=np.array([10]))
    mean = mean_record(model, np.array([1.0]), flips, 60)
    g_steps = step_response_integrals(model, 60)
    expected = g_steps.copy()
    expected[10:] -= 2 * g_steps[:50]
    np.testing.assert_allclose(mean[:, 0], expected)
    assert mean[-1, 0] == pytest.approx(-model.dt, rel=1e-2)


def test_state_at_follows_flips() -> None:
    flips = FlipEvents(shot=np.array([0, 0, 1]), step=np.array([3, 7, 5]))
    s0 = np.array([1.0, -1.0, 1.0])
    np.testing.assert_array_equal(state_at(s0, flips, np.array([2, 2, 9])), [1, -1, 1])
    np.testing.assert_array_equal(state_at(s0, flips, np.array([3, 5, 9])), [-1, 1, 1])
    np.testing.assert_array_equal(state_at(s0, flips, np.array([7, 4, 0])), [1, -1, 1])


def test_records_are_reproducible_with_fixed_seed() -> None:
    model = RecordModel(mu=0.5, flip_rate=0.3, dt=0.05)
    a = simulate_records(model, 200, 100, np.random.default_rng(7))
    b = simulate_records(model, 200, 100, np.random.default_rng(7))
    np.testing.assert_array_equal(a.y, b.y)
    np.testing.assert_array_equal(a.flips.step, b.flips.step)
    c = simulate_records(model, 200, 100, np.random.default_rng(8))
    assert not np.array_equal(a.y, c.y)


def test_flip_statistics_match_telegraph_rate() -> None:
    model = RecordModel(mu=0.0, flip_rate=0.4, dt=0.05)
    batch = simulate_records(model, 20_000, 100, np.random.default_rng(1))
    s_end = state_at(batch.s0, batch.flips, np.full(batch.s0.size, 99))
    mean_same = np.mean(s_end * batch.s0)
    assert mean_same == pytest.approx(math.exp(-0.4 * 99 * 0.05), abs=0.02)
    flips_per_shot = batch.flips.step.size / 20_000
    assert flips_per_shot == pytest.approx(model.flip_step_probability * 99, rel=0.05)


def test_integrated_record_mean_and_variance_without_flips() -> None:
    model = RecordModel(mu=0.8, flip_rate=0.0, dt=0.02, nu=20.0, damping=4.0)
    batch = simulate_records(model, 20_000, 150, np.random.default_rng(3), s0=1.0)
    total = batch.y.sum(axis=0)
    expected_mean = 0.8 * step_response_integrals(model, 150).sum()
    t = 150 * model.dt
    assert total.mean() == pytest.approx(expected_mean, abs=4 * math.sqrt(t / 20_000))
    assert total.var() == pytest.approx(t, rel=0.03)


# --------------------------------------------------------------------- linear filters


def test_boxcar_statistic_is_signed_running_sum() -> None:
    model = RecordModel(mu=-1.0, flip_rate=0.0, dt=0.1)
    y = np.array([[1.0, 0.0], [2.0, -1.0]])
    np.testing.assert_allclose(boxcar_statistic(model, y), [[-1.0, 0.0], [-3.0, 1.0]])


def test_matched_weights_reduce_to_boxcar_and_decay_with_flips() -> None:
    np.testing.assert_allclose(matched_filter_weights(PURE, 4), PURE.dt)
    model = RecordModel(mu=1.0, flip_rate=1.0, dt=0.1)
    w = matched_filter_weights(model, 30)
    np.testing.assert_allclose(w, 0.1 * np.exp(-1.0 * 0.1 * np.arange(30)), rtol=1e-12)


def test_matched_filter_beats_boxcar_when_flips_matter() -> None:
    model = RecordModel(mu=1.0, flip_rate=0.5, dt=0.02)
    batch = simulate_records(model, 40_000, 300, np.random.default_rng(11))
    checks = np.array([299])
    e_box = fixed_time_errors(boxcar_statistic(model, batch.y), batch.s0, checks)[0]
    e_mf = fixed_time_errors(matched_statistic(model, batch.y), batch.s0, checks)[0]
    assert e_mf < e_box


def test_fixed_time_boxcar_reproduces_erfc() -> None:
    batch = simulate_records(PURE, 40_000, 1000, np.random.default_rng(5))
    checks = np.array([249, 499, 999])  # t = 1.25, 2.5, 5 us
    errors = fixed_time_errors(boxcar_statistic(PURE, batch.y), batch.s0, checks)
    expected = np.array([gaussian_fixed_error(1.0, (k + 1) * PURE.dt) for k in checks])
    np.testing.assert_allclose(errors, expected, rtol=0.1)


# --------------------------------------------------------------------- HMM filter


def test_hmm_without_flips_equals_gaussian_log_likelihood_ratio() -> None:
    batch = simulate_records(PURE, 50, 200, np.random.default_rng(2))
    trace = hmm_filter(PURE, batch.y)
    np.testing.assert_allclose(trace.log_odds, 2 * boxcar_statistic(PURE, batch.y), atol=1e-9)
    np.testing.assert_allclose(trace.flip_prob, 0.0, atol=1e-12)
    np.testing.assert_allclose(trace.branch_gap, 1.0)


def test_hmm_log_odds_is_calibrated() -> None:
    model = RecordModel(mu=1.0, flip_rate=0.5, dt=0.02)
    batch = simulate_records(model, 40_000, 200, np.random.default_rng(4))
    trace = hmm_filter(model, batch.y)
    p_plus = 1 / (1 + np.exp(-trace.log_odds[-1]))
    bins = np.linspace(0, 1, 6)
    idx = np.digitize(p_plus, bins[1:-1])
    for b in range(5):
        sel = idx == b
        tol = 0.01 + 3 * math.sqrt(0.25 / max(sel.sum(), 1))
        assert np.mean(batch.s0[sel] > 0) == pytest.approx(p_plus[sel].mean(), abs=tol)


def test_hmm_recovers_planted_flip() -> None:
    model = RecordModel(mu=3.0, flip_rate=0.05, dt=0.01, nu=20.0, damping=4.0)
    n, flip_step = 200, 300
    flips = FlipEvents(shot=np.arange(n), step=np.full(n, flip_step))
    s0 = np.ones(n)
    rng = np.random.default_rng(9)
    y = mean_record(model, s0, flips, 600) + math.sqrt(model.dt) * rng.standard_normal((600, n))
    trace = hmm_filter(model, y)
    assert np.mean(trace.log_odds[-1] > 0) > 0.95  # still reads the initial state
    assert np.median(trace.flip_prob[flip_step - 1]) < 0.05
    assert np.median(trace.flip_prob[-1]) > 0.95
    detect = np.argmax(trace.flip_prob > 0.5, axis=0)
    assert np.median(detect) - flip_step < 100  # detected within 1 us
    # Early on the two s0 branches disagree about s_k; once the s0 evidence has saturated
    # (here long before the flip) both branches agree on s_k and later data cannot help.
    assert np.median(trace.branch_gap[10]) > 0.9
    assert np.median(np.abs(trace.branch_gap[flip_step - 1])) < 0.01
    assert np.median(np.abs(trace.branch_gap[-1])) < 0.01


def test_hmm_beats_matched_filter_with_frequent_flips() -> None:
    model = RecordModel(mu=1.0, flip_rate=0.5, dt=0.02)
    batch = simulate_records(model, 40_000, 300, np.random.default_rng(12))
    checks = np.array([299])
    e_mf = fixed_time_errors(matched_statistic(model, batch.y), batch.s0, checks)[0]
    e_hmm = fixed_time_errors(hmm_filter(model, batch.y).log_odds, batch.s0, checks)[0]
    assert e_hmm < e_mf


# --------------------------------------------------------------------- SPRT and erasure


def test_wald_expected_sample_number_on_pure_gaussian() -> None:
    a = 6.0
    batch = simulate_records(PURE, 10_000, 3000, np.random.default_rng(21))
    result = sprt(hmm_filter(PURE, batch.y), threshold=a, max_steps=3000)
    mean_t = (result.stop_step + 1).mean() * PURE.dt
    delta_mu_sq_over_sigma_sq = (2 * PURE.mu) ** 2
    # Wald: E[T] (Delta mu)^2 / sigma^2 = 2 A (continuous limit, tanh(A/2) ~ 1).
    assert mean_t * delta_mu_sq_over_sigma_sq == pytest.approx(2 * a * math.tanh(a / 2), rel=0.04)
    assert mean_t == pytest.approx(sprt_mean_time(a, 1.0), rel=0.04)
    assert np.mean(result.decision != batch.s0) == pytest.approx(1 / (1 + math.exp(a)), rel=0.3)


def test_dAnjou_speedup_analytic() -> None:
    assert gaussian_speedup(1e-3) == pytest.approx(2.77, abs=0.02)
    assert gaussian_speedup(1e-30) > 3.5
    assert gaussian_speedup(1e-30) < 4.0
    assert gaussian_fixed_error(1.0, gaussian_fixed_time(1e-4, 1.0)) == pytest.approx(1e-4)
    assert sprt_threshold(1e-3) == pytest.approx(math.log(999))
    assert sprt_error(sprt_threshold(1e-3)) == pytest.approx(1e-3)


def test_sprt_approaches_dAnjou_bound_in_simulation() -> None:
    a = sprt_threshold(1e-3)
    batch = simulate_records(PURE, 20_000, 3000, np.random.default_rng(22))
    result = sprt(hmm_filter(PURE, batch.y), threshold=a, max_steps=3000)
    eps = np.mean(result.decision != batch.s0)
    mean_t = (result.stop_step + 1).mean() * PURE.dt
    speedup = gaussian_fixed_time(eps, 1.0) / mean_t
    assert 2.5 < speedup < 3.0


def test_sprt_cap_and_erasure_flags() -> None:
    log_odds = np.array([[0.5, 0.2, 0.1], [3.0, -0.5, 0.3], [0.2, -2.5, 0.6]])
    flip_prob = np.array([[0.0, 0.0, 0.0], [0.6, 0.0, 0.0], [0.6, 0.0, 0.9]])
    trace = HmmTrace(log_odds=log_odds, flip_prob=flip_prob)
    sweep = sprt_stop_steps(trace, np.array([2.0, 10.0]), max_steps=3)
    np.testing.assert_array_equal(sweep, [[1, 2, 2], [2, 2, 2]])
    plain = sprt(trace, threshold=2.0, max_steps=3)
    np.testing.assert_array_equal(plain.stop_step, [1, 2, 2])
    np.testing.assert_array_equal(plain.decision, [1, -1, 1])
    assert not plain.erased.any()
    flagged = sprt(trace, threshold=2.0, max_steps=3, theta=0.5, erase_undecided=True)
    np.testing.assert_array_equal(flagged.erased, [True, False, True])
    gap = np.array([[1.0, 1.0, 1.0], [1.0, 0.001, 1.0], [1.0, 1.0, 0.001]])
    frozen = HmmTrace(log_odds=log_odds, flip_prob=flip_prob, branch_gap=gap)
    early = sprt(frozen, threshold=2.0, max_steps=3, erase_undecided=True, freeze_gap=0.01)
    np.testing.assert_array_equal(early.stop_step, [1, 1, 2])
    np.testing.assert_array_equal(early.erased, [False, True, True])  # undecided when frozen
    with pytest.raises(ValueError):
        sprt(trace, threshold=2.0, max_steps=3, freeze_gap=0.01)  # trace has no branch gap
    capped = sprt(trace, threshold=2.0, max_steps=2)
    np.testing.assert_array_equal(capped.stop_step, [1, 1, 1])
    with pytest.raises(ValueError):
        sprt(trace, threshold=-1.0, max_steps=3)
    with pytest.raises(ValueError):
        sprt(trace, threshold=1.0, max_steps=4)


def test_hmm_log_odds_saturate_when_flips_are_possible() -> None:
    # An early flip cannot be told apart from the other initial codeword, so the s0
    # log-odds saturate near ln(r / Gamma_X) + O(3), independently of dt.
    for dt in (0.02, 0.005):
        model = RecordModel(mu=1.0, flip_rate=0.3, dt=dt)
        batch = simulate_records(model, 2000, int(12 / dt), np.random.default_rng(13))
        final = np.abs(hmm_filter(model, batch.y).log_odds[-1])
        assert np.quantile(final, 0.99) < math.log(1 / 0.3) + 4.0


def test_freeze_stop_cuts_the_saturation_tail() -> None:
    # Gamma_X / r = 0.05 puts the s0 log-odds saturation (~ln(r/Gamma)+3 ~ 6) below A = 8.
    model = RecordModel(mu=1.0, flip_rate=0.05, dt=0.02)
    batch = simulate_records(model, 10_000, 1500, np.random.default_rng(41))
    trace = hmm_filter(model, batch.y)
    capped = protocol_stats(sprt(trace, 8.0, 1500), batch, model)
    frozen = protocol_stats(sprt(trace, 8.0, 1500, freeze_gap=0.01), batch, model)
    assert frozen.mean_time < 0.5 * capped.mean_time
    assert frozen.p_error < capped.p_error + 0.003


def test_erasure_converts_flip_errors_into_erasures() -> None:
    model = RecordModel(mu=1.0, flip_rate=0.05, dt=0.02)
    batch = simulate_records(model, 20_000, 1500, np.random.default_rng(31))
    trace = hmm_filter(model, batch.y)
    a = 4.0
    plain = protocol_stats(sprt(trace, a, 1500), batch, model)
    erasing = protocol_stats(sprt(trace, a, 1500, theta=0.5, erase_undecided=True), batch, model)
    assert erasing.p_erasure > 0
    assert erasing.p_error + erasing.p_hidden < plain.p_error + plain.p_hidden
    assert plain.p_erasure == 0
    assert 0 < plain.mean_time < 1500 * model.dt
    assert plain.time_quantiles[0.5] <= plain.time_quantiles[0.99]


def test_protocol_error_rates_reproducible_with_seed() -> None:
    model = RecordModel(mu=1.0, flip_rate=0.3, dt=0.02)

    def run(seed: int) -> tuple[float, float]:
        batch = simulate_records(model, 5000, 500, np.random.default_rng(seed))
        stats = protocol_stats(sprt(hmm_filter(model, batch.y), 4.0, 500), batch, model)
        return stats.p_error, stats.mean_time

    assert run(5) == run(5)


# --------------------------------------------------------------------- full-model validation


@pytest.mark.slow
def test_surrogate_matches_two_mode_sse_trajectories() -> None:
    """Integrated buffer record of dq.dssesolve trajectories (alpha = 1.2, eps_Z = 0.2 g2) vs
    the surrogate. dynamiqs: dY = <L + L^dag> dt + dW, measurements = bin averages of dY/dt."""
    import dynamiqs as dq
    import jax
    import jax.numpy as jnp

    from catqubit.liouvillian import FockTruncation
    from catqubit.rates import bit_flip_rate

    dq.set_progress_meter(False)
    alpha, na, nb, n_traj, dt_save, t_end = 1.2, 14, 4, 1000, 0.05, 2.0
    p = CatParams.reglade().with_alpha(alpha)
    eps_z = 0.2 * p.g2
    a, b = dq.destroy(na, nb)
    h = p.g2 * (a @ a @ b.dag() + a.dag() @ a.dag() @ b) - p.eps_d * (b + b.dag())
    h = h + eps_z * (a + a.dag())
    jumps = [jnp.sqrt(p.kappa_b) * b, jnp.sqrt(p.kappa_a) * a]
    psis = dq.stack([dq.tensor(dq.coherent(na, s * alpha), dq.basis(nb, 0)) for s in (1, -1)])
    tsave = tuple(float(x) for x in np.round(np.arange(int(t_end / dt_save) + 1) * dt_save, 12))
    keys = jax.random.split(jax.random.key(3), n_traj)
    res = dq.dssesolve(
        h,
        jumps,
        psis,
        tsave,
        keys,
        method=dq.method.Rouchon1(dt=1e-3, normalize=False),
        save_states=False,
    )
    total = np.asarray(res.measurements)[:, :, 0, :].sum(axis=-1) * dt_save  # (2, n_traj)

    gamma = illustrative_flip_rate(p, eps_z, bit_flip_rate(p, FockTruncation(na=18, nb=6)))
    model = RecordModel.from_physics(p, eps_z, eta=1.0, flip_rate=gamma, dt=0.005)
    n_steps = int(round(t_end / model.dt))
    surrogate_mean = abs(model.mu * matched_filter_weights(model, n_steps).sum())
    batch = simulate_records(model, 20_000, n_steps, np.random.default_rng(4), s0=1.0)
    surrogate_var = batch.y.sum(axis=0).var()
    for sign, row in zip((-1, 1), total, strict=True):  # |+alpha> gives Re<b> < 0
        sem = row.std(ddof=1) / math.sqrt(n_traj)
        # Measured offsets: steady <b> 1.7% below the linear estimate; the ring damps faster.
        assert row.mean() == pytest.approx(
            sign * surrogate_mean, abs=3 * sem + 0.08 * surrogate_mean
        )
        assert row.var(ddof=1) == pytest.approx(surrogate_var, rel=0.15)
