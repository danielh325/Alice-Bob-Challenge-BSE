"""Readout optimization under a bit-flip budget (Task 2.4c)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from catqubit.optimize import (
    HermiteTable,
    LinearReadout,
    ReadoutProblem,
    augmented_lagrangian,
    deadbeat_two_step,
    gaussian_error,
)


def _toy_objective(x):
    return (x[0] - 2) ** 2 + (x[1] - 1) ** 2, np.array([2 * (x[0] - 2), 2 * (x[1] - 1)])


def test_augmented_lagrangian_finds_the_active_constraint_optimum():
    res = augmented_lagrangian(
        _toy_objective, lambda x: (x[0] + x[1] - 1, np.array([1.0, 1.0])), np.zeros(2)
    )
    assert res.converged
    assert np.allclose(res.x, [1.0, 0.0], atol=1e-5)
    assert res.multiplier == pytest.approx(2.0, rel=1e-4)


def test_augmented_lagrangian_with_an_inactive_constraint():
    res = augmented_lagrangian(
        _toy_objective, lambda x: (x[0] + x[1] - 10, np.array([1.0, 1.0])), np.zeros(2)
    )
    assert np.allclose(res.x, [2.0, 1.0], atol=1e-6)
    assert res.multiplier == 0.0


def test_hermite_table_reproduces_a_cubic_and_its_slope():
    knots = np.array([0.0, 0.3, 1.0, 1.7])
    f = lambda x: 2 + x - 0.5 * x**2 + 0.25 * x**3  # noqa: E731
    df = lambda x: 1 - x + 0.75 * x**2  # noqa: E731
    table = HermiteTable.build(knots, f(knots), df(knots))
    x = jnp.linspace(0.0, 1.7, 23)
    assert np.allclose(table(x), f(np.asarray(x)), atol=1e-12)
    assert np.allclose(jax.vmap(jax.grad(table))(x), df(np.asarray(x)), atol=1e-10)


def test_hermite_table_rejects_points_outside_its_range():
    table = HermiteTable.build(np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([0.0, 2.0]))
    with pytest.raises(ValueError, match="range"):
        table.check_range(1.5)


def test_hermite_table_needs_increasing_knots():
    with pytest.raises(ValueError, match="increasing"):
        HermiteTable.build(np.array([0.0, 0.0]), np.zeros(2), np.zeros(2))


def test_linear_readout_ringing_matches_the_damped_oscillator(params):
    model = LinearReadout.from_params(params)
    omega0 = 2 * params.alpha * params.g2
    zeta = params.kappa_b / (8 * params.g2 * params.alpha)
    poles = np.linalg.eigvals(np.asarray(model.drift()))
    assert np.allclose(np.abs(poles), omega0, rtol=1e-12)
    assert np.allclose(-poles.real / np.abs(poles), zeta, rtol=1e-12)


def test_constant_drive_settles_to_the_adiabatic_buffer_amplitude(params):
    model = LinearReadout.from_params(params)
    eps = 0.2 * params.g2
    beta = model.buffer_amplitude(jnp.full(40, eps), duration=5.0, substeps=10)
    assert float(beta[-1]) == pytest.approx(-eps / (2 * params.alpha * params.g2), rel=1e-6)


def test_snr_of_a_long_constant_drive_approaches_the_steady_state_rate(params):
    model = LinearReadout.from_params(params)
    eps, duration = 0.2 * params.g2, 40.0
    snr2 = model.snr_squared(jnp.full(200, eps), duration, substeps=8, efficiency=1.0)
    beta_ss = eps / (2 * params.alpha * params.g2)
    steady = 16 * params.kappa_b * beta_ss**2 * duration
    # The ~50% overshoot of the underdamped response (zeta ~ 0.2) makes up for the rise.
    assert float(snr2) == pytest.approx(steady, rel=0.02)


def test_gaussian_error_matches_the_erfc_form():
    assert float(gaussian_error(jnp.asarray(0.0))) == pytest.approx(0.5)
    snr = 2 * np.sqrt(2) * 1.5
    assert float(gaussian_error(jnp.asarray(snr**2))) == pytest.approx(
        0.5 * 0.033894853524689, rel=1e-9
    )


def test_deadbeat_two_step_reaches_steady_state_at_the_switch(params):
    model = LinearReadout.from_params(params)
    eps = 0.2 * params.g2
    shape = deadbeat_two_step(model, eps)
    state = model.propagate_constant(np.zeros(2), shape.first_amplitude, shape.switch_time)
    assert np.allclose(state, model.steady_state(eps), rtol=1e-8, atol=1e-12)
    # From rest, e^{At} u is parallel to u only at half the damped period: the overshoot
    # peak of a reduced first step lands exactly on the target (zero-vibration shaper).
    period = 2 * np.pi / (2 * params.alpha * params.g2)
    assert 0.5 * period < shape.switch_time < 0.6 * period
    assert shape.first_amplitude < eps


def _problem(params, **kw):
    eps_knots = np.linspace(0.0, 1.0, 11) * params.g2
    rates = 1e-6 + 1e-4 * eps_knots**2
    table = HermiteTable.build(eps_knots, rates, 2e-4 * eps_knots)
    return ReadoutProblem(
        model=LinearReadout.from_params(params),
        rate_table=table,
        duration=1.0,
        pieces=12,
        substeps=6,
        **kw,
    )


def test_problem_gradients_match_finite_differences(params):
    prob = _problem(params, chi_ab=1.0, stark_coefficient=1e-3, compensate=False)
    x = np.linspace(0.1, 0.4, prob.pieces) * params.g2
    for fn in (prob.objective, prob.bit_flip_probability):
        val, grad = fn(x)
        h = 1e-6
        k = 3
        xp, xm = x.copy(), x.copy()
        xp[k] += h
        xm[k] -= h
        fd = (fn(xp)[0] - fn(xm)[0]) / (2 * h)
        assert grad[k] == pytest.approx(fd, rel=1e-5)
        assert np.isfinite(val)


def test_stark_compensation_removes_the_quartic_cost(params):
    x = np.full(12, 0.6 * params.g2)
    plain = _problem(params, chi_ab=1.0, stark_coefficient=0.0).bit_flip_probability(x)[0]
    stark = _problem(params, chi_ab=1.0, stark_coefficient=1e-2).bit_flip_probability(x)[0]
    comp = _problem(params, chi_ab=1.0, stark_coefficient=1e-2, compensate=True)
    assert stark > plain
    assert comp.bit_flip_probability(x)[0] == pytest.approx(plain, rel=1e-12)


def test_pareto_points_improve_with_budget(params):
    prob = _problem(params)
    low, high = (prob.solve(budget) for budget in (2e-5, 8e-5))
    assert high.snr_squared > low.snr_squared
    assert low.bit_flip_probability == pytest.approx(2e-5, rel=1e-3)


def test_diagonal_flip_form_reproduces_the_quasi_static_cost(params):
    base = _problem(params)
    c2 = 1e-4
    h = base.duration / base.pieces
    kernel_form = c2 * h * np.eye(base.pieces)
    second = _problem(params, flip_form=kernel_form, c2=c2)
    x = np.linspace(0.05, 0.9, base.pieces) * params.g2
    assert second.bit_flip_probability(x)[0] == pytest.approx(
        base.bit_flip_probability(x)[0], rel=1e-10
    )


def test_flip_form_must_match_the_pieces(params):
    with pytest.raises(ValueError, match="flip_form"):
        _problem(params, flip_form=np.eye(3), c2=1e-4)


def test_linear_order_optimum_is_the_top_generalized_eigenvector(params):
    prob = _problem(params)
    h = prob.duration / prob.pieces
    flip = 1e-4 * h * np.eye(prob.pieces)
    gain, shape = prob.linear_order_optimum(flip)
    x = shape * 0.3 * params.g2 / np.max(np.abs(shape))
    snr2 = prob.snr_squared(x)
    assert snr2 / float(x @ flip @ x) == pytest.approx(gain, rel=1e-6)
    assert gain >= prob.snr_squared(np.ones(prob.pieces)) / float(
        np.ones(prob.pieces) @ flip @ np.ones(prob.pieces)
    )


def test_positive_part_clips_only_negative_noise():
    from catqubit.optimize import positive_part

    rng = np.random.default_rng(3)
    q, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    form = (q * np.array([-1e-15, 1e-3, 0.2, 1.0, 2.0])) @ q.T
    clipped = positive_part(form)
    assert np.linalg.eigvalsh(clipped).min() >= 0
    assert np.allclose(clipped, form, atol=1e-11)


def test_signal_form_replaces_the_oscillator_model(params):
    base = _problem(params)
    x = np.linspace(-0.3, 0.5, base.pieces) * params.g2
    form = 2.0 * np.eye(base.pieces)
    prob = _problem(params, signal_form=form)
    assert prob.snr_squared(x) == pytest.approx(2.0 * float(x @ x), rel=1e-12)
    assert np.allclose(prob.objective(x)[1], -4.0 * x)


def test_signal_form_must_match_the_pieces(params):
    with pytest.raises(ValueError, match="signal_form"):
        _problem(params, signal_form=np.eye(2))
