"""Task 1.3(b) methods: bias pulse during inflation, and the basin model."""

import math

import dynamiqs as dq
import jax.numpy as jnp
import numpy as np
import pytest

from catqubit.observables import fidelity_with, memory_state
from catqubit.preparation import (
    amplified_push,
    basin_fidelity,
    bias_pulse,
    ramp_push_strength,
    ramped_preparation,
)
from catqubit.simulate import inflate

NA, NB = 24, 8


def test_basin_fidelity_limits():
    assert basin_fidelity(0.0) == pytest.approx(0.5)
    assert basin_fidelity(3.0) == pytest.approx(1.0, abs=1e-6)


def test_basin_fidelity_matches_gaussian_tail():
    # x-quadrature of |beta> has variance 1/4, so P(x > 0) = 1 - erfc(sqrt(2) beta) / 2.
    assert basin_fidelity(0.5) == pytest.approx(1 - 0.5 * math.erfc(math.sqrt(2) * 0.5))


def test_basin_fidelity_uses_projection_of_phase_error():
    assert basin_fidelity(1.0, theta=math.pi / 2) == pytest.approx(0.5)
    assert basin_fidelity(1.0, theta=math.pi) == pytest.approx(1 - basin_fidelity(1.0))
    assert basin_fidelity(1.0, theta=math.pi / 3) == pytest.approx(basin_fidelity(0.5))


def test_basin_fidelity_rejects_negative_amplitude():
    with pytest.raises(ValueError, match="beta"):
        basin_fidelity(-0.1)


@pytest.mark.parametrize("theta", [0.0, math.pi / 2, math.pi])
def test_bias_pulse_alone_displaces_vacuum_by_its_area(theta):
    area, duration = 0.8, 0.05
    pulse = bias_pulse(NA, NB, area=area, duration=duration, theta=theta)
    psi0 = dq.tensor(dq.fock(NA, 0), dq.fock(NB, 0))
    res = dq.sesolve(pulse, psi0, jnp.array([0.0, 2 * duration]), progress_meter=False)
    rho_a = memory_state(res.states[-1], NA, NB)
    target = dq.coherent(NA, area * complex(math.cos(theta), math.sin(theta)))
    assert float(fidelity_with(rho_a, target)) == pytest.approx(1.0, abs=1e-6)


def test_bias_pulse_is_off_after_its_duration():
    pulse = bias_pulse(NA, NB, area=1.0, duration=0.05)
    assert np.allclose(np.asarray(pulse(0.06).to_jax()), 0.0)
    assert not np.allclose(np.asarray(pulse(0.02).to_jax()), 0.0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [({"area": -1.0, "duration": 0.05}, "area"), ({"area": 1.0, "duration": 0.0}, "duration")],
)
def test_bias_pulse_validates_arguments(kwargs, match):
    with pytest.raises(ValueError, match=match):
        bias_pulse(NA, NB, **kwargs)


def _final_plus_fidelity(params, **pulse):
    extra = bias_pulse(NA, NB, **pulse)
    res = inflate(NA, NB, params, jnp.linspace(0.0, 2.0, 3), extra_H=extra, save_states=False)
    rho_a = memory_state(res.final_state, NA, NB)
    return float(fidelity_with(rho_a, dq.coherent(NA, params.alpha)))


def test_task_1_3b_short_bias_pulse_during_inflation_prepares_plus_alpha(params):
    assert _final_plus_fidelity(params, area=2.5, duration=0.05) > 0.99


def test_task_1_3b_pulse_phase_selects_the_codeword(params):
    assert _final_plus_fidelity(params, area=2.5, duration=0.05, theta=math.pi) < 0.01


def test_task_1_3b_quadrature_pulse_does_not_break_the_symmetry(params):
    f = _final_plus_fidelity(params, area=2.5, duration=0.05, theta=math.pi / 2)
    assert f == pytest.approx(0.5, abs=0.01)


def test_task_1_3b_overdriven_pulse_spills_into_the_wrong_well(params):
    good = _final_plus_fidelity(params, area=2.5, duration=0.05)
    overdriven = _final_plus_fidelity(params, area=5.0, duration=0.1)
    assert overdriven < good - 0.2


def _square(strength, duration):
    return lambda s: strength if s < duration else 0.0


def test_amplified_push_of_a_short_pulse_is_its_area(params):
    # A pulse much shorter than 1 / lambda is not amplified yet: beta_eff = area.
    beta = amplified_push(_square(1.0 / 1e-3, 1e-3), lambda s: params.eps_d, 0.01, params)
    assert beta == pytest.approx(1.0, rel=1e-2)


def test_amplified_push_without_stabilization_is_the_bare_area(params):
    # eps_d = 0: only kappa_a / 2 decay, so a push at time s equals a start displaced by
    # eps exp(+kappa_a s / 2) (the earlier displacement would have decayed by then).
    beta = amplified_push(_square(2.0, 1.0), lambda s: 0.0, 1.0, params)
    expected = 2.0 * (math.exp(params.kappa_a / 2) - 1) / (params.kappa_a / 2)
    # rel 1e-3: the pulse switches off on the last grid point, a half-step trapezoid error.
    assert beta == pytest.approx(expected, rel=1e-3)


def test_amplified_push_discounts_late_pushes(params):
    def on(s):
        return params.eps_d

    early = amplified_push(_square(1.0, 0.1), on, 0.5, params)
    late = amplified_push(lambda s: 1.0 if 0.4 <= s < 0.5 else 0.0, on, 0.5, params)
    assert late < 0.2 * early


@pytest.mark.parametrize(("kwargs", "match"), [({"t_end": 0.0}, "t_end"), ({"steps": 1}, "steps")])
def test_amplified_push_validates_arguments(params, kwargs, match):
    args = {"push": _square(1.0, 0.1), "buffer_drive": lambda s: 0.0, "t_end": 1.0}
    with pytest.raises(ValueError, match=match):
        amplified_push(**{**args, **kwargs}, params=params)


def test_ramp_push_strength_hits_its_target(params):
    strength = ramp_push_strength(params, t_ramp=1.0, beta_target=1.5)
    ramp = lambda s: params.eps_d * min(s, 1.0)  # noqa: E731
    assert amplified_push(_square(strength, 1.0), ramp, 1.0, params) == pytest.approx(1.5, rel=1e-6)
    # Method C needs a push weaker than g2, unlike the ~10 g2 square pulse of method B.
    assert strength < params.g2


def test_ramped_preparation_validates_ramp(params):
    with pytest.raises(ValueError, match="t_ramp"):
        ramped_preparation(NA, NB, params, t_ramp=0.0, strength=1.0)


def test_ramped_preparation_cancels_the_buffer_drive_at_t0(params):
    _, b = dq.destroy(NA, NB)
    extra = ramped_preparation(NA, NB, params, t_ramp=1.0, strength=0.0)
    expected = params.eps_d * (b + b.dag()).to_jax()
    assert np.allclose(np.asarray(extra(0.0).to_jax()), np.asarray(expected))
    assert np.allclose(np.asarray(extra(1.5).to_jax()), 0.0)


@pytest.mark.slow
def test_task_1_3b_method_c_weak_push_during_ramp_prepares_plus_alpha(params):
    strength = ramp_push_strength(params, t_ramp=1.0, beta_target=1.5)
    extra = ramped_preparation(NA, NB, params, t_ramp=1.0, strength=strength)
    res = inflate(NA, NB, params, jnp.linspace(0.0, 2.5, 3), extra_H=extra, save_states=False)
    rho_a = memory_state(res.final_state, NA, NB)
    assert float(fidelity_with(rho_a, dq.coherent(NA, params.alpha))) > 0.998
