"""Integration tests: simulations against the analytic predictions in the challenge."""

import math

import dynamiqs as dq
import jax.numpy as jnp
import numpy as np
import pytest

from catqubit.models import (
    effective_model,
    single_photon_loss_model,
    two_mode_model,
    two_photon_loss_model,
)
from catqubit.params import CatParams, to_angular
from catqubit.truncation import top_level_population

N = 30


def _photon_number(model, psi0, tsave, dim):
    res = dq.mesolve(
        model.H, model.jump_ops, psi0, tsave, exp_ops=[dq.number(dim)], progress_meter=False
    )
    return np.real(np.asarray(res.expects))[..., 0, :]


def test_task_1_0a_single_photon_loss_matches_exponential_decay():
    kappa = to_angular(1.0)
    model = single_photon_loss_model(N, delta=to_angular(1.0), kappa=kappa)
    tsave = jnp.linspace(0.0, 1.0, 51)
    n = _photon_number(model, dq.coherent(N, 2.0), tsave, N)
    np.testing.assert_allclose(n, 4.0 * np.exp(-kappa * np.asarray(tsave)), rtol=1e-4)


def test_task_1_0b_two_photon_loss_stalls_at_initial_odd_population(params):
    model = two_photon_loss_model(N, kappa2=params.kappa2)
    tsave = jnp.linspace(0.0, 5.0, 101)
    n = _photon_number(model, dq.coherent(N, 2.0), tsave, N)
    odd_population = (1 - math.exp(-2 * 2.0**2)) / 2
    assert n[-1] == pytest.approx(odd_population, abs=1e-4)


def test_task_1_0c_stabilization_from_vacuum_settles_at_alpha_squared(params):
    model = effective_model(N, params)
    tsave = jnp.linspace(0.0, 2.0, 201)
    n = _photon_number(model, dq.fock(N, 0), tsave, N)
    assert n[-1] == pytest.approx(params.alpha**2, rel=0.01)
    assert np.all(np.diff(n) > -1e-6), "photon number should rise monotonically"


def test_task_1_0c_lossless_stabilization_reaches_even_cat_photon_number(params):
    lossless = CatParams(g2=params.g2, kappa_b=params.kappa_b, kappa_a=0.0, alpha=params.alpha)
    tsave = jnp.linspace(0.0, 2.0, 101)
    n = _photon_number(effective_model(N, lossless), dq.fock(N, 0), tsave, N)
    assert n[-1] == pytest.approx(params.alpha**2 * math.tanh(params.alpha**2), rel=1e-4)


def test_task_1_0c_kappa2_sweep_changes_speed_but_not_steady_state(params):
    kappa2 = jnp.array([0.25, 1.0, 4.0]) * params.kappa2
    tsave = jnp.linspace(0.0, 2.0, 401)
    n = _photon_number(effective_model(N, params, kappa2=kappa2), dq.fock(N, 0), tsave, N)
    assert n.shape == (3, 401)
    np.testing.assert_allclose(n[:, -1], params.alpha**2, rtol=0.01)
    t90 = [float(tsave[np.argmax(curve >= 0.9 * params.alpha**2)]) for curve in n]
    assert t90[0] > t90[1] > t90[2]


@pytest.mark.slow
def test_task_1_1_two_mode_inflation_overshoots_then_settles_near_alpha_squared(params):
    na, nb = 24, 6
    model = two_mode_model(na, nb, params)
    a, b = dq.destroy(na, nb)
    psi0 = dq.tensor(dq.fock(na, 0), dq.fock(nb, 0))
    tsave = jnp.linspace(0.0, 2.0, 201)
    res = dq.mesolve(
        model.H,
        model.jump_ops,
        psi0,
        tsave,
        exp_ops=[a.dag() @ a, b.dag() @ b],
        progress_meter=False,
    )
    n_a, n_b = np.real(np.asarray(res.expects))
    assert n_a.max() > 1.1 * params.alpha**2, "memory should overshoot alpha^2"
    assert n_a[-1] == pytest.approx(params.alpha**2, rel=0.03)
    assert n_b.max() > 1.0 and n_b[-1] < 0.1 * n_b.max()
    memory_top = top_level_population(res.states, dims=(na, nb), mode=0, n_levels=2)
    assert np.asarray(memory_top).shape == (201,)
