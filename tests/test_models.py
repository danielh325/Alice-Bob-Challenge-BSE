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
from catqubit.params import CatParams


def _dense(x):
    return np.asarray(x.to_jax())


def test_single_photon_loss_model_operators():
    model = single_photon_loss_model(10, delta=2.0, kappa=0.5)
    np.testing.assert_allclose(_dense(model.H), 2.0 * _dense(dq.number(10)))
    assert len(model.jump_ops) == 1
    np.testing.assert_allclose(_dense(model.jump_ops[0]), math.sqrt(0.5) * _dense(dq.destroy(10)))


def test_two_photon_loss_model_has_zero_hamiltonian_and_a_squared_jump():
    model = two_photon_loss_model(10, kappa2=4.0)
    np.testing.assert_allclose(_dense(model.H), 0.0)
    a = _dense(dq.destroy(10))
    np.testing.assert_allclose(_dense(model.jump_ops[0]), 2.0 * a @ a)


@pytest.mark.parametrize("sign", [+1, -1])
def test_coherent_states_at_plus_minus_alpha_are_dark_states_of_effective_dissipator(params, sign):
    def residual(dim):
        ket = dq.coherent(dim, sign * params.alpha)
        return np.linalg.norm(_dense(effective_model(dim, params).jump_ops[0]) @ _dense(ket))

    vacuum_residual = math.sqrt(params.kappa2) * params.alpha**2
    # Nonzero only through the cut-off Fock tail, so it must vanish as dim grows.
    assert residual(30) < 1e-6 * vacuum_residual
    assert residual(40) < 1e-3 * residual(30)


def test_vacuum_is_not_dark_state_of_effective_dissipator(params):
    model = effective_model(30, params)
    residual = np.linalg.norm(_dense(model.jump_ops[0]) @ _dense(dq.fock(30, 0)))
    assert residual == pytest.approx(math.sqrt(params.kappa2) * params.alpha**2, rel=1e-10)


def test_effective_model_includes_memory_loss(params):
    model = effective_model(30, params)
    assert len(model.jump_ops) == 2
    np.testing.assert_allclose(
        _dense(model.jump_ops[1]), math.sqrt(params.kappa_a) * _dense(dq.destroy(30))
    )


def test_effective_model_drops_memory_loss_when_kappa_a_is_zero(params):
    lossless = CatParams(g2=params.g2, kappa_b=params.kappa_b, kappa_a=0.0, alpha=params.alpha)
    assert len(effective_model(30, lossless).jump_ops) == 1


def test_effective_model_kappa2_override_scales_dissipator(params):
    nominal = _dense(effective_model(30, params).jump_ops[0])
    quadrupled = _dense(effective_model(30, params, kappa2=4 * params.kappa2).jump_ops[0])
    np.testing.assert_allclose(quadrupled, 2.0 * nominal, atol=1e-12)


def test_effective_model_accepts_batch_of_kappa2(params):
    kappa2 = jnp.array([0.25, 1.0, 4.0]) * params.kappa2
    model = effective_model(30, params, kappa2=kappa2)
    assert model.jump_ops[0].shape == (3, 30, 30)


def test_two_mode_hamiltonian_is_hermitian_with_product_dimension(params):
    model = two_mode_model(8, 4, params)
    H = _dense(model.H)
    assert H.shape == (32, 32)
    np.testing.assert_allclose(H, H.conj().T, atol=1e-12)


def test_two_mode_drive_couples_vacuum_to_one_buffer_photon_with_minus_eps_d(params):
    model = two_mode_model(8, 4, params)
    vac = _dense(dq.tensor(dq.fock(8, 0), dq.fock(4, 0)))
    one_b = _dense(dq.tensor(dq.fock(8, 0), dq.fock(4, 1)))
    element = (one_b.conj().T @ _dense(model.H) @ vac).item()
    assert element == pytest.approx(-params.eps_d)


def test_two_mode_exchange_converts_two_memory_photons_into_one_buffer_photon(params):
    model = two_mode_model(8, 4, params)
    two_a = _dense(dq.tensor(dq.fock(8, 2), dq.fock(4, 0)))
    one_b = _dense(dq.tensor(dq.fock(8, 0), dq.fock(4, 1)))
    element = (one_b.conj().T @ _dense(model.H) @ two_a).item()
    assert element == pytest.approx(params.g2 * math.sqrt(2))


def test_two_mode_jump_ops_are_buffer_then_memory_loss(params):
    model = two_mode_model(8, 4, params)
    a, b = dq.destroy(8, 4)
    assert len(model.jump_ops) == 2
    np.testing.assert_allclose(_dense(model.jump_ops[0]), math.sqrt(params.kappa_b) * _dense(b))
    np.testing.assert_allclose(_dense(model.jump_ops[1]), math.sqrt(params.kappa_a) * _dense(a))


@pytest.mark.parametrize(
    "build",
    [
        lambda p: single_photon_loss_model(1, delta=1.0, kappa=1.0),
        # a^2 vanishes identically below three Fock states
        lambda p: two_photon_loss_model(2, kappa2=1.0),
        lambda p: effective_model(2, p),
        lambda p: two_mode_model(2, 4, p),
        lambda p: two_mode_model(8, 1, p),
    ],
)
def test_too_small_truncation_rejected(params, build):
    with pytest.raises(ValueError, match="truncation"):
        build(params)


def test_negative_rate_rejected():
    with pytest.raises(ValueError, match="kappa"):
        single_photon_loss_model(10, delta=1.0, kappa=-1.0)
