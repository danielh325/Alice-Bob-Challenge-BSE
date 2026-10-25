import math

import dynamiqs as dq
import numpy as np
import pytest

from catqubit.states import cat_state

N = 30
ALPHA = 2.1


def _fock_populations(ket):
    return np.abs(np.asarray(ket.to_jax()).ravel()) ** 2


@pytest.mark.parametrize("parity", [+1, -1])
def test_cat_state_is_normalized(parity):
    assert float(cat_state(N, ALPHA, parity).norm()) == pytest.approx(1.0, abs=1e-10)


@pytest.mark.parametrize("parity", [+1, -1])
def test_cat_state_has_definite_parity(parity):
    p = dq.expect(dq.parity(N), cat_state(N, ALPHA, parity))
    assert float(np.real(p)) == pytest.approx(parity, abs=1e-10)


def test_even_cat_has_no_odd_fock_population():
    pops = _fock_populations(cat_state(N, ALPHA, +1))
    assert np.max(pops[1::2]) < 1e-20


def test_odd_cat_has_no_even_fock_population():
    pops = _fock_populations(cat_state(N, ALPHA, -1))
    assert np.max(pops[0::2]) < 1e-20


def test_even_cat_photon_number_is_alpha_squared_tanh():
    n = dq.expect(dq.number(N), cat_state(N, ALPHA, +1))
    assert float(np.real(n)) == pytest.approx(ALPHA**2 * math.tanh(ALPHA**2), rel=1e-6)


def test_odd_cat_photon_number_is_alpha_squared_coth():
    n = dq.expect(dq.number(N), cat_state(N, ALPHA, -1))
    assert float(np.real(n)) == pytest.approx(ALPHA**2 / math.tanh(ALPHA**2), rel=1e-6)


def test_small_odd_cat_is_close_to_single_photon():
    ket = cat_state(N, 1e-3, -1)
    assert _fock_populations(ket)[1] == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("parity", [0, 2, "+"])
def test_invalid_parity_rejected(parity):
    with pytest.raises(ValueError, match="parity"):
        cat_state(N, ALPHA, parity)


def test_too_small_truncation_rejected():
    with pytest.raises(ValueError, match="dim"):
        cat_state(1, ALPHA, +1)
