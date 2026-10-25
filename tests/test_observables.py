import math

import dynamiqs as dq
import numpy as np
import pytest

from catqubit.observables import (
    cat_coherence,
    fidelity_with,
    memory_state,
    parity_expectation,
    purity,
)
from catqubit.states import cat_state

N = 30
ALPHA = 2.1


def _mixture(*kets):
    return dq.stack([k.todm() for k in kets]).mean(axis=0)


def test_memory_state_traces_out_buffer():
    memory = dq.coherent(8, 1.0)
    joint = dq.tensor(memory, dq.fock(3, 1))
    np.testing.assert_allclose(
        np.asarray(memory_state(joint, 8, 3).to_jax()),
        np.asarray(memory.todm().to_jax()),
        atol=1e-12,
    )


def test_memory_state_keeps_time_axis():
    joint = dq.stack([dq.tensor(dq.fock(8, n), dq.fock(3, 0)).todm() for n in range(4)])
    assert memory_state(joint, 8, 3).shape == (4, 8, 8)


@pytest.mark.parametrize("parity", [+1, -1])
def test_parity_of_cat_states(parity):
    p = parity_expectation(cat_state(N, ALPHA, parity))
    assert float(p) == pytest.approx(parity, abs=1e-10)


def test_parity_of_coherent_state_is_exponentially_small():
    expected = math.exp(-2 * ALPHA**2)
    assert float(parity_expectation(dq.coherent(N, ALPHA))) == pytest.approx(expected, rel=1e-6)


def test_purity_distinguishes_pure_from_mixed():
    assert float(purity(dq.coherent(N, ALPHA))) == pytest.approx(1.0, abs=1e-10)
    assert float(purity(_mixture(dq.fock(N, 0), dq.fock(N, 1)))) == pytest.approx(0.5)


def test_fidelity_with_itself_and_orthogonal_state():
    ket = cat_state(N, ALPHA, +1)
    assert float(fidelity_with(ket.todm(), ket)) == pytest.approx(1.0, abs=1e-10)
    assert float(fidelity_with(ket, cat_state(N, ALPHA, -1))) == pytest.approx(0.0, abs=1e-10)


def test_fidelity_of_even_cat_with_one_codeword_is_one_half():
    f = fidelity_with(cat_state(N, ALPHA, +1), dq.coherent(N, ALPHA))
    assert float(f) == pytest.approx(0.5, abs=1e-3)


@pytest.mark.parametrize("parity", [+1, -1])
def test_cat_coherence_is_plus_minus_one_for_cat_states(parity):
    c = cat_coherence(cat_state(N, ALPHA, parity), ALPHA)
    assert float(c) == pytest.approx(parity, abs=1e-3)


def test_cat_coherence_vanishes_for_classical_mixture():
    mixture = _mixture(dq.coherent(N, ALPHA), dq.coherent(N, -ALPHA))
    assert abs(float(cat_coherence(mixture, ALPHA))) < 1e-3


def test_cat_coherence_vanishes_for_single_codeword():
    assert abs(float(cat_coherence(dq.coherent(N, ALPHA), ALPHA))) < 1e-3


def test_observables_are_real_and_batched():
    states = dq.stack([cat_state(N, ALPHA, +1).todm(), cat_state(N, ALPHA, -1).todm()])
    for value in (
        parity_expectation(states),
        purity(states),
        fidelity_with(states, dq.coherent(N, ALPHA)),
        cat_coherence(states, ALPHA),
    ):
        assert value.shape == (2,)
        assert not np.iscomplexobj(np.asarray(value))


def test_fidelity_dimension_mismatch_rejected():
    with pytest.raises(ValueError, match="dimension"):
        fidelity_with(dq.coherent(N, ALPHA), dq.coherent(N + 1, ALPHA))
