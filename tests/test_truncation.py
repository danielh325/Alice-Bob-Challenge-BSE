import dynamiqs as dq
import jax.numpy as jnp
import numpy as np
import pytest

from catqubit.truncation import top_level_population, top_level_projector


def test_single_mode_top_fock_state_has_full_top_population():
    pop = top_level_population(dq.fock(6, 5), dims=(6,), mode=0, n_levels=1)
    assert float(pop) == pytest.approx(1.0)


def test_single_mode_vacuum_has_no_top_population():
    pop = top_level_population(dq.fock(6, 0), dims=(6,), mode=0, n_levels=2)
    assert float(pop) == pytest.approx(0.0)


def test_two_mode_population_is_resolved_per_mode():
    state = dq.tensor(dq.fock(5, 4), dq.fock(3, 0)).todm()
    assert float(top_level_population(state, dims=(5, 3), mode=0, n_levels=1)) == pytest.approx(1.0)
    assert float(top_level_population(state, dims=(5, 3), mode=1, n_levels=1)) == pytest.approx(0.0)


def test_counts_only_requested_number_of_top_levels():
    ket = (dq.fock(6, 3) + dq.fock(6, 5)).unit()
    assert float(top_level_population(ket, dims=(6,), mode=0, n_levels=1)) == pytest.approx(0.5)
    assert float(top_level_population(ket, dims=(6,), mode=0, n_levels=3)) == pytest.approx(1.0)


def test_batched_states_keep_batch_shape():
    states = dq.stack([dq.fock(4, 0).todm(), dq.fock(4, 3).todm(), dq.fock(4, 2).todm()])
    pops = top_level_population(states, dims=(4,), mode=0, n_levels=1)
    np.testing.assert_allclose(np.asarray(pops), [0.0, 1.0, 0.0], atol=1e-12)


def test_returns_real_values():
    pop = top_level_population(dq.coherent(10, 1.5), dims=(10,), mode=0, n_levels=2)
    assert not jnp.iscomplexobj(pop)


@pytest.mark.parametrize("n_levels", [0, 7])
def test_invalid_number_of_levels_rejected(n_levels):
    with pytest.raises(ValueError, match="n_levels"):
        top_level_population(dq.fock(6, 0), dims=(6,), mode=0, n_levels=n_levels)


def test_invalid_mode_rejected():
    with pytest.raises(ValueError, match="mode"):
        top_level_population(dq.fock(6, 0), dims=(6,), mode=1, n_levels=1)


def test_projector_expectation_matches_top_level_population():
    dims = (6, 3)
    state = dq.tensor(dq.coherent(6, 1.4), (dq.fock(3, 1) + dq.fock(3, 2)).unit())
    for mode, n_levels in [(0, 2), (1, 1)]:
        projector = top_level_projector(dims, mode, n_levels)
        assert float(np.real(dq.expect(projector, state))) == pytest.approx(
            float(top_level_population(state, dims, mode, n_levels)), abs=1e-12
        )


def test_projector_acts_on_the_full_product_space():
    assert top_level_projector((6, 3), 1, 1).shape == (18, 18)


def test_projector_validates_arguments():
    with pytest.raises(ValueError, match="n_levels"):
        top_level_projector((6,), 0, 7)
    with pytest.raises(ValueError, match="mode"):
        top_level_projector((6,), 2, 1)
