"""Task 1.3 integration tests: what inflation prepares, and how to prepare |+alpha>."""

import dynamiqs as dq
import jax.numpy as jnp
import numpy as np
import pytest

from catqubit.convergence import fit_decay_rate
from catqubit.observables import cat_coherence, fidelity_with, memory_state, purity
from catqubit.params import CatParams
from catqubit.simulate import inflate
from catqubit.states import cat_state
from catqubit.truncation import top_level_projector

NA, NB = 24, 6


@pytest.fixture(scope="module")
def tsave():
    return jnp.linspace(0.0, 6.0, 61)


@pytest.fixture(scope="module")
def from_vacuum(tsave):
    # Module-scoped so the 6 us two-mode run happens once for all 1.3(a) checks.
    return inflate(NA, NB, CatParams.reglade(), tsave)


def test_inflate_starts_from_vacuum_and_tracks_both_photon_numbers(from_vacuum, tsave):
    n_a, n_b = np.real(np.asarray(from_vacuum.expects))
    assert n_a.shape == n_b.shape == tsave.shape
    assert n_a[0] == pytest.approx(0.0, abs=1e-12)
    assert n_b[0] == pytest.approx(0.0, abs=1e-12)


def test_inflate_rejects_memory_state_of_wrong_dimension(params, tsave):
    with pytest.raises(ValueError, match="dimension"):
        inflate(NA, NB, params, tsave, memory0=dq.coherent(NA + 1, 1.0))


def test_task_1_3a_vacuum_inflates_into_even_cat_not_a_codeword(from_vacuum, params):
    rho_a = memory_state(from_vacuum.states, NA, NB)
    t1 = 10  # t = 1 us, well inside the phase-flip time
    assert float(fidelity_with(rho_a[t1], cat_state(NA, params.alpha, +1))) > 0.8
    assert float(fidelity_with(rho_a[t1], dq.coherent(NA, params.alpha))) == pytest.approx(
        0.5, abs=0.01
    )


def test_task_1_3a_superposition_decays_at_phase_flip_rate(from_vacuum, params, tsave):
    rho_a = memory_state(from_vacuum.states, NA, NB)
    coherence = np.asarray(cat_coherence(rho_a, params.alpha))
    late = np.asarray(tsave) >= 2.0  # after the inflation transient
    rate = fit_decay_rate(np.asarray(tsave)[late], coherence[late])
    assert rate == pytest.approx(2 * params.kappa_a * params.alpha**2, rel=0.05)


def test_task_1_3a_long_time_state_is_a_mixture(from_vacuum):
    rho_a = memory_state(from_vacuum.states, NA, NB)
    assert float(purity(rho_a[-1])) == pytest.approx(0.5, abs=0.01)


def test_task_1_3b_displaced_start_prepares_plus_alpha(params, tsave):
    res = inflate(NA, NB, params, tsave, memory0=dq.coherent(NA, 1.5))
    rho_a = memory_state(res.states, NA, NB)
    assert float(fidelity_with(rho_a[-1], dq.coherent(NA, params.alpha))) > 0.99


def test_inflate_appends_extra_expectation_operators(params):
    a, _ = dq.destroy(NA, NB)
    res = inflate(NA, NB, params, jnp.linspace(0.0, 0.5, 6), extra_exp_ops=[a])
    assert res.expects.shape == (3, 6)


def test_inflate_can_skip_intermediate_states(params):
    res = inflate(NA, NB, params, jnp.linspace(0.0, 0.5, 6), save_states=False)
    assert res.states.shape[-3:] == (1, NA * NB, NA * NB)


def test_inflate_does_not_leak_population_into_top_levels_through_solver_error(params):
    # Regression: at the default ODE tolerance (1e-6) this run put 2.1e-4 into the top
    # two memory levels, against a physical 6.4e-8, and biased the Task 1.2 reference.
    na, nb = 32, 4
    monitor = top_level_projector((na, nb), 0, 2)
    res = inflate(
        na, nb, params, jnp.linspace(0.0, 1.0, 101), extra_exp_ops=[monitor], save_states=False
    )
    assert np.real(np.asarray(res.expects[2])).max() < 1e-6


def test_inflate_solver_tolerance_is_configurable(params):
    tsave = jnp.linspace(0.0, 0.2, 3)
    loose = inflate(NA, NB, params, tsave, save_states=False, solver_tol=1e-4)
    tight = inflate(NA, NB, params, tsave, save_states=False)
    assert not np.allclose(np.asarray(loose.expects), np.asarray(tight.expects), rtol=0, atol=1e-10)


def test_inflate_adds_extra_hamiltonian_term(params):
    a, _ = dq.destroy(NA, NB)
    tsave = jnp.linspace(0.0, 0.3, 4)
    zero = inflate(NA, NB, params, tsave, extra_H=0.0 * (a + a.dag()), save_states=False)
    base = inflate(NA, NB, params, tsave, save_states=False)
    np.testing.assert_allclose(np.asarray(zero.expects), np.asarray(base.expects), atol=1e-10)
