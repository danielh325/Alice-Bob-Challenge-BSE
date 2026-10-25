import dynamiqs as dq
import numpy as np
import pytest
import scipy.sparse as sp

from catqubit.liouvillian import (
    FockTruncation,
    lindblad_superoperator,
    parity_block,
    two_mode_hamiltonian,
    two_mode_liouvillian,
    two_mode_operators,
    vec,
)


def _random_density_matrix(dim, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    rho = m @ m.conj().T
    return rho / np.trace(rho)


def test_rectangular_truncation_operators_match_dynamiqs():
    a, b = two_mode_operators(FockTruncation(na=5, nb=3))
    a_ref, b_ref = dq.destroy(5, 3)
    np.testing.assert_allclose(a.toarray(), np.asarray(a_ref.to_jax()))
    np.testing.assert_allclose(b.toarray(), np.asarray(b_ref.to_jax()))


def test_rectangular_truncation_keeps_every_state():
    trunc = FockTruncation(na=5, nb=3)
    assert trunc.dim == 15
    np.testing.assert_array_equal(trunc.memory_photons, np.repeat(np.arange(5), 3))


def test_tapered_truncation_widens_buffer_only_near_empty_memory():
    trunc = FockTruncation(na=10, nb=3, nb_extra=4, na_taper=5)
    buffer_levels = np.bincount(trunc.memory_photons, minlength=10)
    assert buffer_levels[0] == 7
    assert np.all(np.diff(buffer_levels) <= 0)
    assert np.all(buffer_levels[5:] == 3)
    assert trunc.dim == buffer_levels.sum()


def test_tapered_operators_are_projections_of_the_full_ones():
    trunc = FockTruncation(na=6, nb=2, nb_extra=3, na_taper=3)
    a, b = two_mode_operators(trunc)
    a_full, b_full = two_mode_operators(FockTruncation(na=6, nb=5))
    keep = trunc.kept_indices(full_nb=5)
    np.testing.assert_allclose(a.toarray(), a_full.toarray()[np.ix_(keep, keep)])
    np.testing.assert_allclose(b.toarray(), b_full.toarray()[np.ix_(keep, keep)])


def test_tapered_hamiltonian_is_projection_of_full_hamiltonian(params):
    # Products such as a^dag^2 b pass through intermediate states that a tapered buffer
    # may drop, so H must be built on the full grid and projected, not built from
    # projected a and b.
    trunc = FockTruncation(na=8, nb=2, nb_extra=4, na_taper=5)
    full = FockTruncation(na=8, nb=trunc.nb_max)
    keep = trunc.kept_indices()
    h_full = two_mode_hamiltonian(params, full).toarray()
    np.testing.assert_allclose(
        two_mode_hamiltonian(params, trunc).toarray(), h_full[np.ix_(keep, keep)]
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"na": 2, "nb": 3}, "na"),
        ({"na": 5, "nb": 0}, "nb"),
        ({"na": 5, "nb": 3, "nb_extra": -1}, "nb_extra"),
        ({"na": 5, "nb": 3, "nb_extra": 2, "na_taper": 0}, "na_taper"),
    ],
)
def test_truncation_rejects_bad_sizes(kwargs, match):
    with pytest.raises(ValueError, match=match):
        FockTruncation(**kwargs)


def test_superoperator_matches_dynamiqs_lindbladian():
    rng = np.random.default_rng(1)
    dim = 6
    h = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    h = h + h.conj().T
    jumps = [rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim)) for _ in range(2)]
    rho = _random_density_matrix(dim)
    big_l = lindblad_superoperator(sp.csr_matrix(h), [sp.csr_matrix(j) for j in jumps])
    expected = np.asarray(dq.lindbladian(h, jumps, rho).to_jax())
    np.testing.assert_allclose(big_l @ vec(rho), vec(expected), atol=1e-10)


def test_vec_is_column_stacking():
    m = np.arange(6).reshape(2, 3)
    np.testing.assert_array_equal(vec(m), [0, 3, 1, 4, 2, 5])


def test_two_mode_liouvillian_preserves_trace(params):
    trunc = FockTruncation(na=8, nb=4)
    big_l = two_mode_liouvillian(params, trunc)
    trace_row = vec(np.eye(trunc.dim))
    assert np.max(np.abs(trace_row @ big_l)) < 1e-10


def test_two_mode_liouvillian_matches_dynamiqs_model(params):
    from catqubit.models import two_mode_model

    trunc = FockTruncation(na=6, nb=3)
    model = two_mode_model(6, 3, params)
    rho = _random_density_matrix(trunc.dim, seed=2)
    expected = np.asarray(dq.lindbladian(model.H, model.jump_ops, rho).to_jax())
    big_l = two_mode_liouvillian(params, trunc)
    np.testing.assert_allclose(big_l @ vec(rho), vec(expected), atol=1e-9)


def test_memory_parity_blocks_decouple(params):
    trunc = FockTruncation(na=8, nb=4, nb_extra=2, na_taper=4)
    big_l = two_mode_liouvillian(params, trunc).tocsc()
    odd, odd_idx = parity_block(big_l, trunc, odd=True)
    _, even_idx = parity_block(big_l, trunc, odd=False)
    assert len(odd_idx) + len(even_idx) == trunc.dim**2
    assert abs(big_l[odd_idx, :][:, even_idx]).sum() == 0
    assert abs(big_l[even_idx, :][:, odd_idx]).sum() == 0
    assert odd.shape == (len(odd_idx), len(odd_idx))


def _codeword_vec(alpha, trunc):
    memory = np.asarray(dq.coherent(trunc.na, alpha).to_jax()).ravel()
    buffer = np.zeros(trunc.nb_max)
    buffer[0] = 1.0
    psi = np.kron(memory, buffer)[trunc.kept_indices()]
    return vec(np.outer(psi, psi.conj()))


def test_codeword_is_steady_state_only_without_memory_loss(params):
    from catqubit.params import CatParams

    trunc = FockTruncation(na=30, nb=4)
    p = params.with_alpha(1.0)
    lossless = CatParams(g2=p.g2, kappa_b=p.kappa_b, kappa_a=0.0, alpha=p.alpha)
    rho = _codeword_vec(p.alpha, trunc)
    assert np.linalg.norm(two_mode_liouvillian(lossless, trunc) @ rho) < 1e-10
    assert np.linalg.norm(two_mode_liouvillian(p, trunc) @ rho) > 1e-3


def test_readout_drive_adds_eps_z_times_a_plus_a_dag(params):
    trunc = FockTruncation(na=6, nb=3)
    a, _ = two_mode_operators(trunc)
    driven = two_mode_hamiltonian(params, trunc, eps_z=0.7).toarray()
    bare = two_mode_hamiltonian(params, trunc).toarray()
    np.testing.assert_allclose(driven - bare, 0.7 * (a + a.T).toarray())


def test_thermal_liouvillian_matches_explicit_jump_operators(params):
    from catqubit.models import two_mode_model

    trunc = FockTruncation(na=6, nb=3)
    model = two_mode_model(6, 3, params)
    a, b = dq.destroy(6, 3)
    n_a, n_b = 0.1, 0.011
    jumps = [
        np.sqrt(params.kappa_b * (1 + n_b)) * b,
        np.sqrt(params.kappa_b * n_b) * b.dag(),
        np.sqrt(params.kappa_a * (1 + n_a)) * a,
        np.sqrt(params.kappa_a * n_a) * a.dag(),
    ]
    rho = _random_density_matrix(trunc.dim, seed=3)
    expected = np.asarray(dq.lindbladian(model.H, jumps, rho).to_jax())
    big_l = two_mode_liouvillian(params, trunc, n_th_a=n_a, n_th_b=n_b)
    np.testing.assert_allclose(big_l @ vec(rho), vec(expected), atol=1e-9)


def test_thermal_terms_keep_parity_sectors_decoupled(params):
    trunc = FockTruncation(na=8, nb=4)
    big_l = two_mode_liouvillian(params, trunc, n_th_a=0.1, n_th_b=0.01).tocsc()
    _, odd_idx = parity_block(big_l, trunc, odd=True)
    _, even_idx = parity_block(big_l, trunc, odd=False)
    assert abs(big_l[odd_idx, :][:, even_idx]).sum() == 0


@pytest.mark.parametrize("kwargs", [{"n_th_a": -0.1}, {"n_th_b": -0.1}])
def test_negative_thermal_population_rejected(params, kwargs):
    with pytest.raises(ValueError, match="n_th"):
        two_mode_liouvillian(params, FockTruncation(na=6, nb=3), **kwargs)
