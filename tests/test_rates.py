"""Bit- and phase-flip rates from the two-mode Liouvillian (Tasks 2.1 and 2.2)."""

import dynamiqs as dq
import jax.numpy as jnp
import numpy as np
import pytest

from catqubit.convergence import fit_decay_rate
from catqubit.liouvillian import FockTruncation, parity_block, two_mode_liouvillian
from catqubit.rates import bit_flip_rate, phase_flip_rate, slowest_eigenvalues
from catqubit.simulate import inflate
from catqubit.states import cat_state


def test_ratio_method_matches_slowest_odd_eigenvalue(params):
    p = params.with_alpha(1.2)
    trunc = FockTruncation(na=18, nb=5)
    odd, _ = parity_block(two_mode_liouvillian(p, trunc), trunc, odd=True)
    eig = slowest_eigenvalues(odd, k=2)
    assert bit_flip_rate(p, trunc) == pytest.approx(-eig[0].real, rel=1e-6)


def test_bit_flip_mode_is_well_separated_from_the_next_odd_mode(params):
    p = params.with_alpha(1.2)
    trunc = FockTruncation(na=18, nb=5)
    odd, _ = parity_block(two_mode_liouvillian(p, trunc), trunc, odd=True)
    rates = -slowest_eigenvalues(odd, k=2).real
    assert rates[1] > 100 * rates[0]


def test_bit_flip_rate_falls_exponentially_with_cat_size(params):
    rates = [bit_flip_rate(params.with_alpha(a), FockTruncation(na=16, nb=5)) for a in (0.8, 1.0)]
    assert rates[1] < 0.5 * rates[0]


def test_tapered_truncation_matches_wide_rectangular_buffer(params):
    p = params.with_alpha(1.6)
    wide = bit_flip_rate(p, FockTruncation(na=23, nb=9))
    tapered = bit_flip_rate(p, FockTruncation(na=23, nb=5, nb_extra=4, na_taper=10))
    assert tapered == pytest.approx(wide, rel=1e-3)


def test_bit_flip_rate_refuses_an_unconverged_estimate(params, monkeypatch):
    import catqubit.rates as rates

    monkeypatch.setattr(rates, "MAX_INVERSE_ITERATIONS", 1)
    with pytest.raises(ValueError, match="did not converge"):
        bit_flip_rate(params.with_alpha(1.0), FockTruncation(na=14, nb=4))


def test_phase_flip_rate_is_close_to_two_kappa_a_alpha_squared(params):
    p = params.with_alpha(1.6)
    rate = phase_flip_rate(p, FockTruncation(na=23, nb=6))
    assert rate == pytest.approx(2 * p.kappa_a * p.alpha**2, rel=0.05)


@pytest.mark.slow
def test_bit_flip_rate_matches_time_domain_decay_of_re_a(params):
    p = params.with_alpha(0.8)
    na, nb = 14, 4
    trunc = FockTruncation(na=na, nb=nb)
    gamma = bit_flip_rate(p, trunc)
    a, _ = dq.destroy(na, nb)
    tsave = jnp.linspace(0.0, 4.0 / gamma, 81)
    res = inflate(
        na, nb, p, tsave, memory0=dq.coherent(na, p.alpha), extra_exp_ops=[a], save_states=False
    )
    re_a = np.real(np.asarray(res.expects[2]))
    t = np.asarray(tsave)
    late = t > 0.5 / gamma
    assert fit_decay_rate(t[late], re_a[late]) == pytest.approx(gamma, rel=1e-3)


@pytest.mark.slow
def test_phase_flip_rate_matches_time_domain_parity_decay(params):
    p = params.with_alpha(1.4)
    na, nb = 20, 5
    gamma = phase_flip_rate(p, FockTruncation(na=na, nb=nb))
    tsave = jnp.linspace(0.0, 3.0 / gamma, 61)
    res = inflate(
        na,
        nb,
        p,
        tsave,
        memory0=cat_state(na, p.alpha, +1),
        extra_exp_ops=[dq.tensor(dq.parity(na), dq.eye(nb))],
        save_states=False,
    )
    parity = np.real(np.asarray(res.expects[2]))
    t = np.asarray(tsave)
    late = t > 1.0
    p_inf = 1 / np.cosh(2 * p.alpha**2)
    assert fit_decay_rate(t[late], parity[late] - p_inf) == pytest.approx(gamma, rel=1e-2)


def test_thermal_photons_raise_both_rates(params):
    p = params.with_alpha(1.2)
    trunc = FockTruncation(na=18, nb=5)
    assert bit_flip_rate(p, trunc, n_th_a=0.1) > 1.5 * bit_flip_rate(p, trunc)
    # Each lost photon (rate kappa_a (1 + n) <a^dag a>) and each gained one (kappa_a n <a a^dag>)
    # flips parity: Gamma_Z = 2 kappa_a [alpha^2 (1 + 2n) + n]. The large-alpha form
    # kappa_1 -> kappa_1 (1 + 2n) (Chamberland et al. 2022, App. D) drops the "+ n".
    n = 0.1
    expected = (p.alpha**2 * (1 + 2 * n) + n) / p.alpha**2
    ratio = phase_flip_rate(p, trunc, n_th_a=n) / phase_flip_rate(p, trunc)
    assert ratio == pytest.approx(expected, rel=0.01)
