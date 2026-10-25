"""Bit-flip rate with a readout drive eps_z != 0 (Task 2.4b)."""

import dynamiqs as dq
import jax.numpy as jnp
import numpy as np
import pytest

from catqubit.convergence import fit_decay_rate
from catqubit.driven_rates import (
    Dissipation,
    DriveKernel,
    HamiltonianTerms,
    ResponseKernel,
    buffer_jump,
    buffer_response,
    codeword_contrast,
    deflate_steady_state,
    drive_kernel,
    drive_rate_coefficient,
    drive_susceptibility,
    driven_bit_flip_rate,
    driven_liouvillian,
    hamiltonian_generators,
    pulse_bit_flip_probability,
)
from catqubit.liouvillian import (
    FockTruncation,
    two_mode_hamiltonian,
    two_mode_liouvillian,
    two_mode_operators,
)
from catqubit.rates import bit_flip_rate

SMALL = FockTruncation(na=18, nb=5)
SETTLE = 2.0  # us for the fast modes of the undriven model to decay


def _model(params):
    return HamiltonianTerms.from_params(params), Dissipation.from_params(params)


def _rate(params, trunc, eps_z=0.0, **extra):
    terms = HamiltonianTerms.from_params(params, eps_z=eps_z, **extra)
    return driven_bit_flip_rate(terms, Dissipation.from_params(params), trunc)


def test_plain_terms_rebuild_the_shared_hamiltonian(params):
    p = params.with_alpha(1.2)
    trunc = FockTruncation(na=12, nb=4, nb_extra=2, na_taper=4.0)
    gens = hamiltonian_generators(trunc)
    h = sum(c * gens[k] for k, c in HamiltonianTerms.from_params(p, eps_z=0.3).coefficients())
    ref = two_mode_hamiltonian(p, trunc, eps_z=0.3)
    assert abs(h - ref).max() < 1e-12


def test_plain_liouvillian_matches_the_shared_one(params):
    p = params.with_alpha(1.2)
    trunc = FockTruncation(na=10, nb=4)
    big_l = driven_liouvillian(
        HamiltonianTerms.from_params(p, eps_z=0.2), Dissipation.from_params(p), trunc
    )
    assert abs(big_l - two_mode_liouvillian(p, trunc, eps_z=0.2)).max() < 1e-12


def test_extra_generators_are_the_documented_diagonal_operators():
    trunc = FockTruncation(na=6, nb=4)
    a, b = two_mode_operators(trunc)
    n_a, n_b = (a.T @ a).toarray(), (b.T @ b).toarray()
    gens = {k: v.toarray() for k, v in hamiltonian_generators(trunc).items()}
    assert np.allclose(gens["delta_a"], n_a)
    assert np.allclose(gens["delta_b"], n_b)
    assert np.allclose(gens["chi_ab"], -n_a @ n_b)
    assert np.allclose(gens["kerr_a"], -0.5 * n_a @ (n_a - np.eye(trunc.dim)))
    assert np.allclose(gens["kerr_b"], -0.5 * n_b @ (n_b - np.eye(trunc.dim)))


def test_deflation_moves_only_the_zero_mode(params):
    p = params.with_alpha(0.8)
    trunc = FockTruncation(na=6, nb=3)
    big_l = two_mode_liouvillian(p, trunc, eps_z=0.2).toarray()
    shift = -7.0
    deflated = deflate_steady_state(big_l, trunc, shift).toarray()
    before = np.linalg.eigvals(big_l)
    after = np.linalg.eigvals(deflated)
    zero = np.argmin(np.abs(before))
    expected = np.append(np.delete(before, zero), shift)
    distance = np.abs(after[:, None] - expected[None, :])
    assert distance.min(axis=0).max() < 1e-8
    assert distance.min(axis=1).max() < 1e-8


def test_zero_drive_equals_the_parity_block_rate(params):
    p = params.with_alpha(1.2)
    assert _rate(p, SMALL).rate == pytest.approx(bit_flip_rate(p, SMALL), rel=1e-8)


def test_vanishing_drive_limit_is_continuous_and_quadratic(params):
    p = params.with_alpha(1.2)
    base = bit_flip_rate(p, SMALL)
    assert _rate(p, SMALL, eps_z=1e-5 * p.g2).rate == pytest.approx(base, rel=1e-6)
    eps = 1e-3 * p.g2
    small = _rate(p, SMALL, eps_z=eps).rate
    double = _rate(p, SMALL, eps_z=2 * eps).rate
    assert (double - base) / (small - base) == pytest.approx(4.0, rel=1e-3)


def test_drive_raises_the_rate_and_residual_is_small(params):
    p = params.with_alpha(1.2)
    res = _rate(p, SMALL, eps_z=0.2 * p.g2)
    assert res.rate > 2 * bit_flip_rate(p, SMALL)
    assert res.residual < 1e-6 * res.rate


def test_drive_sign_does_not_matter(params):
    # The antiunitary symmetry T = P_a P_b K maps eps_z -> -eps_z up to a unitary.
    p = params.with_alpha(1.2)
    assert _rate(p, SMALL, eps_z=0.2 * p.g2).rate == pytest.approx(
        _rate(p, SMALL, eps_z=-0.2 * p.g2).rate, rel=1e-9
    )


def test_memory_detuning_raises_the_rate(params):
    # delta_a splits the even/odd cat energies: a coherent X_L rotation, damped by Gamma_Z.
    p = params.with_alpha(1.2)
    assert _rate(p, SMALL, delta_a=0.05).rate > 1.1 * _rate(p, SMALL).rate


def test_underdamped_detuning_is_refused(params):
    # Large delta_a: the slow pair is complex (coherent bit-flip oscillation), no single rate.
    p = params.with_alpha(1.2)
    with pytest.raises(ValueError, match="did not converge"):
        _rate(p, SMALL, delta_a=0.5)


def test_gradient_is_returned_for_every_hamiltonian_knob(params):
    p = params.with_alpha(1.0)
    trunc = FockTruncation(na=14, nb=4)
    terms = HamiltonianTerms.from_params(p, eps_z=0.1)
    res = driven_bit_flip_rate(terms, Dissipation.from_params(p), trunc, with_gradient=True)
    assert set(res.gradient) == {k for k, _ in terms.coefficients()}
    assert res.gradient["eps_z"] > 0


def test_unsqueezed_buffer_jump_is_plain_loss():
    trunc = FockTruncation(na=4, nb=3)
    _, b = two_mode_operators(trunc)
    assert abs(buffer_jump(trunc, 0.0, 0.3) - b).max() < 1e-14


def test_squeezed_buffer_jump_definition():
    trunc = FockTruncation(na=4, nb=5)
    _, b = two_mode_operators(trunc)
    r, phi = 0.4, 0.7
    expected = np.cosh(r) * b + np.exp(1j * phi) * np.sinh(r) * b.T
    assert abs(buffer_jump(trunc, r, phi) - expected).max() < 1e-14


def test_invalid_dissipation_is_rejected():
    with pytest.raises(ValueError, match="squeeze_r"):
        Dissipation(kappa_a=0.1, kappa_b=1.0, squeeze_r=-0.1)
    with pytest.raises(ValueError, match="kappa_b"):
        Dissipation(kappa_a=0.1, kappa_b=0.0)


def test_unknown_extra_term_is_rejected(params):
    with pytest.raises(TypeError):
        HamiltonianTerms.from_params(params, eps_z=0.0, not_a_term=1.0)


def test_codeword_contrast_is_linear_and_close_to_the_adiabatic_estimate(params):
    p = params.with_alpha(1.2)
    eps = 0.05 * p.g2
    one = codeword_contrast(_rate(p, SMALL, eps_z=eps), p, SMALL)
    two = codeword_contrast(_rate(p, SMALL, eps_z=2 * eps), p, SMALL)
    linear = -eps / (p.alpha * p.g2)  # <b>_+ - <b>_- = -2 eps_z / (2 alpha g2)
    assert one.real == pytest.approx(linear, rel=0.1)
    assert abs(one.imag) < 0.05 * abs(one.real)
    assert two / one == pytest.approx(2.0, rel=2e-3)


def test_second_order_coefficient_matches_the_full_driven_rate(params):
    p = params.with_alpha(1.2)
    lin = drive_rate_coefficient(*_model(p), SMALL)
    assert lin.rate0 == pytest.approx(bit_flip_rate(p, SMALL), rel=1e-9)
    eps = 1e-3 * p.g2
    fd = (_rate(p, SMALL, eps_z=eps).rate - lin.rate0) / eps**2
    assert lin.coefficient == pytest.approx(fd, rel=1e-4)


def test_second_order_coefficient_on_a_tapered_buffer(params):
    p = params.with_alpha(1.4)
    trunc = FockTruncation(na=20, nb=4, nb_extra=3, na_taper=6.0)
    lin = drive_rate_coefficient(*_model(p), trunc)
    eps = 1e-3 * p.g2
    fd = (_rate(p, trunc, eps_z=eps).rate - lin.rate0) / eps**2
    assert lin.coefficient == pytest.approx(fd, rel=1e-4)


def test_second_order_coefficient_with_a_squeezed_bath(params):
    p = params.with_alpha(1.0)
    trunc = FockTruncation(na=14, nb=5)
    terms = HamiltonianTerms.from_params(p)
    diss = Dissipation.from_params(p, squeeze_r=0.3, squeeze_phi=0.0)
    lin = drive_rate_coefficient(terms, diss, trunc)
    eps = 1e-3 * p.g2
    full = driven_bit_flip_rate(terms.replace(eps_z=eps), diss, trunc).rate
    assert lin.rate0 == pytest.approx(driven_bit_flip_rate(terms, diss, trunc).rate, rel=1e-9)
    assert lin.coefficient == pytest.approx((full - lin.rate0) / eps**2, rel=1e-4)


def test_drive_kernel_integrates_to_the_second_order_coefficient(params):
    p = params.with_alpha(1.0)
    trunc = FockTruncation(na=14, nb=4)
    lin = drive_rate_coefficient(*_model(p), trunc)
    kernel = drive_kernel(*_model(p), trunc, np.linspace(0.0, 8.0, 4001))
    assert kernel.rate0 == pytest.approx(lin.rate0, rel=1e-9)
    assert -np.trapezoid(kernel.values, kernel.taus) == pytest.approx(lin.coefficient, rel=1e-4)


def test_drive_susceptibility_matches_the_kernel_spectrum(params):
    p = params.with_alpha(1.0)
    trunc = FockTruncation(na=14, nb=4)
    omegas = np.array([0.0, 7.0, 20.0])
    chi = drive_susceptibility(*_model(p), trunc, omegas)
    assert chi[0] == pytest.approx(drive_rate_coefficient(*_model(p), trunc).coefficient, rel=1e-9)
    kernel = drive_kernel(*_model(p), trunc, np.linspace(0.0, 8.0, 8001))
    phase = np.exp(-1j * np.outer(omegas, kernel.taus))
    from_kernel = -np.trapezoid(phase * kernel.values, kernel.taus, axis=1).real
    assert np.allclose(chi, from_kernel, rtol=1e-3)


def test_time_domain_pulse_probability_matches_the_kernel_at_weak_drive(params):
    p = params.with_alpha(1.0)
    trunc = FockTruncation(na=14, nb=4)
    terms, diss = _model(p)
    pulse = np.array([0.02, 0.05, 0.03, 0.0, -0.04]) * p.g2
    piece = 0.1
    exact = pulse_bit_flip_probability(terms, diss, trunc, pulse, piece, settle=2.0)
    kernel = drive_kernel(terms, diss, trunc, np.linspace(0.0, 8.0, 8001))
    assert exact.extra == pytest.approx(kernel.pulse_probability(pulse, piece), rel=2e-3)


def test_time_domain_constant_pulse_approaches_the_quasi_static_rate(params):
    p = params.with_alpha(1.0)
    trunc = FockTruncation(na=14, nb=4)
    terms, diss = _model(p)
    eps, duration = 0.05 * p.g2, 6.0
    exact = pulse_bit_flip_probability(terms, diss, trunc, np.array([eps]), duration, settle=2.0)
    quasi = (_rate(p, trunc, eps_z=eps).rate - exact.rate0) * duration
    assert exact.extra == pytest.approx(quasi, rel=0.03)


def test_buffer_response_integrates_to_the_metastable_contrast(params):
    p = params.with_alpha(1.0)
    trunc = FockTruncation(na=14, nb=4)
    eps = 1e-3 * p.g2
    contrast = codeword_contrast(_rate(p, trunc, eps_z=eps), p, trunc)
    resp = buffer_response(*_model(p), trunc, np.linspace(0.0, 6.0, 3001))
    # The two differ at O(3e-4): codeword_contrast projects the codeword on the *driven* slow
    # mode (c_+- depend on eps and multiply the small undriven offset Tr(b r_1)); the kernel
    # starts from the undriven metastable state. Converged in the grid (checked to 1e-6).
    assert np.trapezoid(resp.values, resp.taus) * eps == pytest.approx(contrast, rel=1e-3)


def test_buffer_response_predicts_the_driven_signal(params):
    """Weak AC drive: convolution of the response kernel vs dynamiqs mesolve."""
    p = params.with_alpha(1.0)
    na, nb = 14, 4
    trunc = FockTruncation(na=na, nb=nb)
    pieces, h = 40, 0.025
    t_mid = (np.arange(pieces) + 0.5) * h
    pulse = 0.02 * p.g2 * np.sin(np.pi * t_mid / (pieces * h)) * np.cos(2 * p.alpha * p.g2 * t_mid)
    resp = buffer_response(*_model(p), trunc, np.linspace(0.0, 1.0, 2001))
    times = np.linspace(0.0, pieces * h, 81)
    predicted = resp.response_matrix(pieces, h, times) @ pulse
    a, b = dq.destroy(na, nb)
    h0 = p.g2 * (a @ a @ b.dag() + a.dag() @ a.dag() @ b) - p.eps_d * (b + b.dag())
    drive = dq.pwc(jnp.linspace(0.0, pieces * h, pieces + 1), jnp.asarray(pulse), a + a.dag())
    jumps = [jnp.sqrt(p.kappa_b) * b, jnp.sqrt(p.kappa_a) * a]
    deltas = []
    for sign in (1, -1):
        # start in the undriven metastable well state (as the kernel assumes)
        settle = dq.mesolve(
            h0,
            jumps,
            dq.tensor(dq.coherent(na, sign * p.alpha), dq.fock(nb, 0)),
            [0.0, SETTLE],
            method=dq.method.Tsit5(rtol=1e-10, atol=1e-10),
            progress_meter=False,
        ).states[-1]
        res = dq.mesolve(
            h0 + drive,
            jumps,
            settle,
            jnp.asarray(times),
            exp_ops=[b],
            method=dq.method.Tsit5(rtol=1e-10, atol=1e-10),
            save_states=False,
            progress_meter=False,
        )
        deltas.append(np.asarray(res.expects[0]))
    # the slow amplitude decayed by exp(-Gamma_X SETTLE) while settling; undo it
    measured = (deltas[0] - deltas[1]) * np.exp(bit_flip_rate(p, trunc) * SETTLE)
    measured = measured - measured[0]  # undriven offset (constant on this time scale)
    assert np.max(np.abs(predicted - measured)) < 2e-3 * np.max(np.abs(measured))


@pytest.mark.slow
def test_driven_rate_matches_time_domain_decay_of_re_a(params):
    """Independent check: dynamiqs mesolve with the drive; Re<a> -> 0 by the T symmetry."""
    p = params.with_alpha(0.8)
    na, nb, eps_z = 14, 4, 0.3 * params.g2
    gamma = _rate(p, FockTruncation(na=na, nb=nb), eps_z=eps_z).rate
    a, b = dq.destroy(na, nb)
    h = (
        p.g2 * (a @ a @ b.dag() + a.dag() @ a.dag() @ b)
        - p.eps_d * (b + b.dag())
        + eps_z * (a + a.dag())
    )
    jumps = [jnp.sqrt(p.kappa_b) * b, jnp.sqrt(p.kappa_a) * a]
    psi0 = dq.tensor(dq.coherent(na, p.alpha), dq.fock(nb, 0))
    tsave = jnp.linspace(0.0, 4.0 / gamma, 81)
    res = dq.mesolve(
        h,
        jumps,
        psi0,
        tsave,
        exp_ops=[a],
        method=dq.method.Tsit5(rtol=1e-9, atol=1e-9),
        save_states=False,
        progress_meter=False,
    )
    re_a = np.real(np.asarray(res.expects[0]))
    t = np.asarray(tsave)
    late = t > 0.5 / gamma
    assert fit_decay_rate(t[late], re_a[late]) == pytest.approx(gamma, rel=1e-3)


def test_signal_form_reproduces_the_oscillator_snr(params):
    """A ResponseKernel built from the 2x2 oscillator's impulse response must give back
    LinearReadout.snr_squared (independent quadrature paths)."""
    import scipy.linalg as sla

    from catqubit.optimize import LinearReadout

    model = LinearReadout.from_params(params)
    drift = np.asarray(model.drift())
    taus = np.linspace(0.0, 2.0, 8001)
    # Delta beta = 2 beta; impulse response of beta to eps is [e^{A tau} (-1, 0)]_beta
    impulse = np.array([2 * (sla.expm(drift * t) @ np.array([-1.0, 0.0]))[1] for t in taus])
    kernel = ResponseKernel(taus, impulse.astype(complex))
    pieces, h = 20, 0.075
    pulse = np.linspace(0.1, 0.5, pieces) * params.g2 * np.cos(np.arange(pieces))
    form = kernel.signal_form(pieces, h, params.kappa_b, efficiency=0.7, samples_per_piece=8)
    expected = float(model.snr_squared(jnp.asarray(pulse), pieces * h, 8, efficiency=0.7))
    # residual ~1.5e-5: trapezoid error (dtau^2) of the cumulative kernel integral
    assert float(pulse @ form @ pulse) == pytest.approx(expected, rel=1e-4)


def test_kernel_grids_are_validated(params):
    short = DriveKernel(np.linspace(0.0, 1.0, 11), np.zeros(11), 0.0)
    with pytest.raises(ValueError, match="kernel grid"):
        short.quadratic_form(10, 0.2)
    response = ResponseKernel(np.linspace(0.0, 1.0, 11), np.zeros(11, dtype=complex))
    with pytest.raises(ValueError, match="response grid"):
        response.response_matrix(4, 0.1, np.array([0.0, 2.0]))
    p = params.with_alpha(1.0)
    with pytest.raises(ValueError, match="uniform"):
        drive_kernel(*_model(p), FockTruncation(na=8, nb=3), np.array([0.0, 0.1, 0.3]))


def test_contrast_needs_eigenvectors(params):
    from catqubit.driven_rates import DrivenRate

    with pytest.raises(ValueError, match="eigenvectors"):
        codeword_contrast(DrivenRate(1.0, -1.0, 0.0), params, SMALL)
