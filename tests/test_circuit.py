"""Tests for the circuit-level ATS Hamiltonian, its effective expansion and its Floquet check."""

import dataclasses
import math

import numpy as np
import pytest
from scipy.special import eval_genlaguerre, j1

from catqubit.circuit import (
    ATSParams,
    DriveComponent,
    FluxTone,
    FockSpace,
    diagonal_coefficients,
    displacement_operator,
    dressed_g2,
    drive_components,
    effective_hamiltonian,
    first_order_hamiltonian,
    flux_drive,
    frame_harmonics,
    josephson_operator,
    lab_hamiltonian,
    match_floquet_states,
    phase_operator,
    pump_displacement,
    quasi_energies,
    rotating_frame_floquet,
    second_order_hamiltonian,
    two_mode_ladders,
)

TWO_PI = 2 * math.pi
G2_MHZ = 0.763


@pytest.fixture
def ats() -> ATSParams:
    return ATSParams.prompt()


# --- parameters and periodicity ------------------------------------------------------------


def test_prompt_parameters_in_rad_per_us(ats):
    assert ats.omega_a == pytest.approx(TWO_PI * 4000)
    assert ats.omega_b == pytest.approx(TWO_PI * 4800)
    assert ats.omega_p == pytest.approx(TWO_PI * 3200)
    assert ats.e_j == pytest.approx(TWO_PI * 40000)
    assert (ats.phi_a, ats.phi_b) == (0.06, 0.3)


def test_pump_amplitude_is_fixed_by_the_g2_formula(ats):
    assert ats.g2_formula == pytest.approx(TWO_PI * G2_MHZ)
    assert ats.g2_formula == pytest.approx(0.5 * ats.e_j * ats.eps_p * ats.phi_a**2 * ats.phi_b)
    assert ats.eps_p == pytest.approx(0.035, abs=1e-3)


def test_frequencies_are_exactly_5_6_4_and_pump_matches_two_photon_resonance(ats):
    assert (ats.harmonic_a, ats.harmonic_b, ats.harmonic_p) == (5, 6, 4)
    assert ats.omega_p == pytest.approx(2 * ats.omega_a - ats.omega_b)
    assert ats.period == pytest.approx(1.25e-3)  # us


def test_params_are_immutable_and_validated(ats):
    with pytest.raises(dataclasses.FrozenInstanceError):
        ats.eps_p = 0.1
    with pytest.raises(ValueError, match="unit"):
        dataclasses.replace(ats, unit=0.0)
    with pytest.raises(ValueError, match="phi"):
        dataclasses.replace(ats, phi_b=-0.1)


def test_with_pump_returns_new_instance(ats):
    weaker = ats.with_pump(0.01)
    assert weaker.eps_p == 0.01
    assert ats.eps_p != 0.01


def test_frame_rotation_is_identity_after_one_period(ats):
    space = FockSpace(4, 3)
    phases = np.exp(1j * frame_harmonics(ats, space).diagonal_energies * ats.unit * ats.period)
    np.testing.assert_allclose(phases, 1.0, atol=1e-9)


def test_lab_hamiltonian_has_period_1p25_ns_and_pump_period_quarter(ats):
    space = FockSpace(4, 3)
    tones = (FluxTone(ats.eps_p, ats.harmonic_p), FluxTone(1e-3, ats.harmonic_a, 0.3))
    t = 0.37e-3
    h0 = lab_hamiltonian(ats, space, tones, t)
    np.testing.assert_allclose(lab_hamiltonian(ats, space, tones, t + ats.period), h0, atol=1e-6)
    pump_only = tones[:1]
    hp = lab_hamiltonian(ats, space, pump_only, t)
    np.testing.assert_allclose(
        lab_hamiltonian(ats, space, pump_only, t + ats.period / 4), hp, atol=1e-6
    )
    assert np.abs(lab_hamiltonian(ats, space, tones, t + ats.period / 4) - h0).max() > 1.0


# --- operators ---------------------------------------------------------------------------


def test_fock_space_validates_and_indexes_memory_major():
    space = FockSpace(3, 2)
    assert space.dim == 6
    assert space.index(1, 1) == 3
    with pytest.raises(ValueError):
        FockSpace(0, 2)
    a, b = two_mode_ladders(space)
    vac = np.zeros(6)
    vac[space.index(1, 0)] = 1.0
    np.testing.assert_allclose(a @ vac, np.eye(6)[space.index(0, 0)])
    np.testing.assert_allclose(b.T @ np.eye(6)[0], np.eye(6)[space.index(0, 1)])


def test_josephson_operator_at_sweet_spot_is_minus_sin_phi(ats):
    space = FockSpace(5, 4)
    big = FockSpace(30, 30)
    phi = phase_operator(ats, big)
    w, v = np.linalg.eigh(phi)
    sin_big = (v * np.sin(w)) @ v.conj().T
    keep = big.embed_indices(space)
    expected = -sin_big[np.ix_(keep, keep)]
    np.testing.assert_allclose(josephson_operator(ats, space), expected, atol=1e-12)


def test_sin_phi_matrix_element_matches_closed_form(ats):
    """<0,1| sin(phi) |2,0> = -phi_a^2 phi_b exp(-Phi^2/2) / sqrt(2) (normal-ordering factor)."""
    space = FockSpace(4, 3)
    sin_phi = -josephson_operator(ats, space)
    big_phi2 = ats.phi_a**2 + ats.phi_b**2
    expected = -(ats.phi_a**2) * ats.phi_b * math.exp(-big_phi2 / 2) / math.sqrt(2)
    assert sin_phi[space.index(0, 1), space.index(2, 0)] == pytest.approx(expected, rel=1e-10)


def test_enlarged_space_matters_for_matrix_function(ats):
    """Exponentiating the truncated phase operator would be wrong near the truncation edge."""
    space = FockSpace(4, 3)
    phi_small = phase_operator(ats, space)
    w, v = np.linalg.eigh(phi_small)
    naive = -(v * np.sin(w)) @ v.T
    exact = josephson_operator(ats, space)
    edge = space.index(3, 2)
    assert abs(naive[edge, edge - 1] - exact[edge, edge - 1]) > 1e-4


def test_displacement_operator_vacuum_overlap(ats):
    space = FockSpace(6, 6)
    d = displacement_operator(space, 0.1, -0.3)
    vac = d[:, 0]
    assert abs(vac[space.index(0, 0)]) ** 2 == pytest.approx(math.exp(-(0.01 + 0.09)), rel=1e-9)
    assert vac[space.index(1, 0)] == pytest.approx(0.1 * math.exp(-0.05), rel=1e-9)


# --- drive decomposition -------------------------------------------------------------------


def test_flux_drive_at_sweet_spot_is_odd_bessel_series(ats):
    tone = (FluxTone(ats.eps_p, ats.harmonic_p),)
    t = 0.21e-3
    expected = -2 * ats.e_j * math.sin(ats.eps_p * math.cos(ats.omega_p * t))
    assert flux_drive(ats, tone, t) == pytest.approx(expected, rel=1e-12)
    comps = drive_components(ats, tone)
    assert {c.harmonic for c in comps} == {4, -4}
    for c in comps:
        assert c.coefficient == pytest.approx(-2 * ats.e_j * j1(ats.eps_p))


def test_drive_components_carry_tone_phase(ats):
    comps = drive_components(ats, (FluxTone(1e-3, 5, 0.7),))
    plus = next(c for c in comps if c.harmonic == 5)
    minus = next(c for c in comps if c.harmonic == -5)
    assert np.angle(plus.coefficient / minus.coefficient) == pytest.approx(1.4)


# --- first-order (RWA) Hamiltonian -----------------------------------------------------------


def _pump(ats: ATSParams) -> tuple[FluxTone, ...]:
    return (FluxTone(ats.eps_p, ats.harmonic_p),)


def _bessel_factor(eps: float) -> float:
    return 2 * j1(eps) / eps


def test_first_order_recovers_prompt_g2_from_cubic_expansion(ats):
    """With sin(phi) -> phi - phi^3/6, the resonant term is exactly g2 (a^2 b^dag + h.c.)."""
    space = FockSpace(4, 3)
    phi = phase_operator(ats, FockSpace(12, 12))
    keep = FockSpace(12, 12).embed_indices(space)
    cubic = -(phi - phi @ phi @ phi / 6)[np.ix_(keep, keep)]
    h1 = first_order_hamiltonian(ats, space, _pump(ats), coupling=cubic)
    element = h1[space.index(0, 1), space.index(2, 0)]
    expected = ats.g2_formula * _bessel_factor(ats.eps_p) * math.sqrt(2)
    assert abs(element) == pytest.approx(expected, rel=1e-10)


def test_first_order_full_sine_g2_carries_normal_ordering_factor(ats):
    space = FockSpace(4, 3)
    h1 = first_order_hamiltonian(ats, space, _pump(ats))
    g2_vac = dressed_g2(h1, space, 2, 0)
    big_phi2 = ats.phi_a**2 + ats.phi_b**2
    expected = ats.g2_formula * _bessel_factor(ats.eps_p) * math.exp(-big_phi2 / 2)
    assert abs(g2_vac) == pytest.approx(expected, rel=1e-10)


def test_first_order_keeps_only_two_photon_exchange(ats):
    space = FockSpace(6, 5)
    h1 = first_order_hamiltonian(ats, space, _pump(ats))
    np.testing.assert_allclose(np.diag(h1), 0.0, atol=1e-12)
    np.testing.assert_allclose(h1, h1.conj().T, atol=1e-12)
    for m in range(space.dim):
        for n in range(space.dim):
            if abs(h1[m, n]) < 1e-9:
                continue
            (ma, mb), (na_, nb_) = space.occupations(m), space.occupations(n)
            assert (ma - na_, mb - nb_) in {(-2, 1), (2, -1)}


@pytest.mark.parametrize(("n_a", "n_b"), [(2, 0), (3, 0), (5, 0), (2, 2), (4, 1)])
def test_dressed_g2_matches_laguerre_closed_form(ats, n_a, n_b):
    space = FockSpace(7, 5)
    h1 = first_order_hamiltonian(ats, space, _pump(ats))
    xa, xb = ats.phi_a**2, ats.phi_b**2
    closed = (
        ats.g2_formula
        * _bessel_factor(ats.eps_p)
        * math.exp(-(xa + xb) / 2)
        * eval_genlaguerre(n_a - 2, 2, xa)
        / math.comb(n_a, 2)
        * eval_genlaguerre(n_b, 1, xb)
        / (n_b + 1)
    )
    assert abs(dressed_g2(h1, space, n_a, n_b)) == pytest.approx(closed, rel=1e-9)


def test_capacitive_memory_drive_is_not_generated_by_the_pump(ats):
    space = FockSpace(5, 4)
    h1 = first_order_hamiltonian(ats, space, _pump(ats))
    assert abs(h1[space.index(1, 0), space.index(0, 0)]) < 1e-12


def test_flux_tone_at_memory_frequency_gives_memory_drive(ats):
    eps_z = 1e-4
    space = FockSpace(5, 4)
    h1 = first_order_hamiltonian(ats, space, (FluxTone(eps_z, ats.harmonic_a),))
    drive = h1[space.index(1, 0), space.index(0, 0)]
    big_phi2 = ats.phi_a**2 + ats.phi_b**2
    expected = ats.e_j * eps_z * ats.phi_a * math.exp(-big_phi2 / 2) * _bessel_factor(eps_z)
    assert abs(drive) == pytest.approx(expected, rel=1e-9)
    assert abs(h1[space.index(0, 1), space.index(0, 0)]) < 1e-12  # buffer not resonantly driven


# --- second-order Hamiltonian -----------------------------------------------------------------


def test_second_order_reproduces_two_level_ac_stark_shift():
    """sigma_x drive at detuning Delta: H2 = (Omega^2 / 4 Delta) sigma_z, here in units of u."""
    coupling = np.array([[0.0, 1.0], [1.0, 0.0]])
    energies = np.array([0, 3])  # excited state at 3u in the frame
    omega_rabi, unit = 0.2, 1.0
    comps = (DriveComponent(omega_rabi / 2, 1), DriveComponent(omega_rabi / 2, -1))
    h2 = second_order_hamiltonian(coupling, energies, comps, unit)
    detuning = 3 - 1  # rotating-wave part; counter-rotating part at 3 + 1
    shift = omega_rabi**2 / 4 * (1 / detuning + 1 / (3 + 1))
    np.testing.assert_allclose(h2, np.diag([-shift, shift]), atol=1e-14)


def test_linear_coupling_gives_only_a_c_number_at_second_order(ats):
    big, space = FockSpace(7, 6), FockSpace(5, 4)
    phi = phase_operator(ats, big)
    energies = frame_harmonics(ats, big).diagonal_energies
    h2_big = second_order_hamiltonian(-phi, energies, drive_components(ats, _pump(ats)), ats.unit)
    keep = big.embed_indices(space)  # away from the truncation edge, where [a, a^dag] != 1
    h2 = h2_big[np.ix_(keep, keep)]
    c = h2[0, 0]
    np.testing.assert_allclose(h2, c * np.eye(space.dim), atol=1e-9 * abs(c))
    amp = 2 * ats.e_j * 2 * j1(ats.eps_p)  # drive amplitude on phi
    expected = -sum(
        (amp * ph / 2) ** 2 * (1 / (w - ats.omega_p) + 1 / (w + ats.omega_p))
        for ph, w in ((ats.phi_a, ats.omega_a), (ats.phi_b, ats.omega_b))
    )
    assert c == pytest.approx(expected, rel=1e-10)


def test_second_order_is_hermitian_and_conserves_total_parity(ats):
    space = FockSpace(6, 5)
    h2 = effective_hamiltonian(ats, space, _pump(ats), order=2) - effective_hamiltonian(
        ats, space, _pump(ats), order=1
    )
    np.testing.assert_allclose(h2, h2.conj().T, atol=1e-9)
    parity = np.array([(-1) ** sum(space.occupations(i)) for i in range(space.dim)])
    np.testing.assert_allclose(h2 * np.subtract.outer(parity, parity), 0.0, atol=1e-9)


def test_second_order_terms_scale_as_pump_squared(ats):
    space = FockSpace(4, 3)
    big = diagonal_coefficients(effective_hamiltonian(ats, space, _pump(ats)), space)
    half = ats.with_pump(ats.eps_p / 2)
    small = diagonal_coefficients(effective_hamiltonian(half, space, _pump(half)), space)
    for key in ("stark_a", "stark_b", "chi_ab"):
        assert big[key] / small[key] == pytest.approx(4.0, rel=1e-3)


def test_diagonal_coefficients_recover_synthetic_hamiltonian():
    space = FockSpace(4, 4)
    a, b = two_mode_ladders(space)
    na, nb = a.T @ a, b.T @ b
    coeffs = {"stark_a": 1.5, "stark_b": -2.0, "kerr_a": 0.3, "kerr_b": -0.7, "chi_ab": 0.11}
    h = (
        coeffs["stark_a"] * na
        + coeffs["stark_b"] * nb
        + coeffs["kerr_a"] / 2 * (a.T @ a.T @ a @ a)
        + coeffs["kerr_b"] / 2 * (b.T @ b.T @ b @ b)
        + coeffs["chi_ab"] * na @ nb
        + 5.0 * np.eye(space.dim)
    )
    got = diagonal_coefficients(h, space)
    for key, value in coeffs.items():
        assert got[key] == pytest.approx(value, abs=1e-12)


# --- Floquet check ----------------------------------------------------------------------------


def test_pump_displacement_matches_linear_response(ats):
    beta_a, beta_b = pump_displacement(ats)
    amp = 2 * ats.e_j * 2 * j1(ats.eps_p) * math.exp(-(ats.phi_a**2 + ats.phi_b**2) / 2)
    for beta, ph, w in ((beta_a, ats.phi_a, ats.omega_a), (beta_b, ats.phi_b, ats.omega_b)):
        assert beta == pytest.approx(-amp * ph * w / (w**2 - ats.omega_p**2), rel=1e-12)


def test_floquet_without_drive_is_free_detuned_evolution(ats):
    detuned = dataclasses.replace(ats, detuning_a=TWO_PI * 3.0, detuning_b=-TWO_PI * 1.0)
    space = FockSpace(3, 2)
    u = rotating_frame_floquet(detuned, space, (FluxTone(0.0, 4),))
    occ = np.array([space.occupations(i) for i in range(space.dim)])
    energies = occ @ np.array([detuned.detuning_a, detuned.detuning_b])
    np.testing.assert_allclose(u, np.diag(np.exp(-1j * energies * ats.period)), atol=1e-9)


def test_floquet_propagator_is_unitary_and_periodic(ats):
    space = FockSpace(3, 3)
    u1 = rotating_frame_floquet(ats, space, _pump(ats))
    np.testing.assert_allclose(u1.conj().T @ u1, np.eye(space.dim), atol=1e-8)
    u2 = rotating_frame_floquet(ats, space, _pump(ats), periods=2)
    np.testing.assert_allclose(u2, u1 @ u1, atol=1e-8)


def test_quasi_energies_fold_into_first_zone(ats):
    phases = np.array([0.1, -3.0, 3.1])
    u = np.diag(np.exp(-1j * phases))
    eps = quasi_energies(u, ats.period)
    half_zone = ats.unit / 2
    assert np.all(np.abs(eps) <= half_zone + 1e-9)
    np.testing.assert_allclose(np.sort(eps), np.sort(phases / ats.period), rtol=1e-12)


def test_floquet_quasi_energies_match_effective_hamiltonian_to_higher_order(ats):
    """Exact Floquet vs H1 + H2 on the same truncated model: the residual on interior states
    (away from the truncation edge) falls faster than eps_p^2, i.e. the expansion is consistent."""
    space = FockSpace(6, 5)
    residuals = []
    for scale in (1.0, 0.5):
        p = ats.with_pump(ats.eps_p * scale)
        tones = _pump(p)
        u = rotating_frame_floquet(p, space, tones)
        h_eff = effective_hamiltonian(p, space, tones, pad=0)  # same truncated model
        match = match_floquet_states(u, h_eff, space, p)
        interior = np.array([n_a <= 3 and n_b <= 1 for n_a, n_b in match.labels])
        assert match.overlaps[interior].min() > 0.99
        shifts = (match.effective - match.effective[0])[interior]
        res = np.abs((match.floquet - match.floquet[0])[interior] - shifts).max()
        assert res < 0.03 * np.abs(shifts).max()
        residuals.append(res)
    assert residuals[0] / residuals[1] > 6.0


# --- validation paths ------------------------------------------------------------------------


@pytest.mark.parametrize(("field", "value"), [("e_j", 0.0), ("eps_p", -1e-3)])
def test_invalid_energy_or_pump_rejected(ats, field, value):
    with pytest.raises(ValueError, match=field.split("_")[0].upper() if field == "e_j" else field):
        dataclasses.replace(ats, **{field: value})


def test_embed_rejects_larger_subspace():
    with pytest.raises(ValueError, match="fit"):
        FockSpace(3, 3).embed_indices(FockSpace(4, 3))


def test_effective_hamiltonian_rejects_unknown_order(ats):
    with pytest.raises(ValueError, match="order"):
        effective_hamiltonian(ats, FockSpace(3, 2), _pump(ats), order=3)


def test_match_reports_min_overlap(ats):
    space = FockSpace(4, 3)
    weak = ats.with_pump(ats.eps_p / 4)
    u = rotating_frame_floquet(weak, space, _pump(weak))
    match = match_floquet_states(
        u, effective_hamiltonian(weak, space, _pump(weak), pad=0), space, weak
    )
    assert match.min_overlap == pytest.approx(match.overlaps.min())
    assert match.labels[0] == (0, 0)

