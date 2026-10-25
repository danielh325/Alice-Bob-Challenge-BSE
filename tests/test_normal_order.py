"""Tests for the exact normal-ordered algebra (normal_order.py).

Run: .venv313/bin/python -m pytest -p no:cacheprovider test_normal_order.py
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

from catqubit.normal_order import (
    IDENTITY as IDENTITY_OP,
)
from catqubit.normal_order import (
    PUMP,
    READOUT,
    CircuitParams,
    Expr,
    GaussRat,
    Sym,
    coefficient_value,
    expr_to_matrix,
    first_order,
    format_coefficient,
    frequency_components,
    frequency_value,
    interaction_picture,
    reference_parameters,
    second_order,
    sin_expansion,
    split_linear,
    wick_product,
)

TWO_PI = 2.0 * math.pi


def _ladder(n: int) -> np.ndarray:
    return np.diag(np.sqrt(np.arange(1, n, dtype=float)), k=1)


def _matrix_of_product(k1, k2, n: int) -> np.ndarray:
    """Fock matrix of the (un-normal-ordered) product of two normal-ordered monomials."""
    params = CircuitParams(w_a=1.0, w_b=1.0, e_j=1.0, phi_a=1.0, phi_b=1.0, eps_p=0.5)
    m1 = expr_to_matrix(Expr.monomial(k1), params, n, n)
    m2 = expr_to_matrix(Expr.monomial(k2), params, n, n)
    return m1 @ m2


def _matrix_of_wick(k1, k2, n: int) -> np.ndarray:
    params = CircuitParams(w_a=1.0, w_b=1.0, e_j=1.0, phi_a=1.0, phi_b=1.0, eps_p=0.5)
    total = np.zeros((n * n, n * n))
    for key, mult in wick_product(k1, k2):
        total = total + mult * expr_to_matrix(Expr.monomial(key), params, n, n)
    return total


def _low_block(n: int, n_max: int) -> np.ndarray:
    """Indices of two-mode Fock states |n_a, n_b> with n_a, n_b <= n_max (a is the slow index)."""
    return np.array([na * n + nb for na in range(n_max + 1) for nb in range(n_max + 1)])


# --------------------------------------------------------------------- Wick reordering


@pytest.mark.parametrize("m,k", [(1, 1), (2, 1), (2, 3), (3, 3)])
def test_wick_single_mode_a_power_times_adag_power(m: int, k: int) -> None:
    n = 10
    idx = _low_block(n, n - 1 - k)
    exact = _matrix_of_product((0, m, 0, 0), (k, 0, 0, 0), n)[np.ix_(idx, idx)]
    wick = _matrix_of_wick((0, m, 0, 0), (k, 0, 0, 0), n)[np.ix_(idx, idx)]
    np.testing.assert_allclose(wick, exact, atol=1e-9)


def test_wick_single_mode_closed_form() -> None:
    # a a^dag = a^dag a + 1
    assert dict(wick_product((0, 1, 0, 0), (1, 0, 0, 0))) == {
        (1, 1, 0, 0): 1,
        (0, 0, 0, 0): 1,
    }
    # b^2 b^dag^2 = b^dag^2 b^2 + 4 b^dag b + 2
    assert dict(wick_product((0, 0, 0, 2), (0, 0, 2, 0))) == {
        (0, 0, 2, 2): 1,
        (0, 0, 1, 1): 4,
        (0, 0, 0, 0): 2,
    }


def test_wick_two_mode_generic_product_matches_fock() -> None:
    n = 9
    k1, k2 = (1, 2, 0, 1), (2, 1, 1, 0)
    idx = _low_block(n, 3)
    exact = _matrix_of_product(k1, k2, n)[np.ix_(idx, idx)]
    wick = _matrix_of_wick(k1, k2, n)[np.ix_(idx, idx)]
    np.testing.assert_allclose(wick, exact, atol=1e-9)


# --------------------------------------------------------------------- Expr basics


def test_dagger_is_involution_and_conjugates() -> None:
    e = Expr.monomial((0, 2, 1, 0), Sym(pump=1, theta=1), GaussRat(Fraction(1, 3), Fraction(2)))
    d = e.dagger()
    assert d.coefficient((2, 0, 0, 1)) == {
        Sym(pump=1, theta=-1): GaussRat(Fraction(1, 3), Fraction(-2))
    }
    assert d.dagger() == e


def test_expr_addition_cancels_to_zero() -> None:
    e = Expr.monomial((1, 0, 0, 0), Sym(phi_a=1), GaussRat(Fraction(1)))
    assert (e + e.scaled(GaussRat(Fraction(-1)))) == Expr.zero()


# --------------------------------------------------------------------- sin expansion


def _fock_sin(params: CircuitParams, n: int) -> np.ndarray:
    a = np.kron(_ladder(n), np.eye(n))
    b = np.kron(np.eye(n), _ladder(n))
    phihat = params.phi_a * (a + a.T) + params.phi_b * (b + b.T)
    vals, vecs = np.linalg.eigh(phihat)
    return (vecs * np.sin(vals)) @ vecs.T


def test_sin_expansion_matches_fock_low_photon_block() -> None:
    """For n_a, n_b <= 2 only monomials with p, q, r, s <= 2 contribute, so D = 7 is exact."""
    params = CircuitParams(w_a=1.0, w_b=1.0, e_j=1.0, phi_a=0.3, phi_b=0.5, eps_p=0.5)
    n = 30
    idx = _low_block(n, 2)
    exact = _fock_sin(params, n)[np.ix_(idx, idx)]
    d7 = expr_to_matrix(sin_expansion(7), params, n, n)[np.ix_(idx, idx)]
    d5 = expr_to_matrix(sin_expansion(5), params, n, n)[np.ix_(idx, idx)]
    np.testing.assert_allclose(d7.real, exact, atol=1e-12)
    assert np.max(np.abs(d5 - exact)) > 1e-6  # the test resolves the degree-7 terms


def test_sin_expansion_is_hermitian_and_odd() -> None:
    s = sin_expansion(7)
    assert s.dagger() == s
    assert all(sum(op) % 2 == 1 for op in s.op_keys())


# --------------------------------------------------------------------- frequencies


def test_frequency_value_at_5_6_ratio() -> None:
    assert frequency_value((0, 0, 1)) == 4  # w_p in units of u = 2pi x 0.8 GHz
    assert frequency_value((1, 0, 0)) == 5
    assert frequency_value((-2, 1, 0)) == -4
    assert frequency_value((-2, 1, 1)) == 0  # a^2 b^dag e^{i w_p t} is resonant


# --------------------------------------------------------------------- second-order sign


def test_ac_stark_sign_on_beam_splitter() -> None:
    """V = g (a^dag b + b^dag a): the a-like level shifts by +g^2/(w_a - w_b)."""
    g, w_a, w_b = 0.01, 1.0, 1.3
    v = Expr.monomial((1, 0, 0, 1), Sym(pump=1), GaussRat(Fraction(1))) + Expr.monomial(
        (0, 1, 1, 0), Sym(pump=1), GaussRat(Fraction(1))
    )
    comps = interaction_picture(v)
    h2 = second_order(comps, comps)
    params = CircuitParams(w_a=w_a, w_b=w_b, e_j=1.0, phi_a=1.0, phi_b=1.0, eps_p=g / 2)
    shift = coefficient_value(h2, (1, 1, 0, 0), params)
    assert shift == pytest.approx(g**2 / (w_a - w_b), rel=1e-12)
    exact_lower = 0.5 * (w_a + w_b) - math.sqrt(0.25 * (w_a - w_b) ** 2 + g**2)
    fourth_order = 2 * (g / (w_a - w_b)) ** 2  # relative size of the neglected g^4 term
    assert shift.real == pytest.approx(exact_lower - w_a, rel=fourth_order)
    assert shift.real < 0  # w_a < w_b: the a-like level is pushed down
    assert coefficient_value(h2, (0, 0, 1, 1), params) == pytest.approx(-shift)


# --------------------------------------------------------------------- first order


def test_reference_parameters_fix_eps_p_from_g2() -> None:
    p = reference_parameters()
    g2 = 0.5 * p.e_j * p.eps_p * p.phi_a**2 * p.phi_b
    assert g2 == pytest.approx(TWO_PI * 0.763)
    assert p.eps_p == pytest.approx(0.0353, abs=2e-4)
    assert p.w_b - 2 * p.w_a == pytest.approx(-TWO_PI * 3200.0)


def test_first_order_g2() -> None:
    h1 = first_order(frequency_components(sin_expansion(7), (PUMP,)))
    assert h1.coefficient((0, 2, 1, 0)) == {
        Sym(phi_a=2, phi_b=1, pump=1, env=1): GaussRat(Fraction(-1, 4))
    }
    p = reference_parameters()
    envelope = math.exp(-(p.phi_a**2 + p.phi_b**2) / 2)
    expected = -0.5 * p.e_j * p.eps_p * p.phi_a**2 * p.phi_b * envelope
    assert coefficient_value(h1, (0, 2, 1, 0), p) == pytest.approx(expected, rel=1e-14)


def test_first_order_pump_only_contains_only_two_photon_exchange() -> None:
    h1 = first_order(frequency_components(sin_expansion(9), (PUMP,)))
    for p, q, r, s in h1.op_keys():
        assert (p - q, r - s) in {(-2, 1), (2, -1)}
    assert h1.dagger() == h1


def test_first_order_readout_drive() -> None:
    h1 = first_order(frequency_components(sin_expansion(5), (PUMP, READOUT)))
    assert h1.coefficient((1, 0, 0, 0)) == {
        Sym(phi_a=1, tone=1, env=1, theta=-1): GaussRat(Fraction(1, 2))
    }


# --------------------------------------------------------------------- second order


@pytest.fixture(scope="module")
def pumped_and_read() -> dict:
    return frequency_components(sin_expansion(5), (PUMP, READOUT))


def test_second_order_is_hermitian(pumped_and_read) -> None:
    h2 = second_order(pumped_and_read, pumped_and_read, max_phi_degree=6)
    assert h2.dagger() == h2
    assert len(h2.op_keys()) > 10


def test_second_order_commutes_with_total_parity(pumped_and_read) -> None:
    h2 = second_order(pumped_and_read, pumped_and_read, max_phi_degree=6)
    assert all(sum(op) % 2 == 0 for op in h2.op_keys())


def test_second_order_pump_only_has_kerr_and_a4b2() -> None:
    comps = frequency_components(sin_expansion(7), (PUMP,))
    h2 = second_order(comps, comps, max_phi_degree=8)
    assert h2.coefficient((2, 2, 0, 0))
    assert h2.coefficient((0, 4, 2, 0))
    for p, q, r, s in h2.op_keys():
        assert (p - q, r - s) in {(0, 0), (-4, 2), (4, -2)}


def test_displacement_plus_quantum_split_is_complete(pumped_and_read) -> None:
    lin, nonlin = split_linear(pumped_and_read)
    total = second_order(pumped_and_read, pumped_and_read, max_phi_degree=6)
    parts = (
        second_order(lin, lin, max_phi_degree=6)
        + second_order(lin, nonlin, max_phi_degree=6)
        + second_order(nonlin, lin, max_phi_degree=6)
        + second_order(nonlin, nonlin, max_phi_degree=6)
    )
    assert parts == total


def test_format_coefficient_mentions_symbols() -> None:
    h1 = first_order(frequency_components(sin_expansion(5), (PUMP,)))
    text = format_coefficient(h1.coefficient((0, 2, 1, 0)))
    assert "-1/4" in text and "A" in text and "phi_a^2" in text and "phi_b" in text


def test_linear_drive_gives_only_c_number_symbolically() -> None:
    """g(a^dag + a): a^dag a/w_a + a a^dag/(-w_a) = -1/w_a exactly (denominators canonical)."""
    v = Expr.monomial((1, 0, 0, 0), Sym(pump=1)) + Expr.monomial((0, 1, 0, 0), Sym(pump=1))
    comps = interaction_picture(v)
    h2 = second_order(comps, comps)
    assert h2.op_keys() == {(0, 0, 0, 0)}
    assert h2.coefficient((0, 0, 0, 0)) == {
        Sym(pump=2, denoms=((1, 0, 0),)): GaussRat(Fraction(-1))
    }


def test_c_a4b2_vanishes_at_leading_phi_degree() -> None:
    """Pure a^k b^dag^l factors commute, so the phi^6 part of a^4 b^dag^2 cancels exactly."""
    comps = frequency_components(sin_expansion(7), (PUMP,))
    h2 = second_order(comps, comps, max_phi_degree=8)
    degrees = {s.phi_degree for s in h2.coefficient((0, 4, 2, 0))}
    assert degrees == {8}


def test_reduced_merges_denominators_equal_under_resonance() -> None:
    """w_b + w_p = 2 w_a when w_p = 2 w_a - w_b; reduced() substitutes this exactly."""
    e = Expr.monomial((1, 1, 0, 0), Sym(denoms=((0, 1, 1),))) + Expr.monomial(
        (1, 1, 0, 0), Sym(denoms=((2, 0, 0),)), GaussRat(Fraction(-1))
    )
    assert e != Expr.zero()
    assert e.reduced() == Expr.zero()
    # sign canonicalisation: 1/(w_a - w_p) = -1/(w_b - w_a) after reduction
    f = Expr.monomial(IDENTITY_OP, Sym(denoms=((1, 0, -1),)))
    assert f.reduced() == Expr.monomial(IDENTITY_OP, Sym(denoms=((-1, 1, 0),)))


def test_general_formula_evaluates_with_detuned_pump() -> None:
    """Denominators keep w_p independent, so a detuned pump changes the numbers."""
    comps = frequency_components(sin_expansion(5), (PUMP,))
    h2 = second_order(comps, comps, max_phi_degree=6)
    p0 = reference_parameters()
    p1 = CircuitParams(
        p0.w_a,
        p0.w_b,
        p0.e_j,
        p0.phi_a,
        p0.phi_b,
        p0.eps_p,
        w_p=1.01 * p0.pump_frequency,
    )
    d0 = coefficient_value(h2, (0, 0, 1, 1), p0)
    d1 = coefficient_value(h2, (0, 0, 1, 1), p1)
    assert d0 != pytest.approx(d1, rel=1e-4)
    assert coefficient_value(h2.reduced(), (0, 0, 1, 1), p0) == pytest.approx(d0, rel=1e-12)


def test_series_h2_matches_all_orders_fock_h2() -> None:
    """Fock-space sum V(nu)V(-nu)/nu built from the exact matrix sin(phihat) agrees with the
    D = 9 normal-ordered series to the size of the first omitted order (phi^12)."""
    from catqubit.normal_order import coefficients, fock_second_order

    p = reference_parameters()
    fock = coefficients(fock_second_order(p, 12))
    comps = frequency_components(sin_expansion(9), (PUMP,))
    h2 = second_order(comps, comps, max_phi_degree=10)
    series = {
        "Delta_a": coefficient_value(h2, (1, 1, 0, 0), p),
        "Delta_b": coefficient_value(h2, (0, 0, 1, 1), p),
        "K_b": 2 * coefficient_value(h2, (0, 0, 2, 2), p),
        "chi_ab": coefficient_value(h2, (1, 1, 1, 1), p),
        "c_a4b2": coefficient_value(h2, (0, 4, 2, 0), p),
    }
    for name, value in series.items():
        assert fock[name].real == pytest.approx(value.real, rel=1e-3), name
    assert fock["Delta_b"].real == pytest.approx(series["Delta_b"].real, rel=1e-6)


def test_format_frequency_and_phase() -> None:
    from catqubit.normal_order import canonical_denominator, format_frequency

    assert format_frequency((1, 0, -1)) == "w_a - w_p"
    assert format_frequency((0, 2, 1)) == "2 w_b + w_p"
    text = format_coefficient(
        {Sym(tone=1, theta=-1, denoms=((1, 0, -1),)): GaussRat(Fraction(0), Fraction(1, 3))}
    )
    assert text == "1/3i/(w_a - w_p) * A_z e^{-1i theta}"
    with pytest.raises(ZeroDivisionError):
        canonical_denominator((-2, 1, 1))


def test_coefficient_value_by_phi_degree() -> None:
    comps = frequency_components(sin_expansion(7), (PUMP,))
    h2 = second_order(comps, comps, max_phi_degree=8)
    p = reference_parameters()
    total = coefficient_value(h2, (0, 0, 1, 1), p)
    parts = sum(coefficient_value(h2, (0, 0, 1, 1), p, [d]) for d in (4, 6, 8))
    assert parts == pytest.approx(total, rel=1e-12)
    assert hash(h2) == hash(h2.dagger())
