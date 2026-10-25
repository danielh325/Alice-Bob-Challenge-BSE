"""Derive the effective rotating-frame Hamiltonian of the pumped ATS (Task 2.4a).

Writes results/task24a/symbolic_terms.json. Pure symbolic algebra (normal_order.py);
numbers come only from evaluating the exact rational formulas.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from catqubit.normal_order import (
    MHZ,
    PUMP,
    READOUT,
    CircuitParams,
    Expr,
    GaussRat,
    OpKey,
    Sym,
    coefficients,
    evaluate_coefficient,
    first_order,
    fock_second_order,
    format_coefficient,
    frequency_components,
    reference_parameters,
    second_order,
    sin_expansion,
    split_linear,
)

# --- symbolic terms ---

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "task24a" / "symbolic_terms.json"
DEGREES = (5, 7, 9)
MAIN_D = 9
EPS_Z_UNIT = 1.0  # rad/us: tone amplitude chosen so that the first-order drive eps_Z = 1 rad/us


@dataclass(frozen=True)
class Target:
    name: str
    op: OpKey
    operator: str
    order: str  # "H1" or "H2"
    pump: int  # power of A
    tone: int  # power of A_z
    factor: int = 1  # convention factor (2 for K/2 terms)


PUMP_TARGETS = (
    Target("g2", (0, 2, 1, 0), "a^2 b^dag", "H1", 1, 0),
    Target("g2_na", (1, 3, 1, 0), "a^dag a^3 b^dag (= a^dag a . a^2 b^dag)", "H1", 1, 0),
    Target("g2_nb", (0, 2, 2, 1), "a^2 b^dag^2 b (= a^2 b^dag . b^dag b)", "H1", 1, 0),
    Target("c_a4b2", (0, 4, 2, 0), "a^4 b^dag^2", "H2", 2, 0),
    Target("Delta_a", (1, 1, 0, 0), "a^dag a", "H2", 2, 0),
    Target("Delta_b", (0, 0, 1, 1), "b^dag b", "H2", 2, 0),
    Target("K_a", (2, 2, 0, 0), "(K_a/2) a^dag^2 a^2", "H2", 2, 0, 2),
    Target("K_b", (0, 0, 2, 2), "(K_b/2) b^dag^2 b^2", "H2", 2, 0, 2),
    Target("chi_ab", (1, 1, 1, 1), "a^dag a b^dag b", "H2", 2, 0),
    Target("c_number", (0, 0, 0, 0), "1", "H2", 2, 0),
)
TONE_TARGETS = (
    Target("eps_Z", (1, 0, 0, 0), "a^dag", "H1", 0, 1),
    Target("eps_Z_na", (2, 1, 0, 0), "a^dag^2 a (= a^dag . a^dag a)", "H1", 0, 1),
    Target("eps_Z_nb", (1, 0, 1, 1), "a^dag b^dag b", "H1", 0, 1),
    Target("pt_a_bdag", (0, 1, 1, 0), "a b^dag (pump x tone)", "H2", 1, 1),
    Target("pt_adag3_b", (3, 0, 0, 1), "a^dag^3 b (pump x tone)", "H2", 1, 1),
    Target("tt_adag2", (2, 0, 0, 0), "a^dag^2 (tone x tone)", "H2", 0, 2),
    Target("tt_stark_a", (1, 1, 0, 0), "a^dag a (tone x tone)", "H2", 0, 2),
    Target("tt_stark_b", (0, 0, 1, 1), "b^dag b (tone x tone)", "H2", 0, 2),
    Target("tt_kerr_a", (2, 2, 0, 0), "(K/2) a^dag^2 a^2 (tone x tone)", "H2", 0, 2, 2),
    Target("tt_chi_ab", (1, 1, 1, 1), "a^dag a b^dag b (tone x tone)", "H2", 0, 2),
)


@dataclass(frozen=True)
class Hamiltonians:
    h1: Expr
    h2: Expr
    h2_disp: Expr
    h2_quantum: Expr


def build(max_degree: int) -> Hamiltonians:
    comps = frequency_components(sin_expansion(max_degree), (PUMP, READOUT))
    lin, nonlin = split_linear(comps)
    top = max_degree + 1  # H2 is complete through phi degree D + 1
    disp = (
        second_order(lin, lin, top)
        + second_order(lin, nonlin, top)
        + second_order(nonlin, lin, top)
    )
    quantum = second_order(nonlin, nonlin, top)
    return Hamiltonians(first_order(comps), disp + quantum, disp, quantum)


def tone_params() -> CircuitParams:
    """Reference circuit with eps_z chosen so that eps_Z = (A_z/2) phi_a E = 1 rad/us."""
    base = reference_parameters()
    eps_z = EPS_Z_UNIT / (base.e_j * base.phi_a * base.envelope)
    return reference_parameters(eps_z=eps_z)


def select(expr: Expr, t: Target) -> dict[Sym, GaussRat]:
    coeff = expr.coefficient(t.op)
    return {
        s: c.times_int(t.factor) for s, c in coeff.items() if (s.pump, s.tone) == (t.pump, t.tone)
    }


def by_degree(coeff: dict[Sym, GaussRat], degrees: set[int]) -> dict[Sym, GaussRat]:
    return {s: c for s, c in coeff.items() if s.phi_degree in degrees}


def numbers(value: complex, t: Target) -> dict:
    """Values in rad/us and MHz (pump terms) or per unit eps_Z (tone terms)."""
    val = value.real if abs(value.imag) < 1e-12 * max(abs(value), 1e-300) else value
    as_list = val if isinstance(val, float) else [val.real, val.imag]
    if t.tone == 0:
        mhz = val / MHZ
        return {
            "value_rad_per_us": as_list,
            "value_mhz": mhz if isinstance(mhz, float) else [mhz.real, mhz.imag],
            "value_khz": (mhz * 1e3) if isinstance(mhz, float) else None,
        }
    unit = {
        1: "dimensionless (coefficient / eps_Z)",
        2: "us/rad (coefficient / eps_Z^2)",
    }
    return {"value_per_eps_Z_power": as_list, "units": unit[t.tone]}


def describe(t: Target, hams: dict[int, Hamiltonians], params: CircuitParams) -> dict:
    main = hams[MAIN_D]
    source = main.h1 if t.order == "H1" else main.h2
    coeff = select(source, t)
    reduced = select(source.reduced(), t)  # exact zeros under w_p = 2 w_a - w_b
    degrees = sorted({s.phi_degree for s in reduced})
    if not degrees:
        return {
            "operator": t.operator,
            "order": t.order,
            "formula": "0 (vanishes identically)",
        }
    lead, nxt = degrees[0], degrees[0] + 2

    def val(c: dict[Sym, GaussRat]) -> complex:
        return evaluate_coefficient(c, params)

    entry = {
        "operator": t.operator,
        "order": t.order,
        "leading_phi_degree": lead,
        "formula": format_coefficient(by_degree(coeff, {lead})),
        "formula_next_order": format_coefficient(by_degree(coeff, {nxt})),
        "formula_reduced_wp_eq_2wa_minus_wb": format_coefficient(by_degree(reduced, {lead})),
        **numbers(val(coeff), t),
        "leading_order": numbers(val(by_degree(coeff, {lead})), t),
        "with_next_order": numbers(val(by_degree(coeff, {lead, nxt})), t),
        "by_phi_degree": {d: numbers(val(by_degree(coeff, {d})), t) for d in degrees},
        "convergence_in_D": {
            str(d): numbers(val(select(h.h1 if t.order == "H1" else h.h2, t)), t)
            for d, h in hams.items()
        },
    }
    if t.order == "H2":
        for label, part in (
            ("disp_part", main.h2_disp),
            ("quantum_part", main.h2_quantum),
        ):
            pc = select(part, t)
            entry[label] = {
                "formula_leading": format_coefficient(by_degree(pc, {lead})),
                "formula_leading_reduced": format_coefficient(
                    by_degree(select(part.reduced(), t), {lead})
                ),
                "leading_order": numbers(val(by_degree(pc, {lead})), t),
                "with_next_order": numbers(val(by_degree(pc, {lead, nxt})), t),
                "all_degrees": numbers(val(pc), t),
            }
    return entry


def other_resonant_ops(main: Hamiltonians, pump: int, tone: int, known: set[OpKey]) -> list:
    ops = sorted(
        {
            op
            for (op, s) in main.h2.terms
            if (s.pump, s.tone) == (pump, tone) and op not in known and s.phi_degree <= 6
        }
    )
    return [list(op) for op in ops]


CONVENTIONS = {
    "hamiltonian": "H = w_a a^dag a + w_b b^dag b + A cos(w_p t) sin(phihat) "
    "[+ A_z cos(w_a t + theta) sin(phihat)], phihat = phi_a(a+a^dag) + phi_b(b+b^dag)",
    "amplitudes": "A = 2 E_J eps_p (pump, phi_Delta = pi/2, phi_Sigma = -pi/2 + eps_p cos w_p t, "
    "first order in eps_p); A_z = 2 E_J eps_z (readout flux tone)",
    "E": "E = exp(-Phi^2/2), Phi^2 = phi_a^2 + phi_b^2, kept exact (H1 ~ E, H2 ~ E^2)",
    "frame": "interaction picture of w_a a^dag a + w_b b^dag b; a^dag^p a^q b^dag^r b^s rotates as "
    "exp(i[(p-q)w_a + (r-s)w_b]t)",
    "H_eff": "H1 + H2 with H1 = zero-frequency part of V_I(t) (RWA), H2 = sum_{nu != 0} "
    "V(nu)V(-nu)/nu (van Vleck / time-averaging, second order); checked on the beam-splitter "
    "AC Stark shift (a-like level shifts by +g^2/(w_a - w_b))",
    "operator_basis": "all coefficients multiply normal-ordered monomials; "
    "H_eff = Delta_a a^dag a + Delta_b b^dag b + (K_a/2) a^dag^2 a^2 + (K_b/2) b^dag^2 b^2 + "
    "chi_ab a^dag a b^dag b + [g2 a^2 b^dag + g2_na a^dag a^3 b^dag + g2_nb a^2 b^dag^2 b + "
    "c_a4b2 a^4 b^dag^2 + h.c.] + c_number",
    "frequencies": "'formula' keeps w_a, w_b, w_p independent (valid for a detuned pump; "
    "tone w_z = w_a folded into w_a); 'formula_reduced_wp_eq_2wa_minus_wb' substitutes "
    "w_p = 2 w_a - w_b (exact cancellations, leading order decided there). Zero frequency "
    "decided at w_a : w_b : w_p = 5 : 6 : 4. Pairs with nu1 + nu2 = 0 only through the "
    "resonance condition use the Hermitian form (1/2) V(nu1) V(nu2) [1/nu1 - 1/nu2]",
    "phi_degree": "total power of phi_a and phi_b in the polynomial prefactor (E excluded); "
    "H1 exact to degree D, H2 exact to degree D+1",
    "disp_vs_quantum": "disp_part = pairs with at least one linear factor (a, a^dag, b, b^dag) "
    "in V(nu)V(-nu): the classical pump-displacement contribution; quantum_part = "
    "nonlinear x nonlinear",
    "sign_of_g2": "g2 (coefficient of a^2 b^dag) is negative in this convention: "
    "-(1/2) E_J eps_p phi_a^2 phi_b E; |g2| matches the prompt formula times E",
    "tone_terms": "readout entries are divided by eps_Z^k with "
    "eps_Z = (A_z/2) phi_a E e^{-i theta} the first-order coefficient of a^dag; "
    "theta = 0 numerically; k = tone power. "
    "Phase law: a^dag, a^dag^2 a, a^dag b^dag b, a b^dag, a^dag^3 b ~ e^{-i theta}; "
    "a^dag^2 ~ e^{-2 i theta}; Stark/Kerr ~ |eps_Z|^2",
    "units": "rad/us (angular); MHz = rad/us / 2pi",
}

FIRST_ORDER_NOTES = (
    "H1 (pump only) contains no number-conserving term (no Stark shift, Kerr, chi, c-number): "
    "every term carries exp(+-i w_p t), and a monomial is resonant only if "
    "5 dn_a + 6 dn_b = -+4 with dn = p - q, r - s. Solutions are (dn_a, dn_b) = "
    "(-2 + 6k, 1 - 5k); sin is odd so p+q+r+s is odd, which forces dn_a + dn_b = k - 1 odd, "
    "i.e. k even. k = 0 gives a^2 b^dag and its number dressings; k = +-2 needs degree >= 19. "
    "So a^dag^4 b^4 (k=1, degree 8) is excluded by parity. With the tone (np = 0, nz = +-1) "
    "the resonant monomials are a^dag dressings only (dn_a, dn_b) = (+-1, 0)."
)


FOCK_N = 14  # per mode; low-photon elements converged (identical at 14, 18, 22)


def fock_crosscheck(terms: dict) -> dict:
    """Compare D = MAIN_D series values with the all-orders-in-phi Fock-space H2."""
    pump = coefficients(fock_second_order(reference_parameters(), FOCK_N))
    tone = coefficients(fock_second_order(tone_params(), FOCK_N))
    out = {
        "method": "H2 = sum V(nu)V(-nu)/nu from exact matrix sin(phihat), N_a = N_b = "
        f"{FOCK_N}; pump-only for pump terms, tone on for pt_a_bdag"
    }
    for name, value in pump.items():
        fock = tone[name].real if name == "pt_a_bdag" else value.real / MHZ
        entry = terms[name]
        series = entry.get("value_mhz", entry.get("value_per_eps_Z_power"))
        out[name] = {
            "fock": fock,
            "series": series,
            "rel_diff": abs(fock - series) / abs(series),
        }
    return out


def main() -> None:
    start = time.time()
    hams = {d: build(d) for d in DEGREES}
    pump_params, t_params = reference_parameters(), tone_params()
    main_h = hams[MAIN_D]
    terms = {t.name: describe(t, hams, pump_params) for t in PUMP_TARGETS}
    terms.update({t.name: describe(t, hams, t_params) for t in TONE_TARGETS})
    known = {t.op for t in PUMP_TARGETS + TONE_TARGETS}
    p = pump_params
    out = {
        "conventions": CONVENTIONS,
        "parameters": {
            "w_a_rad_per_us": p.w_a,
            "w_b_rad_per_us": p.w_b,
            "w_p_rad_per_us": 2 * p.w_a - p.w_b,
            "E_J_rad_per_us": p.e_j,
            "phi_a": p.phi_a,
            "phi_b": p.phi_b,
            "eps_p": p.eps_p,
            "A_rad_per_us": p.amplitude_pump,
            "E": p.envelope,
            "g2_prompt_formula_mhz": 0.5 * p.e_j * p.eps_p * p.phi_a**2 * p.phi_b / MHZ,
            "eps_z_for_unit_eps_Z": t_params.eps_z,
        },
        "degree_D": MAIN_D,
        "degrees_compared": list(DEGREES),
        "first_order_notes": FIRST_ORDER_NOTES,
        "other_second_order_ops_phi_le_6": {
            "pump_x_pump": other_resonant_ops(main_h, 2, 0, known),
            "pump_x_tone": other_resonant_ops(main_h, 1, 1, known),
            "tone_x_tone": other_resonant_ops(main_h, 0, 2, known),
        },
        "terms": terms,
        "fock_crosscheck_all_orders": fock_crosscheck(terms),
        "runtime_s": time.time() - start,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT} in {out['runtime_s']:.1f} s")


if __name__ == "__main__":
    main()
