"""Extract every Track 1 number the slides use from the executed notebook.

Each claim is read from the printed output of a cell in intro_track.ipynb, so
the deck can only quote what a simulation (catqubit + dynamiqs) actually
printed. Inputs (parameters chosen before simulating) are listed separately
with the source line that sets them. Derived claims are simple arithmetic on
extracted claims, and the formula is stored next to the value.

Run:  .venv313/bin/python tools/extract_task1_claims.py
Out:  results/task1/claims.json
"""

import hashlib
import importlib.metadata
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "intro_track.ipynb"
OUT = ROOT / "results" / "task1" / "claims.json"
LIBRARIES = ("dynamiqs", "jax", "numpy", "scipy", "matplotlib")
NUM = r"([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)"

# (key, cell, regex with one or more groups, group names). A single name
# takes group 1; a tuple of names takes groups 1..n in order.
SPECS = [
    ("g2", 1, rf"^g2\s+= {NUM}"),
    ("kappa_b", 1, rf"^kappa_b = {NUM}"),
    ("kappa_a", 1, rf"^kappa_a = {NUM}"),
    ("eps_d", 1, rf"^eps_d .*= {NUM} rad"),
    (("kappa2", "kappa2_mhz"), 1, rf"^kappa2 .*= {NUM} rad/us  \(kappa2/2pi = {NUM} MHz"),
    ("a0_max_err", 3, rf"max \|simulation - analytic\| = {NUM}"),
    ("a0_min_fid", 3, rf"min fidelity .* = {NUM}"),
    ("b0_n_final", 5, rf"^<n>\(3 us\)\s+= {NUM}"),
    ("b0_odd_pop", 5, rf"^\(1 - e\^-8\) / 2\s+= {NUM}"),
    (("b0_p0_final", "b0_p1_final"), 5, rf"final P\(\|0>\), P\(\|1>\) = {NUM}, {NUM}"),
    ("b0_p01_init", 5, rf"initial P\(\|0>\) \+ P\(\|1>\) = {NUM}"),
    ("starter_peak", 9, rf"^peak <a\^dag a> = {NUM} at"),
    ("starter_b_peak", 9, rf"^peak <b\^dag b> = {NUM}"),
    ("starter_top_buffer", 9, rf"top buffer level\s+= {NUM}"),
    (("na_conv", "nb_conv"), 12, rf"N_a = {NUM}, N_b = {NUM}"),
    (("tol6_steady", "tol6_min"), 15, rf"tol=1e-06: steady {NUM}, .*\(min {NUM}\)"),
    (("tol9_steady", "tol9_min"), 15, rf"tol=1e-09: steady {NUM}, .*\(min {NUM}\)"),
    (("n_ss", "n_ss_pct"), 18, rf"converged steady state <n_a>\s+= {NUM}\s+\({NUM}% vs"),
    ("n_lossless", 18, rf"same with kappa_a = 0\s+= {NUM}"),
    ("n_tanh", 18, rf"tanh\(alpha\^2\) \(even cat\)\s+= {NUM}"),
    ("n_shift", 18, rf"shift from kappa_a\s+= {NUM}"),
    ("n_shift_mf", 18, rf"mean-field estimate .*= {NUM}"),
    (("peak_n", "peak_t_ns"), 20, rf"^peak <a\^dag a> = {NUM} at t = {NUM} ns"),
    (("settle_us", "n_3us"), 20, rf"within 1% of alpha\^2 from t = {NUM} us; <a\^dag a>\(3 us\) = {NUM}"),
    (("b_peak", "b_peak_t_ns", "b_end"), 20, rf"peak <b\^dag b> = {NUM} at t = {NUM} ns, <b\^dag b>\(3 us\) = {NUM}"),
    (("top2_mem", "top_buffer"), 20, rf"top-2 memory population = {NUM}, max top buffer population = {NUM}"),
    (("g2_over_kb", "epsd_over_kb"), 20, rf"g2/kappa_b = {NUM}, eps_d/kappa_b = {NUM}"),
    ("parity_control", 22, rf"kappa_a = 0 control: .* = {NUM}"),
    ("coh_decay", 22, rf"decay of cat coherence = {NUM}"),
    ("parity_decay", 22, rf"decay of parity\s+= {NUM}"),
    (("gamma_pred", "lifetime_us"), 22, rf"prediction 2 kappa_a alpha\^2\s+= {NUM} /us .* = {NUM} us"),
    (("liou_gamma", "liou_lifetime"), 22, rf"\(32, 10\): Gamma = {NUM} /us, lifetime {NUM}"),
    (("mix_fid", "mix_n", "mix_parity", "mix_coh", "mix_purity"), 24,
     rf"equal mixture .*F\(\+alpha\) = {NUM}, <n> = {NUM}, <P> = {NUM}, coherence = {NUM}, purity = {NUM}"),
    (("prep_fid", "prep_n", "prep_parity", "prep_coh", "prep_purity"), 24,
     rf"prepared state, t = 1 us: F\(\+alpha\) = {NUM}, <n> = {NUM}, <P> = {NUM}, coherence = {NUM}, purity = {NUM}"),
    ("a_monitor", 26, rf"method A runs: {NUM}"),
    ("b_monitor", 28, rf"method B runs: {NUM}"),
    (("b_area", "b_T_ns", "b_mhz", "b_fid", "b_re_a"), 28,
     rf"chosen pulse: A = {NUM}, T = {NUM} ns, eps/2pi = {NUM} MHz -> F\(\+alpha\) = {NUM}, Re<a> = {NUM}"),
    ("big_pulse_fid", 31, rf"pulse A = 4, T = 10 ns: F\(\+alpha\) = {NUM}"),
    ("big_beta_fid", 31, rf"displaced start beta = 4: F\(\+alpha\) = {NUM}"),
    ("eff_beta4_fid", 31, rf"effective model, beta = 4: F\(\+alpha\) = {NUM}"),
    ("lambda", 36, rf"lambda = kappa_2 alpha\^2 = {NUM}"),
    (("weak005_model", "weak005_sim"), 36, rf"weak push 0.05 g2 .*model F = {NUM}, simulated F = {NUM}"),
    (("weak02_model", "weak02_sim"), 36, rf"weak push 0.2 g2 .*model F = {NUM}, simulated F = {NUM}"),
    ("c_monitor", 38, rf"method C runs: {NUM}"),
    (("robust_a_before", "robust_a_after"), 40, rf"method A: error {NUM} -> {NUM}"),
    (("robust_b_before", "robust_b_after"), 40, rf"method B: error {NUM} -> {NUM}"),
    (("robust_c_before", "robust_c_after"), 40, rf"method C: error {NUM} -> {NUM}"),
    (("d_dark_h", "d_dark_b"), 43, rf"\|H\|alpha,0>\| = {NUM}, \|b\|alpha,0>\| = {NUM}"),
    ("d_err_lossless", 43, rf"method D with kappa_a = 0: error {NUM}"),
    (("d_err", "d_err_circuit"), 43, rf"method D: error {NUM} -> {NUM} with circuit terms"),
    (("d_vs_c", "d_vs_c_circuit"), 43, rf"method D vs method C: {NUM}x lower error, {NUM}x lower"),
]

# Simulation settings: what we chose, not what we measured. Each points at
# the line in the source that sets it.
INPUTS = {
    "alpha": (2.1, "catqubit/params.py: TARGET_ALPHA = 2.1"),
    "g2_mhz": (0.763, "catqubit/params.py: G2_MHZ = 0.763 (Réglade et al., Table S2)"),
    "kappa_b_mhz": (2.6, "catqubit/params.py: KAPPA_B_MHZ = 2.6 (Réglade et al., Table S2)"),
    "kappa_a_mhz": (9.3e-3, "catqubit/params.py: KAPPA_A_MHZ = 9.3e-3 (Réglade et al., Table S2)"),
    "a0_delta_mhz": (1.0, "tools/build_notebook.py: DELTA = to_angular(1.0)"),
    "a0_kappa_mhz": (1.0, "tools/build_notebook.py: KAPPA = to_angular(1.0)"),
    "a0_alpha0": (2.0, "tools/build_notebook.py: dq.coherent(N, 2.0)"),
    "a0_n0": (4.0, "|alpha_0|^2 with alpha_0 = 2"),
    "b0_t_us": (3.0, "tools/build_notebook.py: t_b = jnp.linspace(0.0, 3.0, 301)"),
    "c0_t_us": (2.0, "tools/build_notebook.py: t_c = jnp.linspace(0.0, 2.0, 2001)"),
    "sim_t_us": (3.0, "tools/build_notebook.py: t_1 = jnp.linspace(0.0, 3.0, 301)"),
    "parity_t_us": (6.0, "tools/build_notebook.py: t_3 = jnp.linspace(0.0, 6.0, 121)"),
    "rtol": (1e-3, "catqubit convergence criterion, printed as 'rtol 0.001' in cell 11"),
}


def stdout_by_cell(nb: dict) -> dict[int, list[str]]:
    """Map cell index -> printed lines; refuse cells that never ran or errored."""
    lines: dict[int, list[str]] = {}
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        for out in cell.get("outputs", []):
            if out.get("output_type") == "error":
                raise RuntimeError(f"cell {i} has an error output; re-execute the notebook")
            if out.get("output_type") == "stream":
                lines.setdefault(i, []).extend("".join(out["text"]).splitlines())
        if lines.get(i) and cell.get("execution_count") is None:
            raise RuntimeError(f"cell {i} has output but no execution count")
    return lines


def claim(value: float, cell: int, line: str, text: str) -> dict:
    return {"value": value, "text": text, "cell": cell, "line": line}


def match_one(lines: list[str], pattern: str, cell: int) -> tuple[re.Match, str]:
    hits = [(m, ln) for ln in lines if (m := re.search(pattern, ln))]
    if len(hits) != 1:
        raise RuntimeError(f"cell {cell}: pattern {pattern!r} matched {len(hits)} lines")
    return hits[0]


def parse_rows(lines: list[str], cell: int, pattern: str, names: tuple[str, ...], prefix: str) -> dict:
    """Rows like 'beta = 0.25: F = ..., basin = ...' -> prefix.<row>.<name>."""
    out = {}
    for ln in lines:
        m = re.search(pattern, ln)
        if not m:
            continue
        row = m.group(1)
        for name, txt in zip(names, m.groups()[1:], strict=True):
            out[f"{prefix}.{row}.{name}"] = claim(float(txt), cell, ln.strip(), txt)
    if not out:
        raise RuntimeError(f"cell {cell}: no rows for {prefix}")
    return out


def parse_sweeps(lines: list[str]) -> dict:
    """Cell 11 truncation tables (N_a sweep then N_b sweep) and their verdicts."""
    out, table = {}, None
    names = ("steady", "peak", "top2", "topbuf")
    for ln in lines:
        if ln.strip().startswith("N_a |"):
            table = "na"
        elif ln.strip().startswith("N_b |"):
            table = "nb"
        elif m := re.match(rf"\s*(\d+) \| {NUM}\s+\| {NUM}\s+\| {NUM}\s+\| {NUM}", ln):
            for name, txt in zip(names, m.groups()[1:], strict=True):
                out[f"sweep.{table}.{m.group(1)}.{name}"] = claim(float(txt), 11, ln.strip(), txt)
        elif m := re.search(r"(steady|peak): converged \(rtol 0.001\) from N_(a|b) = (\d+)", ln):
            key = f"n{m.group(2)}_{m.group(1)}_from"
            out[key] = claim(float(m.group(3)), 11, ln.strip(), m.group(3))
    return out


def extract(lines: dict[int, list[str]]) -> dict:
    claims = {}
    for names, cell, pattern in SPECS:
        names = (names,) if isinstance(names, str) else names
        m, ln = match_one(lines[cell], pattern, cell)
        for name, txt in zip(names, m.groups(), strict=True):
            claims[name] = claim(float(txt), cell, ln.strip(), txt)
    claims |= parse_sweeps(lines[11])
    claims |= parse_rows(lines[7], 7, rf"kappa2 x {NUM}\s*: <n>\(2 us\) = {NUM}, t90 = \s*{NUM} ns, "
                         rf"t90 \* kappa2 = {NUM}\s+\(lossless: {NUM}\)",
                         ("n2us", "t90_ns", "t90k2", "t90k2_lossless"), "c0")
    claims |= parse_rows(lines[22], 22, rf"t = {NUM} us: <P> = {NUM}, F\(C\+\) = {NUM}, F\(\+alpha\) = {NUM}, "
                         rf"purity = {NUM}, coherence = {NUM}",
                         ("parity", "fid_cat", "fid_plus", "purity", "coherence"), "cat_t")
    claims |= parse_rows(lines[26], 26, rf"^beta = {NUM}: F\(\+alpha\) = {NUM}, basin model = {NUM}, Re<a> = {NUM}",
                         ("fid", "basin", "re_a"), "a_beta")
    claims |= parse_rows(lines[26], 26, rf"^phase error\s+{NUM} deg: F\(\+alpha\) = {NUM}, basin model = {NUM}",
                         ("fid", "basin"), "a_phase")
    claims |= parse_rows(lines[34], 34, rf"phase error\s+{NUM} deg: error A = {NUM}, error B = {NUM}",
                         ("err_a", "err_b"), "phase_err")
    claims |= parse_rows(lines[36], 36, rf"T =\s+(\d+ ns, A = [\d.]+): beta_eff = {NUM}, model F = {NUM}, "
                         rf"simulated F = {NUM}", ("beta_eff", "model", "sim"), "beff")
    claims |= parse_rows(lines[38], 38, rf"ramp {NUM} us: push eps = {NUM} rad/us = {NUM} g2 \({NUM} MHz\) "
                         rf"-> F\(\+alpha\) = {NUM}, error {NUM}",
                         ("push", "push_g2", "push_mhz", "fid", "err"), "c_ramp")
    claims |= parse_rows(lines[38], 38, rf"method C, phase error\s+{NUM} deg: error = {NUM}", ("err",), "c_phase")
    claims |= parse_pulse_grid(lines[28])
    claims |= parse_rows(lines[43], 43, rf"method D sweep: beta = {NUM}: error {NUM}, Re<a> = {NUM}",
                         ("err", "re_a"), "d_beta")
    claims |= parse_rows(lines[43], 43, rf"method D amplitude {NUM} alpha: error {NUM}", ("err",), "d_amp")
    claims |= parse_rows(lines[43], 43, rf"method D phase error\s+{NUM} deg: error {NUM}", ("err",), "d_phase")
    return claims


def parse_pulse_grid(lines: list[str]) -> dict:
    """Cell 28: F(+alpha) on a (duration T) x (pulse area) grid."""
    out = {}
    for ln in lines:
        m = re.search(r"T =\s+(\d+) ns \| F\(\+alpha\) for areas \[([^\]]+)\]: ([\d. ]+)\(drive up to (\d+) MHz\)", ln)
        if not m:
            continue
        areas = [a.strip() for a in m.group(2).split(",")]
        for area, txt in zip(areas, m.group(3).split(), strict=True):
            out[f"b_grid.{m.group(1)}.{area}"] = claim(float(txt), 28, ln.strip(), txt)
        out[f"b_grid.{m.group(1)}.max_mhz"] = claim(float(m.group(4)), 28, ln.strip(), m.group(4))
    return out


def derived(c: dict) -> dict:
    """Arithmetic on extracted claims; the formula is kept for the audit trail."""
    v = {k: x["value"] for k, x in c.items()}
    alpha2 = INPUTS["alpha"][0] ** 2
    rules = {
        "alpha2": ("alpha^2 with alpha = 2.1 (input)", alpha2),
        "peak_over_alpha2_pct": ("100 * (peak_n / alpha2 - 1)", 100 * (v["peak_n"] / alpha2 - 1)),
        "gamma_agree_pct": ("100 * (gamma_pred / coh_decay - 1)", 100 * (v["gamma_pred"] / v["coh_decay"] - 1)),
        "n_shift_agree_pct": ("100 * (n_shift / n_shift_mf - 1)", 100 * (v["n_shift"] / v["n_shift_mf"] - 1)),
        "b0_p01_init_formula": ("e^-4 (1 + 4) for |alpha_0|^2 = 4", math.exp(-4) * 5),
        "c_err_best": ("min(c_ramp.*.err)", min(v[k] for k in v if k.startswith("c_ramp.") and k.endswith(".err"))),
        "c_vs_a_ratio": ("phase_err.0.0.err_a / c_phase.0.0.err", v["phase_err.0.0.err_a"] / v["c_phase.0.0.err"]),
        "b_grid_overdrive_drop": ("b_grid.50.2.5 - b_grid.50.4.0", v["b_grid.50.2.5"] - v["b_grid.50.4.0"]),
        "b_drive_over_g2": ("b_mhz / g2_mhz (input)", v["b_mhz"] / INPUTS["g2_mhz"][0]),
        "b_over_c_drive": ("b_mhz / c_ramp.1.00.push_mhz", v["b_mhz"] / v["c_ramp.1.00.push_mhz"]),
    }
    return {k: {"value": val, "formula": f} for k, (f, val) in rules.items()}


def main() -> None:
    raw = NOTEBOOK.read_bytes()
    claims = extract(stdout_by_cell(json.loads(raw)))
    payload = {
        "source": {
            "notebook": NOTEBOOK.name,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "builder": "tools/build_notebook.py",
            "libraries": {lib: importlib.metadata.version(lib) for lib in LIBRARIES},
            "note": "claims = printed outputs of executed cells; inputs = chosen settings; derived = arithmetic",
        },
        "claims": claims,
        "derived": derived(claims),
        "inputs": {k: {"value": val, "source": src} for k, (val, src) in INPUTS.items()},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n")
    print(f"{len(claims)} claims, {len(payload['derived'])} derived, {len(INPUTS)} inputs -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
