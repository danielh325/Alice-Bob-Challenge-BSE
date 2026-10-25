# `catqubit.readout_stats`: public API (Task 2.3, for Task 2.4)

This module computes the exact statistics of the integrated homodyne readout record from the
tilted Liouvillian (Landi et al., PRX Quantum 5, 020201 (2024), arXiv:2303.04270; Eqs. 124 and
142–144 checked in the full text). Units are µs and rad/µs. Vectors use column stacking, as in
`catqubit.liouvillian`, so Tr[X ρ] = `vec(X.T) @ vec(ρ)`.

## Conventions

- **Record.** The record is dY = √η ⟨c e^{−iφ} + c† e^{iφ}⟩ dt + dW, with c = √κ_b b and
  white noise E[dW²] = dt. The filtered charge is Q(T) = ∫₀ᵀ f(t) dY(t). With φ = 0, Q measures
  Re⟨b⟩, which is the quadrature that carries the codeword. The mean slope is therefore
  2√(ηκ_b)·Re⟨b⟩.
- **Generating function.** It is computed as
  E[e^{sQ(T)}] = e^{s² F2(T)/2} · Tr[𝒯 exp(∫(L + s f(t) J) dt) ρ0], with F2 = ∫₀ᵀ f² and
  J ρ = √η (c e^{−iφ} ρ + ρ c† e^{iφ}). Setting s = ik gives the characteristic function.
- **Codeword sign.** "+" is |+α⟩⊗|0⟩. With ε_Z > 0 and φ = 0, its record mean is **negative**,
  because ⟨b⟩+ ≈ −ε_Z/(2αg2).
- **Errors.** All errors assume equal priors. `error_bayes` is ½∫min(P+, P−), the
  likelihood-ratio optimum. `error_threshold` is the error of the single best cut on Q.
- **Γ_X.** Γ_X is the decay rate of Re⟨a⟩, which is the sum of the two one-way rates. The
  one-way flip rate is Γ_X/2.

## Building blocks

```python
Homodyne(eta: float = 1.0, phi: float = 0.0)          # frozen; requires 0 < eta <= 1 (heterodyne: eta/2)
IDEAL_HOMODYNE = Homodyne()
TiltedGenerator(liouvillian, tilt, trace_row)          # frozen; .at(s) -> L + s J (csr)
tilt_superoperator(c, detector) -> csr                 # the J above
homodyne_generator(big_l, c, detector=IDEAL_HOMODYNE) -> TiltedGenerator   # any Lindbladian
cat_readout_generator(params, trunc, eps_z, detector=IDEAL_HOMODYNE) -> TiltedGenerator
    # two_mode_liouvillian(params, trunc, eps_z) with c = sqrt(kappa_b) b
codeword_state(params, trunc, sign) -> vec(|sign*alpha><..| (x) |0><0|)   # complex, normalized
PiecewiseFilter(edges: tuple, weights: tuple)          # f = weights[i] on [edges[i], edges[i+1]);
                                                       # edges[0] = 0; the last weight holds forever
    .weight(t), .squared_integral(times) -> F2(T), .last_edge
BOXCAR = PiecewiseFilter((0.0,), (1.0,))
SlowModes(t_switch: float, n_modes: int = 6, sigma: complex = SLOW_SHIFT, from_start: bool = False)
```

## Long windows: `SlowModes`

Up to `t_switch`, the tilted state is propagated exactly with `expm_multiply`. After it, the
state is continued with the `n_modes` eigenmodes of L + s·f_last·J closest to `sigma`. These
are the two well populations plus slow coherences, computed with one sparse LU per s
(`catqubit.readout_dynamics.liouvillian_modes`). The dropped fast part is e^{−(κ_b/4) t_switch}
or smaller, and it is reported as `switch_error`.

- `t_switch` must not precede the last filter edge, so the filter must be constant after it.
- `from_start=True` is allowed only for constant filters. It projects ρ0 itself onto the slow
  modes and skips the propagation to `t_switch`. This is much cheaper for long windows, but
  `switch_error` is then `nan`, because the dropped part is not measured. Validate it once
  against `from_start=False`, as `studies/task23_readout.py` does.

## Statistics

```python
tilted_trace(gen, rho0s, s, times, filt=BOXCAR, slow=None) -> (n_rho, nT) complex
    # Tr[...] WITHOUT the white-noise factor
generating_function(gen, rho0s, s_values, times, filt=BOXCAR, slow=None, workers=1)
    -> (n_rho, ns, nT) complex      # E[e^{sQ}] including e^{s^2 F2/2}; workers>1: spawn pool over s
record_cumulants(gen, rho0, times, filt=BOXCAR) -> (mean, var)
    # exact (block generator, no finite differences); 3x dimension
fd_cumulants(gen, rho0s, times, filt=BOXCAR, slow=None, workers=1, step=1e-3) -> (mean, var)  # (n_rho, nT)
FourierGrid.covering(half_width, f2_min, floor=1e-16) -> FourierGrid(dk, k_max); .ks, .window
readout_error(gen, rho_plus, rho_minus, times, filt=BOXCAR, slow=None, workers=1) -> ReadoutError
chernoff_bound(gen, rho_plus, rho_minus, times, thresholds, s_values, filt=BOXCAR, slow=None, workers=1)
    -> (nT,)  # upper bound on the threshold error at the given thresholds
gaussian_error(mean_a, mean_b, var_a, var_b) -> 1/2 erfc(|dmu| / (sqrt2 (sig_a + sig_b)))
matched_filter_error(delta_b, times, kappa_b, eta) -> cumulative 1/2 erfc(sqrt((eta kappa_b/2) int |Delta b|^2 dt))
```

`ReadoutError` is a frozen dataclass with these fields:

- `times`;
- `mean` and `var`, each of shape (2, nT), with rows +α and −α;
- `error_bayes`, `error_threshold` and `thresholds`;
- `error_gauss`, the Gaussian estimate built from the exact moments;
- `distributions`, a tuple of `RecordDistribution(time, q, p_plus, p_minus)` on a uniform
  periodic Q grid;
- `switch_error`.

### How `readout_error` works

- It splits `times` at `slow.t_switch`. Each side gets its own Fourier grid.
- The window holds every mean ± 8σ, and k_max is set by the shortest time on that side.
- Errors come from the exact Fourier-series CDFs between the crossings of the two densities.
- **Precision floor.** Errors below about 1e-12 are not resolved. For those, use
  `chernoff_bound` with real s > 0. It returns ½[min_s e^{−sq}E_low e^{sQ} + min_s e^{sq}E_high e^{−sQ}].

## Cost (α = 2.1, `FockTruncation(26, 4, 7, 13.2)`, dimension 25,600)

- **Per s value**, `expm_multiply` costs about 2.8 CPU-s per 0.5 µs.
- **Slow modes:** one LU plus two Arnoldi runs per s value.
- **A full `readout_error` call** needs about 4 + (number of k values) s values.
- **Number of k values:** about 2π·k_max/window per side, typically 60–140. Use `workers`.
- **Optimizers** should prefer `generating_function` at a few s values, or `fd_cumulants`,
  inside loops over full distributions.
