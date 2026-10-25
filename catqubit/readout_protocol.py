"""Adaptive transmon-free Z readout of a dissipative cat: record surrogate, filters, SPRT.

Physics (Réglade et al. 2024, Sec. S8E; Gautier et al. 2023, Eq. 15b). A memory drive
eps_Z (a + a^dag) displaces the buffer to <b>_ss = -s eps_Z / (2 alpha g2), with s = +1 for
|+alpha> and s = -1 for |-alpha>. The buffer reaches this value after an underdamped transient:
linearizing H_2ph about the well gives  b'' + (kappa_b/2) b' + nu^2 b = nu^2 b_ss  with natural
frequency nu = 2 alpha g2 and envelope rate kappa_b/4.

Record (homodyne of the buffer output, efficiency eta):
    y_k = mu * X_k + sqrt(dt) * xi_k,   X_k = int_{t_k}^{t_k+dt} x(t) dt,
with mu = -sqrt(eta) 2 sqrt(kappa_b) eps_Z / (2 alpha g2) (signed, per sqrt(us)) and x(t) the
linear response of the buffer to the hidden telegraph codeword s(t). x is normalized so that
x -> s in steady state. The codeword flips at Gamma_X / 2 each way, where Gamma_X is the decay rate
of Re<a> (the factor-2 convention of catqubit.rates). A flip inverts the mean with the same
ringing kernel. Flips happen on the step grid (at step boundaries, from step 1 on).
Heterodyne detection halves eta.

The measurement rate mu^2 is the power signal-to-noise ratio per unit time r of D'Anjou et al.
2016 (Eq. 1). With eta = 1 it equals 4 eps_Z^2 / (kappa2 alpha^2).

Every array of records or filter traces is time-major, with shape (n_steps, n_shots).
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import erfc, erfcinv, expit

from catqubit.params import CatParams

HETERODYNE_EFFICIENCY_FACTOR = 0.5
# exp() arguments are clipped here so that extreme records cannot overflow the filter.
MAX_EXP_ARGUMENT = 700.0
DEFAULT_QUANTILES = (0.5, 0.9, 0.99, 0.999)


@dataclass(frozen=True)
class RecordModel:
    """Classical surrogate of the readout record.

    mu: signed record amplitude (us^-1/2); the mean record of codeword s is mu s dt when settled.
    flip_rate: Gamma_X in rad/us (decay rate of Re<a>); each one-way jump happens at Gamma_X / 2.
    dt: record time step (us).
    nu, damping: natural angular frequency and envelope rate (rad/us) of the buffer response.
        Leave both as None for an instantaneous response (no ringing).
    """

    mu: float
    flip_rate: float
    dt: float
    nu: float | None = None
    damping: float | None = None

    def __post_init__(self) -> None:
        if self.dt <= 0:
            raise ValueError(f"dt must be positive, got {self.dt}")
        if self.flip_rate < 0:
            raise ValueError(f"flip_rate must be non-negative, got {self.flip_rate}")
        if (self.nu is None) != (self.damping is None):
            raise ValueError("nu and damping must be given together")
        if self.nu is not None and not 0 < self.damping < self.nu:
            raise ValueError(
                f"need an underdamped response 0 < damping < nu, got {self.damping}, {self.nu}"
            )

    @classmethod
    def from_physics(
        cls,
        params: CatParams,
        eps_z: float,
        eta: float,
        flip_rate: float,
        dt: float,
        heterodyne: bool = False,
    ) -> "RecordModel":
        """Surrogate from the linear buffer response; heterodyne detection halves eta."""
        if not 0 < eta <= 1:
            raise ValueError(f"eta must lie in (0, 1], got {eta}")
        eta_eff = eta * (HETERODYNE_EFFICIENCY_FACTOR if heterodyne else 1.0)
        b_ss = eps_z / (2 * params.alpha * params.g2)
        return cls(
            mu=-math.sqrt(eta_eff) * 2 * math.sqrt(params.kappa_b) * b_ss,
            flip_rate=flip_rate,
            dt=dt,
            nu=2 * params.alpha * params.g2,
            damping=params.kappa_b / 4,
        )

    @property
    def measurement_rate(self) -> float:
        """mu^2 (1/us): D'Anjou's power SNR per unit time r."""
        return self.mu**2

    @property
    def flip_step_probability(self) -> float:
        """Probability that the codeword differs after one step: (1 - exp(-Gamma_X dt)) / 2."""
        return -math.expm1(-self.flip_rate * self.dt) / 2


def illustrative_flip_rate(params: CatParams, eps_z: float, gamma_x0: float) -> float:
    """ILLUSTRATIVE Gamma_X(eps_Z): gamma_x0 plus a linear-order drive term.

    The drive term 2 x (2 eps_Z^2 / (kappa2 alpha^2)) exp(-2 alpha^2) is the brief's estimate,
    not a computed rate. It equals mu^2(eta=1) exp(-2 alpha^2): drive-induced flips grow in
    proportion to the measurement rate.
    """
    drive = 2 * (2 * eps_z**2 / (params.kappa2 * params.alpha**2)) * math.exp(-2 * params.alpha**2)
    return gamma_x0 + drive


# --------------------------------------------------------------------- ringing kernel


def _integrated_step_response(model: RecordModel, t: np.ndarray) -> np.ndarray:
    """int_0^t g, with g(t) = 1 - e^{-gt}(cos wt + (g/w) sin wt) the unit step response."""
    nu, gam = model.nu, model.damping
    w = math.sqrt(nu**2 - gam**2)
    decay, cos, sin = np.exp(-gam * t), np.cos(w * t), np.sin(w * t)
    int_cos = (decay * (w * sin - gam * cos) + gam) / nu**2
    int_sin = (w - decay * (gam * sin + w * cos)) / nu**2
    return t - int_cos - gam / w * int_sin


def step_response_integrals(model: RecordModel, n_steps: int) -> np.ndarray:
    """G_k = int over step k of the unit step response started at t = 0 (G_k -> dt when settled)."""
    if model.nu is None:
        return np.full(n_steps, model.dt)
    edges = np.arange(n_steps + 1) * model.dt
    return np.diff(_integrated_step_response(model, edges))


# --------------------------------------------------------------------- record generator


@dataclass(frozen=True)
class FlipEvents:
    """Codeword flips: shot[i] flips at the start of step[i]. Sorted by (shot, step)."""

    shot: np.ndarray
    step: np.ndarray

    @classmethod
    def empty(cls) -> "FlipEvents":
        return cls(shot=np.zeros(0, dtype=int), step=np.zeros(0, dtype=int))


@dataclass(frozen=True)
class RecordBatch:
    """Records y (n_steps, n_shots), initial codewords s0 and the hidden flips."""

    y: np.ndarray
    s0: np.ndarray
    flips: FlipEvents


def sample_flips(
    model: RecordModel, n_shots: int, n_steps: int, rng: np.random.Generator
) -> FlipEvents:
    """Telegraph flips on the step grid (steps 1..n_steps-1), via geometric waiting times."""
    p = model.flip_step_probability
    if p == 0:
        return FlipEvents.empty()
    position = np.zeros(n_shots, dtype=np.int64)
    active = np.arange(n_shots)
    shots, steps = [], []
    while active.size:
        position[active] += rng.geometric(p, size=active.size)
        active = active[position[active] < n_steps]
        shots.append(active)
        steps.append(position[active].copy())
    shot, step = np.concatenate(shots), np.concatenate(steps)
    order = np.lexsort((step, shot))
    return FlipEvents(shot=shot[order], step=step[order])


def _flip_signs(s0: np.ndarray, flips: FlipEvents) -> np.ndarray:
    """Codeword just before each flip: s0 times (-1)^(number of earlier flips of that shot)."""
    if flips.shot.size == 0:
        return np.zeros(0)
    first = np.r_[True, flips.shot[1:] != flips.shot[:-1]]
    group_start = np.maximum.accumulate(np.where(first, np.arange(flips.shot.size), 0))
    rank = np.arange(flips.shot.size) - group_start
    return s0[flips.shot] * (-1.0) ** rank


def mean_record(model: RecordModel, s0: np.ndarray, flips: FlipEvents, n_steps: int) -> np.ndarray:
    """Noise-free record mu X_k, shape (n_steps, n_shots), of the codeword paths."""
    g_steps = step_response_integrals(model, n_steps)
    mean = model.mu * g_steps[:, None] * np.asarray(s0, dtype=float)[None, :]
    for shot, step, before in zip(flips.shot, flips.step, _flip_signs(s0, flips), strict=True):
        mean[step:, shot] -= 2 * before * model.mu * g_steps[: n_steps - step]
    return mean


def state_at(s0: np.ndarray, flips: FlipEvents, steps: np.ndarray) -> np.ndarray:
    """Codeword of each shot at its own step (flips at step <= steps[shot] included)."""
    counted = flips.step <= steps[flips.shot]
    n_flips = np.bincount(flips.shot[counted], minlength=s0.size)
    return s0 * (-1.0) ** n_flips


def simulate_records(
    model: RecordModel,
    n_shots: int,
    n_steps: int,
    rng: np.random.Generator,
    s0: float | None = None,
) -> RecordBatch:
    """Telegraph-plus-Gaussian records. s0 = None draws equiprobable codewords."""
    if s0 is None:
        codewords = rng.choice(np.array([-1.0, 1.0]), size=n_shots)
    else:
        codewords = np.full(n_shots, float(s0))
    flips = sample_flips(model, n_shots, n_steps, rng)
    y = mean_record(model, codewords, flips, n_steps)
    y += math.sqrt(model.dt) * rng.standard_normal((n_steps, n_shots))
    return RecordBatch(y=y, s0=codewords, flips=flips)


# --------------------------------------------------------------------- linear filters


def boxcar_statistic(model: RecordModel, y: np.ndarray) -> np.ndarray:
    """Running boxcar sum, signed so that a positive value favours s0 = +1."""
    return math.copysign(1.0, model.mu) * np.cumsum(y, axis=0)


def matched_filter_weights(model: RecordModel, n_steps: int) -> np.ndarray:
    """Mean-difference weights E[mu X_k | s0 = +1] / mu (Ryan et al. 2015), times sign(mu).

    E[s(t) | s0] = s0 exp(-Gamma_X t); the linear response convolves its increments with G.
    """
    g_steps = step_response_integrals(model, n_steps)
    mean_state = (1 - 2 * model.flip_step_probability) ** np.arange(n_steps)
    increments = np.diff(mean_state, prepend=0.0)
    return math.copysign(1.0, model.mu) * np.convolve(increments, g_steps)[:n_steps]


def matched_statistic(model: RecordModel, y: np.ndarray) -> np.ndarray:
    """Running matched-filter sum; positive favours s0 = +1."""
    return np.cumsum(matched_filter_weights(model, y.shape[0])[:, None] * y, axis=0)


def fixed_time_errors(statistic: np.ndarray, s0: np.ndarray, check_steps: np.ndarray) -> np.ndarray:
    """Fraction of shots whose sign(statistic) at each check step differs from s0 (ties -> +1)."""
    decisions = np.where(statistic[check_steps] >= 0, 1.0, -1.0)
    return np.mean(decisions != s0[None, :], axis=1)


# --------------------------------------------------------------------- HMM forward filter


@dataclass(frozen=True)
class HmmTrace:
    """Online filter output: log P(s0=+|y_1:k)/P(s0=-|y_1:k) and P(s_k != s0 | y_1:k).

    branch_gap = P(s_k=+ | s0=+, y) - P(s_k=+ | s0=-, y). Future records inform only s_k, so once
    both s0 branches agree on s_k (gap -> 0) the s0 log-odds are frozen: waiting cannot help.
    """

    log_odds: np.ndarray
    flip_prob: np.ndarray
    branch_gap: np.ndarray | None = None


def hmm_filter(model: RecordModel, y: np.ndarray) -> HmmTrace:
    """Optimal forward (Wonham) filter over the 4 hidden states (s0, s_k), equal priors.

    s0 never changes, so the filter runs two 2-state chains, one per s0, started at s_k = s0.
    Emission: the mean record of s_k is mu s_k G_k. This uses the t = 0 step response for the
    initial transient and treats flips as settling instantly (the true record rings for
    about 1 us after a flip). This is the only model mismatch with the generator.
    """
    n_steps, n_shots = y.shape
    p = model.flip_step_probability
    gain = model.mu * step_response_integrals(model, n_steps) / model.dt
    log_odds = np.empty((n_steps, n_shots))
    flip_prob = np.empty((n_steps, n_shots), dtype=np.float32)
    branch_gap = np.empty((n_steps, n_shots), dtype=np.float32)
    plus_given_plus = np.ones(n_shots)  # P(s_k = + | s0 = +, y)
    plus_given_minus = np.zeros(n_shots)  # P(s_k = + | s0 = -, y)
    llr = np.zeros(n_shots)
    for k in range(n_steps):
        if k > 0:
            plus_given_plus = p + (1 - 2 * p) * plus_given_plus
            plus_given_minus = p + (1 - 2 * p) * plus_given_minus
        ratio = np.exp(np.clip(-2 * gain[k] * y[k], -MAX_EXP_ARGUMENT, MAX_EXP_ARGUMENT))
        norm_plus = plus_given_plus + (1 - plus_given_plus) * ratio
        norm_minus = plus_given_minus + (1 - plus_given_minus) * ratio
        llr += np.log(norm_plus / norm_minus)
        plus_given_plus = plus_given_plus / norm_plus
        plus_given_minus = plus_given_minus / norm_minus
        p_s0_plus = expit(llr)
        log_odds[k] = llr
        flip_prob[k] = p_s0_plus * (1 - plus_given_plus) + (1 - p_s0_plus) * plus_given_minus
        branch_gap[k] = plus_given_plus - plus_given_minus
    return HmmTrace(log_odds=log_odds, flip_prob=flip_prob, branch_gap=branch_gap)


# --------------------------------------------------------------------- SPRT and erasure


@dataclass(frozen=True)
class SprtResult:
    """Per shot: decided codeword, stopping step (0-based) and erasure flag."""

    decision: np.ndarray
    stop_step: np.ndarray
    erased: np.ndarray


def _check_sprt_inputs(trace: HmmTrace, thresholds: np.ndarray, max_steps: int) -> None:
    if np.any(thresholds <= 0):
        raise ValueError(f"thresholds must be positive, got {thresholds}")
    if not 1 <= max_steps <= trace.log_odds.shape[0]:
        raise ValueError(f"max_steps must lie in [1, {trace.log_odds.shape[0]}], got {max_steps}")


def _first_frozen_step(trace: HmmTrace, max_steps: int, freeze_gap: float) -> np.ndarray:
    if trace.branch_gap is None:
        raise ValueError("freeze_gap needs a trace with branch_gap (from hmm_filter)")
    frozen = np.abs(trace.branch_gap[:max_steps]) <= freeze_gap
    return np.where(frozen.any(axis=0), np.argmax(frozen, axis=0), max_steps - 1)


def sprt_stop_steps(
    trace: HmmTrace,
    thresholds: np.ndarray,
    max_steps: int,
    freeze_gap: float | None = None,
) -> np.ndarray:
    """Stopping steps (n_thresholds, n_shots): first k with |log-odds| >= A, else max_steps - 1.

    With freeze_gap, also stop once |branch_gap| <= freeze_gap (the s0 evidence is frozen).
    One pass over the running maximum of |log-odds| serves every threshold.
    """
    thresholds = np.asarray(thresholds, dtype=float)
    _check_sprt_inputs(trace, thresholds, max_steps)
    running_max = np.maximum.accumulate(np.abs(trace.log_odds[:max_steps]), axis=0).T.copy()
    stops = np.empty((thresholds.size, running_max.shape[0]), dtype=np.int64)
    for shot, column in enumerate(running_max):
        stops[:, shot] = np.searchsorted(column, thresholds, side="left")
    stops = np.minimum(stops, max_steps - 1)
    if freeze_gap is not None:
        stops = np.minimum(stops, _first_frozen_step(trace, max_steps, freeze_gap)[None, :])
    return stops


def sprt_outcome(
    trace: HmmTrace,
    stop_step: np.ndarray,
    threshold: float,
    theta: float | None = None,
    erase_undecided: bool = False,
) -> SprtResult:
    """Decision sign(log-odds) at the stop. Erasure (optional): stopped undecided, |log-odds| < A
    (at the cap or frozen), or P(s_k != s0 | y) > theta at the stop."""
    shots = np.arange(stop_step.size)
    final = trace.log_odds[stop_step, shots]
    erased = np.zeros(stop_step.size, dtype=bool)
    if erase_undecided:
        erased |= np.abs(final) < threshold
    if theta is not None:
        erased |= trace.flip_prob[stop_step, shots] > theta
    return SprtResult(decision=np.where(final >= 0, 1.0, -1.0), stop_step=stop_step, erased=erased)


def sprt(
    trace: HmmTrace,
    threshold: float,
    max_steps: int,
    theta: float | None = None,
    erase_undecided: bool = False,
    freeze_gap: float | None = None,
) -> SprtResult:
    """Wald's SPRT on the filter log-odds: stop at |log-odds| >= A, at the hard cap, or (with
    freeze_gap) once the s0 evidence is frozen."""
    stop = sprt_stop_steps(trace, np.array([threshold]), max_steps, freeze_gap)[0]
    return sprt_outcome(trace, stop, threshold, theta=theta, erase_undecided=erase_undecided)


@dataclass(frozen=True)
class ProtocolStats:
    """Fractions of all shots. error: kept and wrong about s0. hidden: kept, right about s0, but
    the codeword had flipped by the stop (an unheralded bit flip during readout)."""

    p_error: float
    p_erasure: float
    p_hidden: float
    mean_time: float
    time_quantiles: dict[float, float]
    n_shots: int


def stats_from_arrays(
    decision: np.ndarray,
    erased: np.ndarray,
    stop_step: np.ndarray,
    s0: np.ndarray,
    s_stop: np.ndarray,
    dt: float,
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
) -> ProtocolStats:
    """Aggregate per-shot outcomes (arrays may be concatenated over chunks)."""
    kept = ~erased
    times = (stop_step + 1) * dt
    return ProtocolStats(
        p_error=float(np.mean(kept & (decision != s0))),
        p_erasure=float(np.mean(erased)),
        p_hidden=float(np.mean(kept & (decision == s0) & (s_stop != s0))),
        mean_time=float(times.mean()),
        time_quantiles={q: float(np.quantile(times, q)) for q in quantiles},
        n_shots=int(s0.size),
    )


def protocol_stats(result: SprtResult, batch: RecordBatch, model: RecordModel) -> ProtocolStats:
    """Error, erasure and hidden-flip fractions plus stopping-time statistics of one batch."""
    s_stop = state_at(batch.s0, batch.flips, result.stop_step)
    return stats_from_arrays(
        result.decision, result.erased, result.stop_step, batch.s0, s_stop, model.dt
    )


# --------------------------------------------------------------------- Gaussian latching theory


def gaussian_fixed_error(rate: float, t: float) -> float:
    """Fixed-window error (1/2) erfc(sqrt(r t / 2)) (D'Anjou et al. 2016, Eq. 2)."""
    return float(0.5 * erfc(math.sqrt(rate * t / 2)))


def gaussian_fixed_time(eps: float, rate: float) -> float:
    """Window needed for error eps: inverse of gaussian_fixed_error."""
    return float(2 * erfcinv(2 * eps) ** 2 / rate)


def sprt_threshold(eps: float) -> float:
    """Wald threshold A = ln((1 - eps) / eps)."""
    return math.log((1 - eps) / eps)


def sprt_error(threshold: float) -> float:
    """SPRT error 1 / (1 + e^A) (D'Anjou et al. 2016, Eq. 3, no cap, continuous time)."""
    return 1 / (1 + math.exp(threshold))


def sprt_mean_time(threshold: float, rate: float) -> float:
    """SPRT mean time (A / 2r) tanh(A / 2) (D'Anjou et al. 2016, Eq. 3)."""
    return threshold / (2 * rate) * math.tanh(threshold / 2)


def gaussian_speedup(eps: float) -> float:
    """Fixed window / SPRT mean time at equal error; tends to 4 as eps -> 0."""
    return gaussian_fixed_time(eps, 1.0) / sprt_mean_time(sprt_threshold(eps), 1.0)
