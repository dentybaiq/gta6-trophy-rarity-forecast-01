"""gta6-trophy-rarity-forecast-01: earners-vs-players forecaster.

What moves an Exophase rate is two counts, not one: the players tracked
N(t) and, per trophy, the earners E_i(t), with rate_i = 100 * E_i / N.
Exophase refreshes a game's counts in jumps hours apart (the practice
cases straddle one: N 638 -> 3,868 between 04:00Z and 04:30Z on launch
night, every rate but the one at 100% falling because the newly tracked
players hold fewer trophies), so between refreshes the page is frozen
and a naive trend fit on the raw reads is a fit on duplicates.

This forecaster therefore:

  1. collapses consecutive identical reads to the distinct snapshots
     (the refresh points);
  2. fits a power law in log time to the players tracked,
     log N = a_N + b_N * ln(1 + hours since 2026-11-19T00:00Z),
     and reads it at the target, weighting recent snapshots more
     (the launch surge decelerates, so the recent growth rate is
     the honest one to extrapolate);
  3. fits, per trophy, a power law to the earners,
     log(E_i + 0.5) = alpha_i + beta_i * ln(1 + hours),
     shrinking each slope toward the cross-trophy median (empirical
     Bayes: a trophy's own slope is noisy on a handful of snapshots)
     and clamping slopes to >= 0 (earners never fall) and to a sane
     ceiling, then re-centring the intercept so the line passes
     through the data's centroid;
  4. forecasts rate_i = 100 * E_i(target) / N(target), clamped to
     [0.01, 99.99], and never above the last read's earners-implied
     ceiling of 99.99;
  5. sets each trophy's logit spread from a horizon-growing base term
     plus the trophy's own fit residual and the players fit residual,
     so trophies whose history is noisy or whose extrapolation is
     long get honestly wider quantiles.

The median is the model rate; q_tau = inv_logit(logit(m) + z_tau * s)
with z the standard normal quantiles at (0.05, 0.25, 0.5, 0.75, 0.95).
Everything is deterministic, standard library only, and runs in well
under the time cap.

How the grader runs it: `python3 -P -s /bot/main.py`
(PYTHONHASHSEED=0) in python:3.13-slim, standard library only: no
network, no pip packages. -P keeps this file's directory OFF sys.path,
so a module shipped beside main.py imports only after the sys.path
line below.

What an entry must do:
  - read every <id>.in in the input directory (the grader runs one case
    at a time, with only that case's file there);
  - write <id>.out to the output directory: {"trophies": [{"id": 1,
    "q": [q05, q25, q50, q75, q95]}, ...]}, one row for each of the
    case's trophies, each id once, no other key, each q five finite
    numbers from 0 to 100 (percent), non-decreasing; at most 65,536
    bytes.

Outside the grader's container, MAKEFUN_INPUT_DIR and
MAKEFUN_OUTPUT_DIR stand in for /input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model` finds a model.py shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Launch reference: the release day began 2026-11-19T00:00:00Z. Time is
# measured as x = ln(1 + hours since launch), the same clock the house
# trend baseline uses, so reads just before launch still have a
# defined (slightly negative) x.
LAUNCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# --- model constants (fixed in advance, never tuned on any data) ---
# Earners never fall, and a power-law slope above this would mean the
# earner count more than squaring every e-fold of log time: cap it.
BETA_MAX = 2.0
# The players-tracked count never falls, and its log-time slope is
# capped the same way.
BETA_N_MAX = 2.0
# The players forecast is capped at this multiple of the last read, so
# a steep early power law cannot explode the denominator.
N_GROWTH_CAP = 10.0
# Base logit spread per square-root day of horizon (the house
# persistence baseline uses 0.5; a model that explains the drift earns
# a narrower base, but not an aggressive one: a launch week's refresh
# timing alone is worth real uncertainty).
BASE_SIGMA = 0.35
# Bounds for the per-trophy residual contribution to the spread.
RESID_MIN = 0.05
RESID_MAX = 1.0
# How strongly a trophy's own slope is shrunk toward the cross-trophy
# median (empirical Bayes, in units of the slope's standard error).
SHRINK_TAU = 0.5
# Flat-history fallback: when the whole history is a single snapshot
# (Exophase had not refreshed yet), the fitted slopes are all zero,
# but a refresh is coming and the counts are growing. The launch-night
# refresh took N 638 -> 3,868 in under 5 hours and the tracked count
# kept climbing steeply through the first day, so the prior slopes
# below are the launch period's: players tracked grows fast in log
# time, earners about half as fast (the newly tracked players hold
# fewer trophies, which is why rates fall at a refresh). These apply
# only when there is a single distinct snapshot; a history with real
# refreshes in it is fitted, not prior-driven.
FLAT_B_N = 2.0
FLAT_BETA = 1.0
# Extra logit spread when the forecast rests on the flat-history prior
# rather than on fitted slopes: the prior's median is a guess, so the
# quantiles must be honestly wide.
FLAT_SPREAD = 0.5


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    return math.log(clamp(pct) / (100 - clamp(pct)))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return 100 / (1 + math.exp(-lg))
    e = math.exp(lg)
    return 100 * e / (1 + e)


def round4(x: float) -> float | int:
    """x rounded to 4 decimals, half up, as JavaScript's Math.round(x * 10000) / 10000 rounds it."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_launch(t: datetime) -> float:
    return (t - LAUNCH).total_seconds() / 3600.0


def log_time(t: datetime) -> float:
    """x = ln(1 + hours since launch). Defined even just before launch."""
    return math.log1p(max(-0.999, hours_since_launch(t)))


def wols(xs: list[float], ys: list[float], ws: list[float]) -> tuple[float, float, float]:
    """Weighted least squares y ~ a + b*x. Returns (a, b, residual std).

    With < 2 points, or a degenerate x, returns (weighted mean y, 0, 0).
    The residual std is the weighted root-mean-square residual.
    """
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0
    sw = sum(ws)
    if sw <= 0:
        return sum(ys) / n, 0.0, 0.0
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    if n < 2:
        return my, 0.0, 0.0
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 0:
        return my, 0.0, 0.0
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    a = my - b * mx
    resid = math.sqrt(sum(w * (y - (a + b * x)) ** 2 for w, x, y in zip(ws, xs, ys)) / sw)
    return a, b, resid


def wslope_se(xs: list[float], ws: list[float], resid: float) -> float:
    """Standard error of the weighted OLS slope."""
    n = len(xs)
    if n < 3:
        return float("inf")
    sw = sum(ws)
    if sw <= 0:
        return float("inf")
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 0:
        return float("inf")
    return resid / math.sqrt(sxx)


def median(vals: list[float]) -> float:
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def distinct_reads(history: list[dict]) -> list[dict]:
    """Collapse consecutive identical reads (same players and rates).

    Between Exophase refreshes the page is frozen, so the raw reads are
    mostly duplicates; the distinct snapshots are the refresh points.
    A frozen run is anchored at its LAST read: that is the most recent
    moment the snapshot was observed, so the extrapolation from it to
    the target is over the shortest honest distance.
    """
    out: list[dict] = []
    for h in history:
        if out:
            prev = out[-1]
            if h.get("players_tracked") == prev.get("players_tracked") and h.get("rates") == prev.get("rates"):
                # Same snapshot again: move its anchor to this later read.
                out[-1] = h
                continue
        out.append(h)
    return out


def earners_log(rates_row: list, n: float, idx: int) -> float:
    """log(E_i + 0.5) for trophy idx at a read with players n."""
    r = rates_row[idx] if idx < len(rates_row) and rates_row[idx] is not None else 0.0
    e = max(0.0, float(r)) * n / 100.0
    return math.log(e + 0.5)


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")

    trophies = case["trophies"]
    n_tr = len(trophies)
    horizon_days = case["horizon_minutes"] / 1440.0
    target_t = parse_time(case["target"])
    x_target = log_time(target_t)

    reads = distinct_reads(case["history"])
    xs = [log_time(parse_time(h["at"])) for h in reads]
    # Recency weights: the launch surge decelerates, so recent snapshots
    # say more about the growth rate at the target than launch-night
    # ones do. Exponential decay with a 24-hour half-life in clock time.
    t_hours = [hours_since_launch(parse_time(h["at"])) for h in reads]
    t_last_h = t_hours[-1] if t_hours else 0.0
    ws = [math.exp(-(t_last_h - t) / 24.0 * math.log(2)) for t in t_hours]
    # Players tracked at each distinct snapshot; fall back to the
    # previous value (then to 1) if a read lacks it.
    ns: list[float] = []
    last_n = 1.0
    for h in reads:
        n = h.get("players_tracked")
        if n is None or n <= 0:
            n = last_n
        else:
            last_n = float(n)
        ns.append(float(n))
    n_last = ns[-1]

    # --- players model: log N ~ a + b * x (recency-weighted) ---
    log_ns = [math.log(n) for n in ns]
    if len(reads) <= 1:
        # A single snapshot: Exophase had not refreshed yet, so the
        # fitted slope is zero by construction. Use the launch-period
        # prior (a refresh is coming; the counts are growing).
        b_n = FLAT_B_N
        resid_n = 0.0
        a_n = log_ns[0] - b_n * xs[0] if xs else 0.0
    else:
        a_n, b_n, resid_n = wols(xs, log_ns, ws)
        b_n = min(max(b_n, 0.0), BETA_N_MAX)
        # Re-centre the intercept on the clamped slope so the line still
        # passes through the data's (weighted) centroid.
        sw = sum(ws)
        mx = sum(w * x for w, x in zip(ws, xs)) / sw
        my = sum(w * y for w, y in zip(ws, log_ns)) / sw
        a_n = my - b_n * mx
    log_n_target = a_n + b_n * x_target
    n_target = math.exp(log_n_target)
    n_target = min(max(n_target, n_last), n_last * N_GROWTH_CAP)

    # --- earners model, per trophy (recency-weighted) ---
    betas: list[float] = []
    ses: list[float] = []
    fits: list[tuple[float, float, float, float]] = []  # (alpha, beta, resid, y_mean)
    for i in range(n_tr):
        ys = [earners_log(h["rates"], ns[j], i) for j, h in enumerate(reads)]
        if len(reads) <= 1:
            # Single snapshot: no slope is identifiable; use the
            # launch-period prior (earners grow, but slower than the
            # players tracked, so rates fall at a refresh).
            a_i = ys[0] - FLAT_BETA * xs[0] if xs else ys[0]
            b_i = FLAT_BETA
            resid_i = 0.0
            y_mean = ys[0]
        else:
            a_i, b_i, resid_i = wols(xs, ys, ws)
            sw = sum(ws)
            y_mean = sum(w * y for w, y in zip(ws, ys)) / sw if sw > 0 else (sum(ys) / len(ys) if ys else 0.0)
        se_i = wslope_se(xs, ws, resid_i) if len(reads) > 1 else float("inf")
        fits.append((a_i, b_i, resid_i, y_mean))
        betas.append(b_i)
        ses.append(se_i)

    beta_med = median(betas)
    # Cross-trophy spread of the slopes, for the empirical-Bayes prior.
    tau = max(SHRINK_TAU, 0.5 * (max(betas) - min(betas)) if betas else SHRINK_TAU)

    base_spread = BASE_SIGMA * math.sqrt(max(horizon_days, 0.0))
    resid_n_c = min(max(resid_n, RESID_MIN), RESID_MAX)
    sw = sum(ws)
    mx = sum(w * x for w, x in zip(ws, xs)) / sw if sw > 0 else 0.0

    rows = []
    for i, trophy in enumerate(trophies):
        a_i, b_i, resid_i, y_mean = fits[i]
        se_i = ses[i]
        # Empirical-Bayes shrinkage of the slope toward the median.
        if se_i == float("inf") or se_i <= 0:
            b_shrunk = b_i
        else:
            w_own = 1.0 / (se_i * se_i)
            w_prior = 1.0 / (tau * tau)
            b_shrunk = (b_i * w_own + beta_med * w_prior) / (w_own + w_prior)
        b_shrunk = min(max(b_shrunk, 0.0), BETA_MAX)
        # Re-centre the intercept on the shrunk slope.
        a_shrunk = y_mean - b_shrunk * mx

        log_e_target = a_shrunk + b_shrunk * x_target
        e_target = math.exp(log_e_target) - 0.5
        # Earners cannot fall below the last snapshot's earners, and
        # cannot exceed the players tracked (rate 100%).
        last_rate = reads[-1]["rates"][i] if i < len(reads[-1]["rates"]) and reads[-1]["rates"][i] is not None else 0.0
        e_last = max(0.0, float(last_rate)) * n_last / 100.0
        e_target = min(max(e_target, e_last), n_target * 0.9999)
        rate = 100.0 * e_target / n_target if n_target > 0 else 0.0
        rate = clamp(rate)

        # Spread: horizon base + this trophy's fit residual + the
        # players fit residual, combined in quadrature; plus the
        # flat-history prior term when the forecast rests on the
        # prior rather than on fitted slopes.
        resid_i_c = min(max(resid_i, RESID_MIN), RESID_MAX)
        flat_term = FLAT_SPREAD if len(reads) <= 1 else 0.0
        s = math.sqrt(base_spread ** 2 + resid_i_c ** 2 + resid_n_c ** 2 + flat_term ** 2)
        lg = logit(rate)
        qs = [round4(clamp(inv_logit(lg + z * s))) for z in Z]
        # Enforce non-decreasing (rounding can tie adjacent values,
        # which is fine, but never let them decrease).
        for k in range(1, len(qs)):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
