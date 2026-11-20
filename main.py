"""Forecast GTA VI's PS5 trophy earn rates on Exophase at a case's target.

Model: count decomposition. An earn rate is earned_count / players_tracked,
and both move fast after a launch, so this program forecasts the two counts
separately and divides, instead of extrapolating the rate itself.

  - players_tracked P(t): least-squares line of ln(P) on ln(1 + hours since
    launch), fitted on the recent distinct reads, read at the target.
  - earned(t) per trophy: E = rate * P / 100, reconstructed from each read,
    then a least-squares line of ln(E + 0.5) on the same clock, read at the
    target. Earned counts never fall, so the forecast is floored at the
    last observed count.
  - median rate = 100 * E(target) / P(target), blended in logit space with
    the last observed rate (persistence) by the fits' quality.
  - spread: a random-walk term over the horizon, the fits' prediction
    intervals at the target, and an allowance for Exophase's refresh jumps,
    combined on the logit scale.

Standard library only. Deterministic: no randomness, inputs read in name
order, trophies written in the case's order.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file would import.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# Launch day start (UTC), the clock's origin: the house fits against hours
# from this instant, and the launch surge decays in log time from it.
LAUNCH = datetime(2026, 11, 19, tzinfo=timezone.utc)

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

# Random-walk spread of a rate's move, in logit units per square-root day.
# Slightly below the house baseline's 0.5: the median here carries the trend,
# so the spread need not cover the drift the baseline ignores.
SIGMA_RW = 0.45
# Allowance for Exophase's refresh jumps, in logit units: the answering read
# shows the rates as of the last refresh before the target, and a refresh
# moves every rate at once (launch night's first refresh cut most rates by
# roughly a fifth to two fifths in relative terms).
SIGMA_REFRESH = 0.25
# Spread floor and cap, in logit units: never overconfident, never absurd.
SPREAD_MIN = 0.35
SPREAD_MAX = 1.35
# Smoothing count for earned-count logs: keeps never-earned trophies finite.
EARNED_SMOOTH = 0.5
# Recent window for the fits, in hours: the last day's trend is what the
# target inherits; older reads describe the launch surge, not the week.
WINDOW_HOURS = 24.0
# Least-squares slope cap, in ln units per ln hour: faster than this is a
# data artefact, not a launch curve.
SLOPE_MAX = 3.0


def parse_time(text):
    """An ISO 8601 UTC timestamp (with or without fractions) to a datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def clock_x(moment):
    """Log-time coordinate: ln(1 + hours from launch), floored at launch."""
    hours = (moment - LAUNCH).total_seconds() / 3600.0
    return math.log1p(max(0.0, hours))


def clamp_rate(pct):
    """An earn rate clamped to the range the logit accepts."""
    return min(99.99, max(0.01, pct))


def logit(pct):
    """ln(p / (1 - p)) of a percent, clamped first (0 and 100 have no logit)."""
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(value):
    """A logit back to a percent, kept inside (0, 100)."""
    if value >= 0:
        return 100.0 / (1.0 + math.exp(-value))
    top = math.exp(value)
    return 100.0 * top / (1.0 + top)


def round4(x):
    """x rounded to 4 decimals, half up; a whole number comes back an int."""
    scaled = x * 10000.0
    whole = math.floor(scaled + 0.5)
    value = whole / 10000.0
    return int(value) if value.is_integer() else value


def norm_ppf(p):
    """The standard normal's quantile at p (Acklam's rational approximation).

    Accurate to about 1e-9, far tighter than the 4 decimals the forecast
    prints. Used so the program honours whatever levels a case asks for.
    """
    a = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
    b = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00)
    p_low, p_high = 0.02425, 1.0 - 0.02425
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        num = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        den = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        return num / den
    if p <= p_high:
        q = p - 0.5
        r = q * q
        num = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        den = (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        return num / den
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    num = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
    den = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    return -num / den


def least_squares(xs, ys):
    """Least-squares line y ~ a + b*x. Returns (slope, intercept, resid_std,
    n, x_mean, sxx); a degenerate set (fewer than 2 points, or all x equal)
    returns a flat line at the mean of y with zero residual spread."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0, 0, 0.0, 0.0
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    sxx = sum((x - x_mean) ** 2 for x in xs)
    if n < 2 or sxx <= 0.0:
        return 0.0, y_mean, 0.0, n, x_mean, 0.0
    sxy = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = y_mean - slope * x_mean
    if n > 2:
        resid = math.sqrt(sum((y - (intercept + slope * x)) ** 2
                              for x, y in zip(xs, ys)) / (n - 2))
    else:
        resid = 0.0
    return slope, intercept, resid, n, x_mean, sxx


def prediction_std(resid, n, x_mean, sxx, x0):
    """The standard error of a least-squares prediction at x0: the residual
    spread widened by leverage, how far x0 sits from the fit's centre."""
    if n == 0 or resid <= 0.0:
        return 0.0
    leverage = 1.0 + 1.0 / n
    if sxx > 0.0:
        leverage += (x0 - x_mean) ** 2 / sxx
    return resid * math.sqrt(leverage)


def distinct_reads(history):
    """The reads with new information, oldest first: Exophase refreshes on
    its own schedule, so long runs of reads repeat the same page exactly."""
    kept = []
    last_sig = None
    for read in history:
        sig = (read.get("players_tracked"),
               tuple(read.get("rates") or []))
        if sig != last_sig:
            kept.append(read)
            last_sig = sig
    return kept


def window_reads(reads, checkpoint, hours=WINDOW_HOURS):
    """The reads in the last `hours` before the checkpoint; fewer than three
    and the whole history stands in, so a young case still fits."""
    if not reads:
        return []
    cutoff = checkpoint.timestamp() - hours * 3600.0
    recent = [r for r in reads if parse_time(r["at"]).timestamp() >= cutoff]
    return recent if len(recent) >= 3 else reads


def fit_counts(reads, target_x):
    """Fit ln(players) on log-time over the given reads and read it at the
    target. Returns (players_at_target, prediction_std, last_players)."""
    xs, ys = [], []
    for read in reads:
        players = read.get("players_tracked") or 0
        if players > 0:
            xs.append(clock_x(parse_time(read["at"])))
            ys.append(math.log(players))
    slope, intercept, resid, n, x_mean, sxx = least_squares(xs, ys)
    last_players = reads[-1].get("players_tracked") or 1
    if n < 2:
        return float(last_players), 0.0, float(last_players)
    slope = min(max(slope, 0.0), SLOPE_MAX)
    players = math.exp(intercept + slope * target_x)
    # Players never shrink between reads in practice; never forecast below
    # the last count, and never above fifty times it (a sanity cap).
    players = min(max(players, float(last_players)), 50.0 * last_players)
    return players, prediction_std(resid, n, x_mean, sxx, target_x), float(last_players)


def forecast_trophy(trophy, index, reads, target_x, players_target,
                    players_std, last_rate):
    """One trophy's five quantiles at the target.

    Earned counts come from the reads (rate x players / 100) and are fitted
    in log-time; the median rate is the forecast earned count over the
    forecast players, blended with persistence by the fits' quality; the
    spread combines the horizon's random walk, both fits' prediction
    intervals and the refresh allowance, on the logit scale.
    """
    xs, ys = [], []
    for read in reads:
        rates = read.get("rates") or []
        if index >= len(rates) or rates[index] is None:
            continue
        players = read.get("players_tracked") or 0
        if players <= 0:
            continue
        earned = rates[index] * players / 100.0
        xs.append(clock_x(parse_time(read["at"])))
        ys.append(math.log(earned + EARNED_SMOOTH))
    slope, intercept, resid, n, x_mean, sxx = least_squares(xs, ys)
    if n >= 2:
        slope = min(max(slope, 0.0), SLOPE_MAX)
        earned = math.exp(intercept + slope * target_x) - EARNED_SMOOTH
        earned_std = prediction_std(resid, n, x_mean, sxx, target_x)
    else:
        earned = None
        earned_std = 0.0
    last_earned = None
    for read in reversed(reads):
        rates = read.get("rates") or []
        players = read.get("players_tracked") or 0
        if index < len(rates) and rates[index] is not None and players > 0:
            last_earned = rates[index] * players / 100.0
            break
    if earned is None or last_earned is None:
        model_rate = last_rate
    elif last_rate >= 95.0:
        # A saturated rate: earned counts track the player base one for one
        # (everyone holds this trophy), so any gap between the two fitted
        # slopes is estimation noise, not signal, and extrapolating it would
        # drag a ~100% rate down for no reason. Hold the last rate.
        model_rate = last_rate
    else:
        # Earned counts never fall: floor the forecast at the last count,
        # and never above the whole player base.
        earned = min(max(earned, last_earned), players_target)
        model_rate = 100.0 * earned / players_target
    # Blend toward persistence by fit quality: tight fits trust the model,
    # noisy ones lean on the last rate. The model always carries the most.
    noise = earned_std + players_std
    weight = 0.55 + 0.35 * math.exp(-noise)
    median_logit = (weight * logit(model_rate)
                    + (1.0 - weight) * logit(last_rate))
    return median_logit, earned_std


def forecast_case(case, name):
    """A case's forecast rows: one {"id", "q"} row per trophy, in id order."""
    levels = case.get("levels") or LEVELS
    z_scores = [norm_ppf(level) for level in levels]
    checkpoint = parse_time(case["at"])
    target = parse_time(case["target"])
    target_x = clock_x(target)
    horizon_days = case["horizon_minutes"] / 1440.0
    reads = distinct_reads(case.get("history") or [])
    if not reads:
        raise ValueError(f"{name} has no history")
    fit_reads = window_reads(reads, checkpoint)
    players_target, players_std, _ = fit_counts(fit_reads, target_x)
    rows = []
    for index, trophy in enumerate(case["trophies"]):
        last_rate = None
        for read in reversed(reads):
            rates = read.get("rates") or []
            if index < len(rates) and rates[index] is not None:
                last_rate = rates[index]
                break
        if last_rate is None:
            last_rate = 0.01
        try:
            median_logit, earned_std = forecast_trophy(
                trophy, index, fit_reads, target_x, players_target,
                players_std, last_rate)
        except Exception:
            median_logit, earned_std = logit(last_rate), 0.0
        spread = math.sqrt((SIGMA_RW * math.sqrt(horizon_days)) ** 2
                           + earned_std ** 2 + players_std ** 2
                           + SIGMA_REFRESH ** 2)
        spread = min(max(spread, SPREAD_MIN), SPREAD_MAX)
        quantiles = [round4(clamp_rate(inv_logit(median_logit + z * spread)))
                     for z in z_scores]
        # Rounding can only keep order (inv_logit is monotone and the z
        # scores rise), but enforce non-decreasing explicitly: the format
        # requires it, and a tie is allowed.
        for k in range(1, len(quantiles)):
            if quantiles[k] < quantiles[k - 1]:
                quantiles[k] = quantiles[k - 1]
        rows.append({"id": trophy["id"], "q": quantiles})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case, path.name)
        except Exception:
            # Never crash a run: fall back to persistence, the last rate
            # with the house spread, which is always a valid forecast.
            last = case["history"][-1]
            spread = 0.5 * math.sqrt(case["horizon_minutes"] / 1440.0)
            rows = []
            for i, trophy in enumerate(case["trophies"]):
                rate = last["rates"][i]
                lg = logit(rate if rate is not None else 0)
                rows.append({"id": trophy["id"],
                             "q": [round4(clamp_rate(inv_logit(lg + norm_ppf(lv) * spread)))
                                   for lv in (case.get("levels") or LEVELS)]})
            forecast = {"trophies": rows}
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
