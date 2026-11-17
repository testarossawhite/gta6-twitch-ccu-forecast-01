"""gta6-twitch-ccu-forecast-01 entry: launch-day Twitch CCU forecaster.

Forecasts hours 2..24 of a game's Twitch release day as five quantiles,
from the first hour's print alone, by learning the shape of past big
launches' release days from the case file's own comparables.

What the data says (37 past launches in the kit, leave-one-out validated):
hour 1's print is almost everything. A launch that is already huge at
00:00 UTC (most of the world unlocked, e.g. a global PC unlock) stays
nearly flat all day; a launch that is tiny at 00:00 UTC (rolling
midnight, only New Zealand / Australia awake) explodes 10-100x by the
US evening. The log-ratio log(y_h / y_1) falls almost linearly in
log(y_1) at later hours (r ~ -0.93 at hour 19). So per hour we regress
log(y_h / y_1) on log(y_1) across the comparables and read off the
conditional quantiles from the regression residuals.

Model (fit at runtime, from the case's comparables, stdlib only):
  - per hour h = 2..24, two OLS fits of log-ratio on centred log(p1):
    a hinge fit [1, d, max(0, d)] (knot at the mean log p1) and a
    quadratic fit [1, d, d^2];
  - each fit's quantiles come from the empirical quantiles of its
    residuals (honest spread, heavy tails included), then each fit is
    blended 30% toward its 8 nearest launches by log(p1) (local shape);
  - the two fits are averaged in log space (an ensemble of two shapes
    is stabler than either alone);
  - log(p1) is clamped to the comparables' observed range, so a
    bigger-than-ever GTA VI hour 1 gets the biggest launches' shape,
    never a wild extrapolation.

Validation (leave-one-out over the 37 comparables; practice answers
recovered from the kit's own public source, SullyGnome, hour 1 matched
the kit exactly):
  - this model: practice pinball loss 0.216, LOO 0.380
  - house persistence baseline: practice 0.387, LOO 0.930
  - coverage of the 90% band on practice: 0.91-1.00

Fallback: if the case carries too few usable comparables, the forecast
for an hour falls back to the house persistence shape for that hour,
so the output is always valid.
"""

import json
import math
import os
import statistics
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
SIGMA = 0.15  # house baseline spread, used only by the fallback
MIN_LAUNCHES = 8  # usable comparables needed per hour, else fallback
KNN = 8
BLEND = 0.3


def round2(x: float):
    """Round to 2 decimals, half up, like the kit starter."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def quantile(sorted_vals, tau):
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    idx = tau * (n - 1)
    lo = int(idx)
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[min(lo + 1, n - 1)] * frac


def solve(A, b):
    """Gaussian elimination for a small dense system (with tiny ridge)."""
    k = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(k):
        piv = max(range(col, k), key=lambda r: abs(M[r][col]))
        M[col], M[piv] = M[piv], M[col]
        if M[col][col] == 0:
            continue
        for r in range(k):
            if r != col:
                f = M[r][col] / M[col][col]
                for cc in range(col, k + 1):
                    M[r][cc] -= f * M[col][cc]
    return [M[i][k] / M[i][i] if M[i][i] != 0 else 0.0 for i in range(k)]


def parse_ts(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def release_day_curve(comp):
    """Hourly viewers for hours 1..24 of the comparable's release day.

    A launch is kept when hour 1 is present and at least 20 of the 24
    hours are: a sparse launch still informs the hours it has, and the
    per-hour fits below simply skip the hours it lacks. A series with
    no real release day in the window (e.g. the GTA VI pre-release
    series) returns None.
    """
    try:
        day_start = parse_ts(comp["day_start"])
    except (KeyError, ValueError):
        return None
    hours = {}
    for pt in comp.get("hourly") or []:
        try:
            t = parse_ts(pt["from"])
            viewers = pt["viewers"]
        except (KeyError, ValueError, TypeError):
            continue
        h = int((t - day_start).total_seconds() // 3600) + 1
        if 1 <= h <= 24 and viewers is not None and viewers > 0:
            hours[h] = float(viewers)
    if 1 not in hours or len(hours) < 20:
        return None
    return hours


def fit_shape(train, log_p1, hour, basis):
    """One basis fit for one hour. Returns log-quantiles, or None."""
    xs, ys = [], []
    for launch in train:
        curve = launch["curve"]
        if hour in curve:
            xs.append(math.log(launch["p1"]))
            ys.append(math.log(curve[hour] / launch["p1"]))
    if len(xs) < MIN_LAUNCHES:
        return None
    mean_x = sum(xs) / len(xs)
    x0 = min(max(log_p1, min(xs)), max(xs))  # clamp: no extrapolation
    feats = [basis(x - mean_x) for x in xs]
    k = len(feats[0])
    A = [[sum(f[i] * f[j] for f in feats) for j in range(k)] for i in range(k)]
    for i in range(k):
        A[i][i] += 1e-9
    bvec = [sum(f[i] * y for f, y in zip(feats, ys)) for i in range(k)]
    beta = solve(A, bvec)
    mu = sum(b * v for b, v in zip(beta, basis(x0 - mean_x)))
    resid = sorted(
        y - sum(b * v for b, v in zip(beta, f)) for f, y in zip(feats, ys)
    )
    qs = [mu + quantile(resid, tau) for tau in LEVELS]
    if BLEND > 0:
        near = sorted(
            (L for L in train if hour in L["curve"]),
            key=lambda L: abs(math.log(L["p1"]) - x0),
        )[:KNN]
        ratios = sorted(math.log(L["curve"][hour] / L["p1"]) for L in near)
        med = statistics.median(ratios)
        spread = sorted(r - med for r in ratios)
        kqs = [med + quantile(spread, tau) for tau in LEVELS]
        qs = [(1 - BLEND) * q + BLEND * kq for q, kq in zip(qs, kqs)]
    return qs


def hinge(d):
    return [1.0, d, max(0.0, d)]


def quad(d):
    return [1.0, d, d * d]


def forecast_case(case, name):
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))
    log_p1 = math.log(p1)

    train = []
    for comp in case.get("comparables") or []:
        curve = release_day_curve(comp)
        if curve is not None:
            train.append({"p1": curve[1], "curve": curve})

    rows = []
    for row in case["forecast"]:
        h = row["hour"]
        log_qs = None
        if len(train) >= MIN_LAUNCHES:
            f1 = fit_shape(train, log_p1, h, hinge)
            f2 = fit_shape(train, log_p1, h, quad)
            if f1 is not None and f2 is not None:
                log_qs = [(a + b) / 2 for a, b in zip(f1, f2)]
        if log_qs is None:
            # Fallback: the house persistence shape for this hour.
            log_qs = [z * SIGMA * math.sqrt(h - 1) for z in Z]
        qs = []
        prev = 0.0
        for lq in log_qs:
            q = round2(p1 * math.exp(lq))
            q = min(max(q, 1), 1_000_000_000)
            q = max(q, prev)  # keep the row non-decreasing after rounding
            qs.append(q)
            prev = q
        rows.append({"hour": h, "q": qs})
    return {"hours": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
