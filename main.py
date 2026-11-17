"""gta6-twitch-ccu-forecast-01 entry (Python 3.13, standard library only).

Forecasts the Twitch viewers of the "Grand Theft Auto VI" category for each
hour 2..24 of the release day (2026-11-19 UTC), as five quantiles, from the
case's hour-1 print.

Model (fitted offline on the 38 comparable launch curves shipped in the
case file; the program re-fits it at run time from the case's own
`comparables`, so the final case's 42 comparables are used as given):

  For each comparable launch, take its release-day hourly curve
  (hours 1..24, from its `hourly` list and `day_start`) and keep only the
  "rolling-like" launches: hour 1 already material (peak / hour-1 <= 20),
  a late peak (peak hour >= 10) and hour-1 >= 1000 viewers.  That is the
  shape of a midnight-local rolling unlock (GTA VI, the four practice
  launches, most console AAA launches): the audience builds through the
  day as time zones unlock, instead of exploding from ~0 at one global
  unlock hour.  Simultaneous-global launches (Silksong, Wukong,
  Battlefield 6, ...) and the GTA VI category's own pre-release history
  are left out.

  For each hour h = 2..24, fit by least squares over the kept launches

      ln(v_h / v_1) = a_h + b_h * ln(v_1)

  i.e. the day's growth depends on how big hour 1 already is: launches
  whose hour 1 is already huge (the world is already watching) grow
  little, small hour-1 launches grow many-fold (corr about -0.7 at the
  evening peak).  The size effect saturates: the log hour-1 used for
  prediction is capped at ln(300_000), because the biggest launches on
  record (365k-533k in hour 1) all peak at 1.25-1.34x hour 1, never
  below it -- an uncapped log-linear fit would wrongly predict a
  *decline* from hour 1 for a GTA-VI-sized hour 1.

  Quantiles at level tau are then

      q_tau(h) = p1 * exp(mu_h + z_tau * s_h)

  with mu_h the fitted value at the (capped) log hour-1, s_h the
  residual standard deviation of that hour's fit (floored at 0.25 so
  the spread never claims more precision than the launch-to-launch
  scatter allows), and z_tau the standard normal quantiles
  (-1.645, -0.674, 0, 0.674, 1.645).

Local validation (practice answers fetched from the public SullyGnome
pages the kit documents, for the four practice launches only):
mean pinball loss 0.206 on the practice cases, against 0.387 for the
house baseline; leave-one-out over the 24 rolling-like comparables
0.272, against 0.41 for the house baseline on the same curves.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0)
in python:3.13-slim, standard library only: no network, no pip packages.
-P keeps this file's directory OFF sys.path, so a module shipped beside
main.py imports only after the sys.path line below.

What an entry must do:
  - read every <id>.in in the input directory (the grader runs one case
    at a time, with only that case's file there): JSON
    {"schema": "makefun/twitch-ccu-case/v1", "id", "game", "day_start",
    "checkpoint", "levels", "known", "forecast", "history",
    "history_hourly", "streams", "categories_hourly", "comparables",
    "notes"}, where `known` holds hour 1's print {hour: 1, from, until,
    viewers, channels, reads} and `forecast` lists the hours to
    forecast, 2..24;
  - write <id>.out to the output directory: {"hours": [{"hour": 2, "q":
    [q05, q25, q50, q75, q95]}, ...]}, exactly 23 rows, hours 2..24
    each once, no other key, each q five finite numbers above 0 and at
    most 1e9 (viewers; fractions allowed), non-decreasing; at most
    65,536 bytes.  Anything else is an invalid case, scored as if every
    quantile equalled hour 1's print, and an entry with an invalid case
    takes no prize.

Outside the grader's container, MAKEFUN_INPUT_DIR and
MAKEFUN_OUTPUT_DIR stand in for /input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import statistics
import sys
from datetime import datetime
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import
# model` finds a module shipped beside this file (none is needed here).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# A rolling-like launch: hour 1 already material (peak / hour-1 <= 20),
# a late peak (peak hour >= 10) and hour-1 >= 1000 viewers.
MAX_PEAK_RATIO = 20.0
MIN_PEAK_HOUR = 10
MIN_HOUR1 = 1000
# The size effect saturates here: beyond this hour-1 size the fitted
# log-ratio is evaluated at the cap, not extrapolated further down.
LOG_HOUR1_CAP = math.log(300_000)
# The spread never claims more precision than this, in log units.
SD_FLOOR = 0.25
# The house baseline's spread (fallback when a fit has too few curves).
BASELINE_SIGMA = 0.15


def parse_time(text):
    """ISO 8601 UTC timestamp -> datetime (aware)."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def round2(x):
    """x rounded to 2 decimals, half up, as JavaScript's
    Math.round(x * 100) / 100 rounds it.  A whole number comes back as
    an int, so json.dumps writes 1240 where JavaScript writes 1240
    (not 1240.0)."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def release_day_curve(comp):
    """{hour: viewers} for the comparable's release day (hours 1..24),
    from its `hourly` points and `day_start`.  Hours with no point are
    left out; a point of 0 viewers is left out (no log of it)."""
    day_start = parse_time(comp["day_start"])
    curve = {}
    for point in comp.get("hourly") or []:
        viewers = point.get("viewers")
        if not viewers or viewers < 1:
            continue
        hour = int((parse_time(point["from"]) - day_start).total_seconds() // 3600) + 1
        if 1 <= hour <= 24:
            curve[hour] = viewers
    return curve


def is_rolling_like(curve):
    """True when the launch's release-day shape is the rolling-unlock
    shape this forecast is built for."""
    hour1 = curve.get(1)
    if not hour1 or hour1 < MIN_HOUR1:
        return False
    if len(curve) < 20:
        return False
    peak_hour = max(curve, key=curve.get)
    peak = curve[peak_hour]
    if peak_hour < MIN_PEAK_HOUR:
        return False
    return peak / hour1 <= MAX_PEAK_RATIO


def ols(xs, ys):
    """Least squares y = a + b*x; returns (a, b, residual sd)."""
    n = len(xs)
    mx = statistics.mean(xs)
    my = statistics.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        b = 0.0
    else:
        b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    sd = math.sqrt(sum(r * r for r in resid) / max(n - 2, 1))
    return a, b, sd


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1, first["viewers"])
    log_p1 = math.log(p1)
    log_p1_capped = min(log_p1, LOG_HOUR1_CAP)

    # The training set: this case's own comparables, rolling-like only.
    # The GTA VI category's own pre-release history is not a launch and
    # is left out by name as well as by shape (it has no release day).
    curves = []
    for comp in case.get("comparables") or []:
        if "Grand Theft Auto VI" in comp.get("name", ""):
            continue
        curve = release_day_curve(comp)
        if is_rolling_like(curve):
            curves.append(curve)

    hours = []
    for row in case["forecast"]:
        h = row["hour"]
        xs = []
        ys = []
        for curve in curves:
            v = curve.get(h)
            if v and v >= 1:
                xs.append(math.log(curve[1]))
                ys.append(math.log(v / curve[1]))
        if len(xs) >= 6:
            a, b, sd = ols(xs, ys)
            sd = max(sd, SD_FLOOR)
            mu = a + b * log_p1_capped
            # Guard against a degenerate fit: the median ratio stays
            # inside the range the rolling-like launches ever show
            # (peak / hour-1 tops out near 12x; the filter already
            # excludes anything above 20x).
            mu = min(max(mu, math.log(0.05)), math.log(20.0))
            quantiles = [p1 * math.exp(mu + z * sd) for z in Z]
        else:
            # Too few curves for a fit: fall back to the house baseline
            # (flat at hour 1, spread widening over the day).
            quantiles = [
                p1 * math.exp(z * BASELINE_SIGMA * math.sqrt(h - 1)) for z in Z
            ]
        quantiles = [round2(min(max(q, 0.01), 1_000_000_000)) for q in quantiles]
        # Keep the row non-decreasing after rounding.
        for i in range(1, 5):
            if quantiles[i] < quantiles[i - 1]:
                quantiles[i] = quantiles[i - 1]
        hours.append({"hour": h, "q": quantiles})
    hours.sort(key=lambda r: r["hour"])
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
