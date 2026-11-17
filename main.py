#!/usr/bin/env python3
"""GTA VI Twitch CCU forecast entry (v3 method, rebuilt from the prose spec).

Method: pre-midnight anchor (hour 0 from history_hourly at day_start - 1h).
a_case = log(v1 / v0). Each comparable contributes its curve in
r0(h) = log(viewers_h / viewers_0) space, weighted by unlock-pattern
similarity times a Gaussian similarity (sigma 0.12) of its pre-midnight
-> hour-1 spike to the case's own spike. Per-hour weighted quantiles
(cumulative-weight interpolation) are shrunk toward the weighted median
(w = n / (n + 30)), capped at 1.0 * |z_tau| * 0.15 * sqrt(h - 1) around the
median with the 3-decimal z-scores 1.645 / 0.674, then shifted so the
weighted-median hour-1 ratio matches a_case (damping 1.0). Fallback (no
pre-midnight point): hour-1 anchor with unlock weighting only.

Unlock cascade (first match wins, case-insensitive): "rolling" + "midnight"
-> 3.0, "rolling" -> 2.5, "midnight local" -> 2.0, "simultaneous" /
"single global" / "single time" -> 1.0, anything else -> 1.5.

Stdlib only, deterministic, offline. Reads cases from $MAKEFUN_INPUT_DIR
(default /input), writes /output/<id>.out JSON with 23 rows (hours 2-24).
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# --- Tuned constants (v3; tuned on the practice mean only) ---
SIGMA = 0.12            # Gaussian width for a_case similarity
K_SHRINK = 30           # shrinkage: w = n / (n + K_SHRINK)
SPREAD_CAP_FACTOR = 1.0  # spread cap multiplier
SPREAD_BASE = 0.15      # spread cap: factor * |z| * 0.15 * sqrt(h - 1)
DAMP = 1.0              # calibration-shift damping (1.0 = full shift)

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
# Normal-quantile magnitudes, rounded to 3 decimals (pinned by the spec).
Z_VALUES = {0.05: 1.645, 0.25: 0.674, 0.5: 0.0, 0.75: 0.674, 0.95: 1.645}

FORECAST_HOURS = range(2, 25)


def parse_ts(text):
    """Parse an ISO-8601 UTC timestamp like '2026-09-15T00:00:00Z'."""
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def fmt_ts(moment):
    """Format a datetime back to the same ISO-8601 UTC form."""
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def shift_ts(text, hours):
    """Return the timestamp string `hours` after the given timestamp."""
    return fmt_ts(parse_ts(text) + timedelta(hours=hours))


def unlock_weight(unlock_text):
    """Weight a comparable by how its launch-unlock pattern matches a
    rolling-midnight launch. Cascade (first match wins):
    rolling + midnight -> 3.0, rolling -> 2.5, midnight local -> 2.0,
    simultaneous / single global / single time -> 1.0, anything else -> 1.5.
    """
    text = (unlock_text or "").lower()
    if "rolling" in text and "midnight" in text:
        return 3.0
    if "rolling" in text:
        return 2.5
    if "midnight local" in text:
        return 2.0
    if "simultaneous" in text or "single global" in text or "single time" in text:
        return 1.0
    return 1.5


def weighted_quantile(pairs, tau):
    """Weighted quantile by cumulative-weight interpolation.

    `pairs` is a list of (value, weight) with positive weights. Values are
    sorted ascending; the cumulative weight is walked until it reaches
    tau * total weight, and the result is linearly interpolated between the
    bracketing values on the cumulative-weight axis.
    """
    ordered = sorted(pairs, key=lambda pair: pair[0])
    total = sum(weight for _, weight in ordered)
    if total <= 0:
        return ordered[0][0]
    target = tau * total
    prev_cum = 0.0
    prev_val = ordered[0][0]
    cum = 0.0
    for val, weight in ordered:
        cum += weight
        if cum >= target:
            if cum == prev_cum:
                return val
            frac = (target - prev_cum) / (cum - prev_cum)
            return prev_val + frac * (val - prev_val)
        prev_cum = cum
        prev_val = val
    return ordered[-1][0]


def weighted_median(pairs):
    """Weighted median (tau = 0.5) via cumulative-weight interpolation."""
    return weighted_quantile(pairs, 0.5)


def hourly_viewers(entries):
    """Map an hourly series (entries with 'from' + 'viewers') to a
    {timestamp string: viewers} dict for exact timestamp lookups."""
    return {entry["from"]: entry["viewers"] for entry in entries}


def quantile_rows(hour_pairs, anchor, shift):
    """Build the 23 forecast rows (hours 2-24) from per-hour weighted
    log-ratio samples.

    hour_pairs: {hour: [(log-ratio, weight), ...]}. For each hour the
    weighted quantiles are shrunk toward the weighted median
    (w = n / (n + K_SHRINK)), capped at
    SPREAD_CAP_FACTOR * |z_tau| * SPREAD_BASE * sqrt(h - 1) around the
    median, shifted by `shift`, and exponentiated against `anchor`.
    """
    rows = []
    for h in FORECAST_HOURS:
        pairs = hour_pairs.get(h) or []
        if not pairs:
            rows.append({"hour": h, "q": [float(anchor)] * len(LEVELS)})
            continue
        n = len(pairs)
        med = weighted_median(pairs)
        shrink_w = n / (n + K_SHRINK)
        qs = []
        for tau in LEVELS:
            q = weighted_quantile(pairs, tau)
            q = med + shrink_w * (q - med)
            cap = SPREAD_CAP_FACTOR * Z_VALUES[tau] * SPREAD_BASE * math.sqrt(h - 1)
            dev = q - med
            if dev > cap:
                q = med + cap
            elif dev < -cap:
                q = med - cap
            qs.append(anchor * math.exp(q + shift))
        qs = [min(max(v, 1e-6), 1e9) for v in qs]
        for i in range(1, len(qs)):
            if qs[i] < qs[i - 1]:
                qs[i] = qs[i - 1]
        rows.append({"hour": h, "q": qs})
    return rows


def forecast_primary(case, v0, v1):
    """Pre-midnight-anchored forecast. a_case = log(v1 / v0); each
    comparable is weighted by unlock pattern times Gaussian similarity of
    its own a_comp to a_case; the calibration shift aligns the
    weighted-median hour-1 ratio with a_case."""
    a_case = math.log(v1 / v0)
    hour_pairs = {h: [] for h in FORECAST_HOURS}
    r1_pairs = []
    for comp in case.get("comparables", []):
        day_start = comp["day_start"]
        series = hourly_viewers(comp.get("hourly", []))
        vc0 = series.get(shift_ts(day_start, -1))
        vc1 = series.get(day_start)
        if not vc0 or not vc1 or vc0 <= 0 or vc1 <= 0:
            continue
        a_comp = math.log(vc1 / vc0)
        weight = unlock_weight(comp.get("unlock", "")) * math.exp(
            -((a_comp - a_case) ** 2) / (2 * SIGMA * SIGMA)
        )
        r1_pairs.append((a_comp, weight))
        for h in FORECAST_HOURS:
            vh = series.get(shift_ts(day_start, h - 1))
            if vh and vh > 0:
                hour_pairs[h].append((math.log(vh / vc0), weight))
    shift = DAMP * (a_case - weighted_median(r1_pairs))
    return {"hours": quantile_rows(hour_pairs, v0, shift)}


def forecast_fallback(case, v1):
    """Hour-1-anchored fallback (no pre-midnight point): log-ratios are
    taken against each comparable's hour-1 viewers, unlock weighting only,
    no calibration shift."""
    hour_pairs = {h: [] for h in FORECAST_HOURS}
    for comp in case.get("comparables", []):
        day_start = comp["day_start"]
        series = hourly_viewers(comp.get("hourly", []))
        vc1 = series.get(day_start)
        if not vc1 or vc1 <= 0:
            continue
        weight = unlock_weight(comp.get("unlock", ""))
        for h in FORECAST_HOURS:
            vh = series.get(shift_ts(day_start, h - 1))
            if vh and vh > 0:
                hour_pairs[h].append((math.log(vh / vc1), weight))
    return {"hours": quantile_rows(hour_pairs, v1, 0.0)}


def forecast_case(case):
    """Forecast one case dict. Uses the pre-midnight anchor when the
    history has a point at day_start - 1h, else the hour-1 fallback."""
    day_start = case["day_start"]
    history = hourly_viewers(case.get("history_hourly", []))
    v1 = history.get(day_start)
    if v1 is None or v1 <= 0:
        for entry in case.get("known", []):
            if entry.get("hour") == 1:
                v1 = entry.get("viewers")
                break
    if v1 is None or v1 <= 0:
        raise ValueError("no hour-1 viewers in case %s" % case.get("id"))
    v0 = history.get(shift_ts(day_start, -1))
    if v0 is not None and v0 > 0:
        return forecast_primary(case, v0, v1)
    return forecast_fallback(case, v1)


def main(argv=None):
    """Read every *.in case in $MAKEFUN_INPUT_DIR (default /input) and
    write /output/<id>.out (or $MAKEFUN_OUTPUT_DIR). Extra argv: input dir,
    output dir."""
    argv = list(sys.argv[1:] if argv is None else argv)
    input_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    output_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    if argv:
        input_dir = Path(argv[0])
    if len(argv) > 1:
        output_dir = Path(argv[1])
    output_dir.mkdir(parents=True, exist_ok=True)
    for case_file in sorted(input_dir.glob("*.in")):
        case = json.loads(case_file.read_text())
        result = forecast_case(case)
        cid = case.get("id") or case_file.stem
        (output_dir / (cid + ".out")).write_text(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
