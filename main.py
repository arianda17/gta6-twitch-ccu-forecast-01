#!/usr/bin/env python3
"""GTA VI Twitch CCU forecast -- v3 method, re-implemented by Muse.

Forecasts hourly Twitch concurrent viewers for hours 2-24 of a launch day
as quantiles (5/25/50/75/95) of the viewer count.

Method (v3, pinned constants from an honest grid search on the practice
mean only): anchor on the pre-midnight hour (hour 0, from history_hourly at
day_start - 1h). Each comparable's curve is expressed as log(viewers_h /
viewers_0). Comparables are weighted by unlock-pattern similarity to a
rolling-midnight launch and by Gaussian similarity (sigma 0.12) of their
pre-midnight -> hour-1 spike to the case's own spike. Per-hour weighted
quantiles of the log-ratios are shrunk toward the median (K 30) and
spread-capped (cap factor 1.0, sigma_h = 0.15*sqrt(h-1)); the whole curve
is then calibrated so the weighted-median hour-1 ratio matches the case's
own pre-midnight -> hour-1 transition. Without a pre-midnight point it
falls back to the hour-1 print as anchor, unlock-pattern weighting only.
No smoothing (window 1).

Practice mean (log pinball loss): 0.176642
  p00 0.259206, p01 0.135130, p02 0.129290, p03 0.182941.

Stdlib only, deterministic, offline, no network, no model calls.
Reads case files (*.in) from $MAKEFUN_INPUT_DIR (default /input), writes
*.out JSON to $MAKEFUN_OUTPUT_DIR (default /output).
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

TAUS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Standard-normal quantiles, rounded to 3 decimals (as tuned in v3).
Z_SCORE = {0.05: -1.645, 0.25: -0.674, 0.5: 0.0, 0.75: 0.674, 0.95: 1.645}
SHRINK_K = 30.0
CAP_FACTOR = 1.0
SMOOTH_WINDOW = 1
SIM_SIGMA = 0.12
HOURS = list(range(2, 25))


def parse_time(text):
    """Parse an ISO 8601 UTC timestamp like 2026-09-15T00:00:00Z."""
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text)


def quantile_plain(values, q):
    """Plain empirical quantile with linear interpolation."""
    if not values:
        return 0.0
    ordered = sorted(values)
    n = len(ordered)
    if n == 1:
        return float(ordered[0])
    pos = (n - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(ordered[lo])
    frac = pos - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def quantile_weighted(values, weights, q):
    """Weighted quantile by cumulative-weight interpolation.

    Values are sorted ascending with their weights; the quantile is the
    value where the cumulative weight first reaches q * total, linearly
    interpolated across the weight interval of the bracketing value.
    """
    if not values:
        return 0.0
    n = len(values)
    if n == 1:
        return float(values[0])
    order = sorted(range(n), key=lambda i: values[i])
    vs = [values[i] for i in order]
    ws = [max(0.0, float(weights[i])) for i in order]
    total = sum(ws)
    if total <= 0:
        ws = [1.0] * n
        total = float(n)
    target = q * total
    cum = 0.0
    for i in range(n):
        cum += ws[i]
        if cum >= target:
            if i == 0:
                return float(vs[0])
            prev = cum - ws[i]
            if ws[i] <= 0:
                return float(vs[i])
            frac = (target - prev) / ws[i]
            frac = max(0.0, min(1.0, frac))
            return vs[i - 1] + (vs[i] - vs[i - 1]) * frac
    return float(vs[-1])


def smooth(values, window):
    """Centered moving average; edges average over the available window."""
    n = len(values)
    if window <= 1 or n <= 1:
        return list(values)
    half = window // 2
    out = []
    for i in range(n):
        a = max(0, i - half)
        b = min(n, i + half + 1)
        out.append(sum(values[a:b]) / (b - a))
    return out


def unlock_weight(text):
    """Weight a comparable by how close its unlock pattern is to GTA VI's
    rolling midnight-local unlock (New Zealand first, then EU, UK, US)."""
    u = (text or "").lower()
    if "rolling" in u and "midnight" in u:
        return 3.0
    if "rolling" in u:
        return 2.5
    if "midnight local" in u or "midnight-local" in u:
        return 2.0
    if "simultaneous" in u or "single global" in u or "single time" in u:
        return 1.0
    return 1.5


def spike_similarity(a_comp, a_case, sigma):
    """Gaussian similarity between a comparable's pre-midnight -> hour-1
    log-ratio and the case's own."""
    if sigma <= 0:
        return 1.0
    d = a_comp - a_case
    return math.exp(-(d * d) / (2.0 * sigma * sigma))


def comp_hourly_map(comp):
    """Return (day_start, {datetime: viewers}) for a comparable, or None."""
    try:
        ds = parse_time(comp["day_start"])
        hmap = {}
        for pt in comp.get("hourly") or []:
            hmap[parse_time(pt["from"])] = pt.get("viewers")
        return ds, hmap
    except Exception:
        return None


def viewers_at(hmap, when):
    """Positive float viewer count at a timestamp, or None."""
    v = hmap.get(when)
    if v is None:
        return None
    try:
        fv = float(v)
    except Exception:
        return None
    return fv if fv > 0 else None


def case_pre_midnight(case):
    """Return (v0, v1, a_case) if the case has a pre-midnight hour, else None.
    v0 = viewers at day_start - 1h (from history_hourly), v1 = hour-1 print,
    a_case = log(v1/v0)."""
    try:
        ds = parse_time(case["day_start"])
        known = case.get("known", [])
        h1 = None
        for row in known:
            if row.get("hour") == 1:
                h1 = row
                break
        if h1 is None:
            h1 = known[0] if known else None
        if h1 is None:
            return None
        v1 = float(h1.get("viewers", 0) or 0)
        if v1 <= 0:
            return None
        v0 = None
        for pt in case.get("history_hourly", []) or []:
            try:
                if parse_time(pt["from"]) == ds - timedelta(hours=1):
                    v0 = float(pt.get("viewers", 0) or 0)
                    break
            except Exception:
                continue
        if v0 is None or v0 <= 0:
            return None
        return v0, v1, math.log(v1) - math.log(v0)
    except Exception:
        return None


def shrink_and_cap(raw_q, counts, hours):
    """Shrink per-hour quantiles toward the median and cap the spread,
    then enforce non-decreasing quantiles within each hour (log space)."""
    out = {h: {} for h in hours}
    for h in hours:
        n = counts[h]
        w = n / (n + SHRINK_K) if n > 0 else 0.0
        med = raw_q[h][0.5]
        sigma_h = 0.15 * math.sqrt(h - 1)
        for tau in TAUS:
            if tau == 0.5:
                val = med
            else:
                val = med + w * (raw_q[h][tau] - med)
                cap = abs(Z_SCORE[tau]) * sigma_h * CAP_FACTOR
                delta = val - med
                delta = max(-cap, min(cap, delta))
                val = med + delta
            out[h][tau] = val
        row = [out[h][tau] for tau in TAUS]
        for i in range(1, len(row)):
            if row[i] < row[i - 1]:
                row[i] = row[i - 1]
        for tau, v in zip(TAUS, row):
            out[h][tau] = v
    return out


def smooth_medians(q_by_hour, hours):
    """Smooth the median curve across hours (window 1 = no-op) and shift
    every quantile of each hour by the median's movement."""
    meds = [q_by_hour[h][0.5] for h in hours]
    smoothed = smooth(meds, SMOOTH_WINDOW)
    for h, m in zip(hours, smoothed):
        shift = m - q_by_hour[h][0.5]
        for tau in TAUS:
            q_by_hour[h][tau] += shift
    return q_by_hour


def forecast(case):
    known = case.get("known", [])
    h1 = None
    for row in known:
        if row.get("hour") == 1:
            h1 = row
            break
    if h1 is None:
        h1 = known[0] if known else {"viewers": 1}
    p1 = float(h1.get("viewers", 1) or 1)
    if p1 < 1:
        p1 = 1.0

    hours = HOURS
    pre = case_pre_midnight(case)

    if pre is not None:
        v0, v1, a_case = pre
        anchor = math.log(v0)
        ratios = {h: [] for h in hours}
        weights = {h: [] for h in hours}
        spike_vals = []
        spike_ws = []
        for comp in case.get("comparables", []):
            built = comp_hourly_map(comp)
            if built is None:
                continue
            ds_c, hmap = built
            v0_c = viewers_at(hmap, ds_c - timedelta(hours=1))
            v1_c = viewers_at(hmap, ds_c)
            if v0_c is None or v1_c is None:
                continue
            log_v0 = math.log(v0_c)
            a_comp = math.log(v1_c) - log_v0
            w = unlock_weight(comp.get("unlock")) * spike_similarity(
                a_comp, a_case, SIM_SIGMA
            )
            for h in hours:
                vh = viewers_at(hmap, ds_c + timedelta(hours=h - 1))
                if vh is not None:
                    ratios[h].append(math.log(vh) - log_v0)
                    weights[h].append(w)
            spike_vals.append(a_comp)
            spike_ws.append(w)

        raw = {h: {} for h in hours}
        for h in hours:
            vals = ratios[h]
            ws = weights[h]
            if not vals:
                s_h = 0.15 * math.sqrt(h - 1)
                for tau in TAUS:
                    raw[h][tau] = Z_SCORE[tau] * s_h
            else:
                for tau in TAUS:
                    raw[h][tau] = quantile_weighted(vals, ws, tau)

        q = shrink_and_cap(raw, {h: len(ratios[h]) for h in hours}, hours)
        q = smooth_medians(q, hours)

        # Calibrate: shift every hour so the weighted-median hour-1 ratio
        # matches the case's own pre-midnight -> hour-1 transition.
        r1_case = math.log(p1) - math.log(v0)
        if spike_vals and spike_ws:
            med_r1 = quantile_weighted(spike_vals, spike_ws, 0.5)
        elif spike_vals:
            med_r1 = quantile_plain(spike_vals, 0.5)
        else:
            med_r1 = 0.0
        shift = r1_case - med_r1
        for h in hours:
            for tau in TAUS:
                q[h][tau] += shift
    else:
        # Fallback: no pre-midnight point, so anchor on the hour-1 print
        # with unlock-pattern weighting only (no spike similarity).
        anchor = math.log(p1)
        ratios = {h: [] for h in hours}
        weights = {h: [] for h in hours}
        for comp in case.get("comparables", []):
            built = comp_hourly_map(comp)
            if built is None:
                continue
            ds_c, hmap = built
            v1_c = viewers_at(hmap, ds_c)
            if v1_c is None:
                continue
            log_v1 = math.log(v1_c)
            w = unlock_weight(comp.get("unlock"))
            for h in hours:
                vh = viewers_at(hmap, ds_c + timedelta(hours=h - 1))
                if vh is not None:
                    ratios[h].append(math.log(vh) - log_v1)
                    weights[h].append(w)

        raw = {h: {} for h in hours}
        for h in hours:
            vals = ratios[h]
            ws = weights[h]
            if not vals:
                s_h = 0.15 * math.sqrt(h - 1)
                for tau in TAUS:
                    raw[h][tau] = Z_SCORE[tau] * s_h
            else:
                for tau in TAUS:
                    raw[h][tau] = quantile_weighted(vals, ws, tau)

        q = shrink_and_cap(raw, {h: len(ratios[h]) for h in hours}, hours)
        q = smooth_medians(q, hours)

    # Build the output rows: exponentiate off the anchor, enforce the
    # output constraints (finite, > 0, <= 1e9, non-decreasing).
    rows = []
    for h in hours:
        vals = []
        for tau in TAUS:
            qv = math.exp(anchor + q[h][tau])
            if not math.isfinite(qv):
                s_h = 0.15 * math.sqrt(h - 1)
                qv = p1 * math.exp(Z_SCORE[tau] * s_h)
            qv = max(1e-9, min(1e9, qv))
            vals.append(qv)
        for i in range(1, len(vals)):
            if vals[i] < vals[i - 1]:
                vals[i] = vals[i - 1]
        vals = [round(v, 6) for v in vals]
        for i in range(1, len(vals)):
            if vals[i] < vals[i - 1]:
                vals[i] = vals[i - 1]
        rows.append({"hour": h, "q": vals})
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast_out = forecast(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast_out, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
