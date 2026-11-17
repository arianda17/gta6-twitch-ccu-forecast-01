import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = {0.05: -1.645, 0.25: -0.674, 0.5: 0.0, 0.75: 0.674, 0.95: 1.645}
K_SHRINK = 30.0
SPREAD_CAP_FACTOR = 1.0  # cap implied spread to this multiple of baseline z*sigma_h
SMOOTH_WINDOW = 1  # no smoothing (centered moving average window)
ACASE_SIGMA = 0.12  # Gaussian kernel width for a_case similarity weighting


def parse_ts(s):
    # ISO 8601 UTC like 2026-09-15T00:00:00Z
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


def empirical_quantile(values, q):
    """Linear-interpolation quantile (pos = (n-1)*q)."""
    if not values:
        return 0.0
    vs = sorted(values)
    n = len(vs)
    if n == 1:
        return float(vs[0])
    pos = (n - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(vs[lo])
    frac = pos - lo
    return vs[lo] + (vs[hi] - vs[lo]) * frac


def weighted_quantile(vals, ws, q):
    """Weighted quantile via cumulative-weight interpolation."""
    if not vals:
        return 0.0
    n = len(vals)
    if n == 1:
        return float(vals[0])
    order = sorted(range(n), key=lambda i: vals[i])
    vs = [vals[i] for i in order]
    w = [max(0.0, float(ws[i])) for i in order]
    total = sum(w)
    if total <= 0:
        w = [1.0] * n
        total = float(n)
    target = q * total
    cum = 0.0
    for i in range(n):
        cum += w[i]
        if cum >= target:
            if i == 0:
                return float(vs[0])
            prev_cum = cum - w[i]
            if w[i] <= 0:
                return float(vs[i])
            frac = (target - prev_cum) / w[i]
            if frac < 0.0:
                frac = 0.0
            elif frac > 1.0:
                frac = 1.0
            return vs[i - 1] + (vs[i] - vs[i - 1]) * frac
    return float(vs[-1])


def smooth_centered(values, window):
    """Centered moving average; edges use smaller windows."""
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


def unlock_weight(unlock_text):
    """Weight comparables by unlock-pattern similarity to GTA VI (rolling midnight local)."""
    u = (unlock_text or "").lower()
    # GTA VI: rolling midnight-local unlock across time zones (NZ first, then EU, UK, US).
    if "rolling" in u and "midnight" in u:
        return 3.0
    if "rolling" in u:
        return 2.5
    if "midnight local" in u or "midnight-local" in u:
        return 2.0
    if "simultaneous" in u or "single global" in u or "single time" in u:
        return 1.0
    # Early access / unverified / other: neutral-ish weight
    return 1.5


def acase_similarity(a_comp, a_case, sigma):
    """Gaussian similarity of comparable's pre-midnight-to-hour1 log-ratio to case's."""
    if sigma <= 0:
        return 1.0
    diff = a_comp - a_case
    return math.exp(-(diff * diff) / (2.0 * sigma * sigma))


def build_comp_maps(comp):
    """Return (ds, hmap) where hmap maps datetime -> viewers, or None if unusable."""
    try:
        ds = parse_ts(comp["day_start"])
        hourly = comp.get("hourly") or []
        hmap = {}
        for pt in hourly:
            t = parse_ts(pt["from"])
            hmap[t] = pt.get("viewers")
        return ds, hmap
    except Exception:
        return None


def get_viewer(hmap, t):
    v = hmap.get(t)
    if v is None:
        return None
    try:
        fv = float(v)
    except Exception:
        return None
    if fv <= 0:
        return None
    return fv


def get_case_pre_midnight(case):
    """Return (v0, v1, a_case) for the case if pre-midnight hour exists, else None."""
    try:
        ds = parse_ts(case["day_start"])
        # hour 1 print
        known = case.get("known", [])
        h1_row = None
        for row in known:
            if row.get("hour") == 1:
                h1_row = row
                break
        if h1_row is None:
            h1_row = known[0] if known else None
        if h1_row is None:
            return None
        v1 = float(h1_row.get("viewers", 0) or 0)
        if v1 <= 0:
            return None
        # pre-midnight: ds - 1h
        v0 = None
        for pt in case.get("history_hourly", []) or []:
            try:
                if parse_ts(pt["from"]) == ds - timedelta(hours=1):
                    v0 = float(pt.get("viewers", 0) or 0)
                    break
            except Exception:
                continue
        if v0 is None or v0 <= 0:
            return None
        a_case = math.log(v1) - math.log(v0)
        return v0, v1, a_case
    except Exception:
        return None


def forecast_case(case):
    # Hour-1 anchor (always available)
    known = case.get("known", [])
    h1_row = None
    for row in known:
        if row.get("hour") == 1:
            h1_row = row
            break
    if h1_row is None:
        h1_row = known[0] if known else {"viewers": 1}
    p = float(h1_row.get("viewers", 1) or 1)
    if p < 1:
        p = 1.0

    pre = get_case_pre_midnight(case)
    hours_seq = list(range(2, 25))

    if pre is not None:
        v0_case, v1_case, a_case = pre
        anchor_log = math.log(v0_case)
        ratios_by_hour = {h: [] for h in hours_seq}
        weights_by_hour = {h: [] for h in hours_seq}
        r1_vals = []
        r1_weights = []

        for comp in case.get("comparables", []):
            built = build_comp_maps(comp)
            if built is None:
                continue
            ds_c, hmap = built
            v0_c = get_viewer(hmap, ds_c - timedelta(hours=1))
            v1_c = get_viewer(hmap, ds_c)
            if v0_c is None or v1_c is None:
                continue
            lv0 = math.log(v0_c)
            a_comp = math.log(v1_c) - lv0
            w = unlock_weight(comp.get("unlock")) * acase_similarity(a_comp, a_case, ACASE_SIGMA)
            for h in hours_seq:
                th = ds_c + timedelta(hours=h - 1)
                vh = get_viewer(hmap, th)
                if vh is not None:
                    ratios_by_hour[h].append(math.log(vh) - lv0)
                    weights_by_hour[h].append(w)
            r1_vals.append(a_comp)
            r1_weights.append(w)

        # Empirical weighted quantiles per hour (ratios relative to v0)
        raw_q = {h: {} for h in hours_seq}
        for h in hours_seq:
            vals = ratios_by_hour[h]
            ws = weights_by_hour[h]
            if not vals:
                sigma_h = 0.15 * math.sqrt(h - 1)
                for tau in LEVELS:
                    raw_q[h][tau] = Z[tau] * sigma_h
            else:
                for tau in LEVELS:
                    raw_q[h][tau] = weighted_quantile(vals, ws, tau)

        # Shrinkage toward median + spread cap
        shrunk_q = {h: {} for h in hours_seq}
        for h in hours_seq:
            n = len(ratios_by_hour[h])
            w = n / (n + K_SHRINK) if n > 0 else 0.0
            med = raw_q[h][0.5]
            sigma_h = 0.15 * math.sqrt(h - 1)
            for tau in LEVELS:
                if tau == 0.5:
                    val = med
                else:
                    val = med + w * (raw_q[h][tau] - med)
                    z = Z[tau]
                    cap = abs(z) * sigma_h * SPREAD_CAP_FACTOR
                    delta = val - med
                    if delta > cap:
                        delta = cap
                    elif delta < -cap:
                        delta = -cap
                    val = med + delta
                shrunk_q[h][tau] = val
            # Re-enforce non-decreasing quantiles within hour (log space)
            tvals = [shrunk_q[h][tau] for tau in LEVELS]
            for i in range(1, len(tvals)):
                if tvals[i] < tvals[i - 1]:
                    tvals[i] = tvals[i - 1]
            for tau, v in zip(LEVELS, tvals):
                shrunk_q[h][tau] = v

        # Smooth median curve across hours (window=1 => no-op)
        med_series = [shrunk_q[h][0.5] for h in hours_seq]
        med_smooth = smooth_centered(med_series, SMOOTH_WINDOW)
        for h, m in zip(hours_seq, med_smooth):
            shift = m - shrunk_q[h][0.5]
            for tau in LEVELS:
                shrunk_q[h][tau] += shift

        # Calibrate to case's actual pre-midnight -> hour1 transition
        r1_case = math.log(p) - math.log(v0_case)
        if r1_vals and r1_weights:
            med_r1 = weighted_quantile(r1_vals, r1_weights, 0.5)
        elif r1_vals:
            med_r1 = empirical_quantile(r1_vals, 0.5)
        else:
            med_r1 = 0.0
        shift = r1_case - med_r1
        for h in hours_seq:
            for tau in LEVELS:
                shrunk_q[h][tau] += shift

    else:
        # Fallback: hour-1 anchor (v1) with unlock-pattern weighting, no a_case similarity
        anchor_log = math.log(p)
        ratios_by_hour = {h: [] for h in hours_seq}
        weights_by_hour = {h: [] for h in hours_seq}

        for comp in case.get("comparables", []):
            built = build_comp_maps(comp)
            if built is None:
                continue
            ds_c, hmap = built
            v1_c = get_viewer(hmap, ds_c)
            if v1_c is None:
                continue
            lv1 = math.log(v1_c)
            w = unlock_weight(comp.get("unlock"))
            for h in hours_seq:
                th = ds_c + timedelta(hours=h - 1)
                vh = get_viewer(hmap, th)
                if vh is not None:
                    ratios_by_hour[h].append(math.log(vh) - lv1)
                    weights_by_hour[h].append(w)

        raw_q = {h: {} for h in hours_seq}
        for h in hours_seq:
            vals = ratios_by_hour[h]
            ws = weights_by_hour[h]
            if not vals:
                sigma_h = 0.15 * math.sqrt(h - 1)
                for tau in LEVELS:
                    raw_q[h][tau] = Z[tau] * sigma_h
            else:
                for tau in LEVELS:
                    raw_q[h][tau] = weighted_quantile(vals, ws, tau)

        shrunk_q = {h: {} for h in hours_seq}
        for h in hours_seq:
            n = len(ratios_by_hour[h])
            w = n / (n + K_SHRINK) if n > 0 else 0.0
            med = raw_q[h][0.5]
            sigma_h = 0.15 * math.sqrt(h - 1)
            for tau in LEVELS:
                if tau == 0.5:
                    val = med
                else:
                    val = med + w * (raw_q[h][tau] - med)
                    z = Z[tau]
                    cap = abs(z) * sigma_h * SPREAD_CAP_FACTOR
                    delta = val - med
                    if delta > cap:
                        delta = cap
                    elif delta < -cap:
                        delta = -cap
                    val = med + delta
                shrunk_q[h][tau] = val
            tvals = [shrunk_q[h][tau] for tau in LEVELS]
            for i in range(1, len(tvals)):
                if tvals[i] < tvals[i - 1]:
                    tvals[i] = tvals[i - 1]
            for tau, v in zip(LEVELS, tvals):
                shrunk_q[h][tau] = v

        med_series = [shrunk_q[h][0.5] for h in hours_seq]
        med_smooth = smooth_centered(med_series, SMOOTH_WINDOW)
        for h, m in zip(hours_seq, med_smooth):
            shift = m - shrunk_q[h][0.5]
            for tau in LEVELS:
                shrunk_q[h][tau] += shift

    # Build output rows, enforce constraints
    final = []
    for h in hours_seq:
        q_vals = []
        for tau in LEVELS:
            logq = anchor_log + shrunk_q[h][tau]
            qv = math.exp(logq)
            if not math.isfinite(qv):
                sigma_h = 0.15 * math.sqrt(h - 1)
                qv = p * math.exp(Z[tau] * sigma_h)
            if qv <= 0:
                qv = 1e-9
            if qv > 1e9:
                qv = 1e9
            q_vals.append(qv)
        # enforce non-decreasing within row
        for i in range(1, len(q_vals)):
            if q_vals[i] < q_vals[i - 1]:
                q_vals[i] = q_vals[i - 1]
        # clamp and re-fix monotonicity
        for i in range(len(q_vals)):
            if q_vals[i] <= 0:
                q_vals[i] = 1e-9
            if q_vals[i] > 1e9:
                q_vals[i] = 1e9
        for i in range(1, len(q_vals)):
            if q_vals[i] < q_vals[i - 1]:
                q_vals[i] = q_vals[i - 1]
        # round to keep compact; re-fix monotonicity after rounding
        q_vals = [round(v, 6) for v in q_vals]
        for i in range(1, len(q_vals)):
            if q_vals[i] < q_vals[i - 1]:
                q_vals[i] = q_vals[i - 1]
        final.append({"hour": h, "q": q_vals})

    return {"hours": final}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = sorted(INPUT_DIR.glob("*.in"))
    for path in paths:
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
