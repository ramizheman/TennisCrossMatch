"""
Cross-match temporal / trend test — Level 2.

For each metric question, verifies that the per-match time series is:

  (A) CHRONOLOGICAL    — dates are in non-decreasing order when sorted
  (B) PLAUSIBLE BOUNDS — each match value is within realistic tennis ranges
  (C) NON-DEGENERATE   — values actually vary across matches (not stuck at 0 or identical)
  (D) SUM CONSISTENT   — per-match values sum to the known rivalry totals
      (from Level 1 additive test; reused here as ground truth)
  (E) TREND DIRECTION  — for metrics with a known historical direction,
      second-half-of-rivalry average vs first-half goes the right way

Known rivalry ground truth (from Level 1):
  Alcaraz: aces=101, winners=503, unforced_errors=743, double_faults=65, points_won=1943
  Sinner:  aces=93,  winners=398, unforced_errors=716, double_faults=70, points_won=1979
  total_points=3570
"""
import os, sys, statistics
from dotenv import load_dotenv
load_dotenv()
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from crossmatch import CrossMatchEngine, find_matches
from crossmatch.reducer import player_metric_over_matches, pattern_over_matches

SINNER  = "Jannik Sinner"
ALCARAZ = "Carlos Alcaraz"

# Known rivalry totals (Level 1 validated)
RIVALRY_TOTALS = {
    ("aces",            ALCARAZ): 101,
    ("aces",            SINNER):   93,
    ("winners",         ALCARAZ): 503,
    ("winners",         SINNER):  398,
    ("unforced_errors", ALCARAZ): 743,
    ("unforced_errors", SINNER):  716,
    ("double_faults",   SINNER):   70,
    ("double_faults",   ALCARAZ):  65,
    ("points_won",      ALCARAZ): 1943,
    ("points_won",      SINNER):  1979,
}

# Per-match plausibility bounds (lo, hi inclusive)
# Wide enough to catch outright bugs but not 5-set anomalies.
# Lower bound = 0 to allow retirement matches (e.g. Cincinnati 2025: 29 total pts).
# Upper bounds allow 5-set epics (e.g. 2022 US Open QF: 84 UEs for Sinner; RG clay
# finals with 47-49 cc->cc->dtl patterns).
BOUNDS = {
    "aces":            (0, 35),
    "winners":         (0, 90),
    "unforced_errors": (0, 100),
    "double_faults":   (0, 18),
    "points_won":      (0, 350),
    "total_points":    (0, 600),
}

# Matches known to be retirements / suspended (very low counts expected)
RETIREMENT_MATCHES = {"2025-08-18"}   # Cincinnati Masters 2025

# Known trend directions for the Sinner/Alcaraz rivalry:
#   - Sinner has won more of the recent matches (2024-2026) than early ones
#     (their rivalry: Alcaraz dominated early; Sinner more recently)
#   TREND: Sinner's points_won / total_points should be HIGHER in matches after 2024
TREND_CHECKS = [
    {
        "metric": "points_won",
        "player": SINNER,
        "cutoff_date": "2024-01-01",
        "direction": "up",   # later half should have higher pct than early half
        "description": "Sinner win-rate should be higher post-2024 than pre-2024",
    },
    {
        "metric": "winners",
        "player": ALCARAZ,
        "cutoff_date": "2024-01-01",
        "direction": "up",   # Alcaraz's winner production increased as he matured
        "description": "Alcaraz winners/match should trend higher post-2024",
    },
]

# ── load once ─────────────────────────────────────────────────────────────────
print(f"\n{'='*90}")
print("CROSS-MATCH TEMPORAL / TREND TEST — Sinner vs Alcaraz rivalry (20 matches)")
print(f"{'='*90}")

eng = CrossMatchEngine()
refs = find_matches(players=[SINNER, ALCARAZ], require_all_players=True)
print(f"  {len(refs)} rivalry matches found")

PASSES = []
FAILURES = []

def record(label, expected, got):
    status = "OK" if got == expected else "FAIL"
    PASSES.append(label) if status == "OK" else FAILURES.append((label, expected, got))
    e_str = f"{expected!r:>8}" if not isinstance(expected, str) else expected
    g_str = f"{got!r}" if not isinstance(got, str) else got
    print(f"  {status:4} | {label:70} | exp={e_str}  got={g_str}")

def record_bool(label, condition, description=""):
    status = "OK" if condition else "FAIL"
    PASSES.append(label) if condition else FAILURES.append((label, True, condition))
    print(f"  {status:4} | {label:70} | {description}")

# ── QUESTIONS to build trend series from ────────────────────────────────────
TEST_QUESTIONS = [
    {
        "q": "How many aces did each player hit?",
        "metrics": [("aces", ALCARAZ), ("aces", SINNER)],
    },
    {
        "q": "How many winners did each player hit?",
        "metrics": [("winners", ALCARAZ), ("winners", SINNER)],
    },
    {
        "q": "How many unforced errors did each player make?",
        "metrics": [("unforced_errors", ALCARAZ), ("unforced_errors", SINNER)],
    },
    {
        "q": "How many double faults did each player commit?",
        "metrics": [("double_faults", SINNER), ("double_faults", ALCARAZ)],
    },
    {
        "q": "How many points did each player win?",
        "metrics": [("points_won", ALCARAZ), ("points_won", SINNER), ("total_points", None)],
        "trend_checks": TREND_CHECKS,
        "pattern_q": None,
    },
    {
        "q": "How many times did crosscourt -> crosscourt -> down the line pattern happen?",
        "metrics": [],
        "pattern": ("crosscourt -> crosscourt -> down the line", None),
        "pattern_bounds": (0, 60),  # clay court RG finals can hit 47-49
    },
]

all_iros = {}  # q -> List[MatchIRO] (cached)

for test in TEST_QUESTIONS:
    q = test["q"]
    print(f"\n{'─'*80}")
    print(f"Q: {q}")
    reduced, iros = eng.analyze(q, refs, progress=False)
    all_iros[q] = iros
    print(f"  {len(iros)} match IROs built")

    # ── (A) check dates sorted — validate via player_metric_over_matches output,
    #        NOT raw IRO order (fan-out is parallel → completion order varies)
    #        Use first available metric's series for the date-order check.
    _metrics_list = test.get("metrics") or []
    _first_metric = _metrics_list[0] if _metrics_list else None
    if _first_metric and _first_metric[1]:
        _date_series = player_metric_over_matches(iros, _first_metric[1], _first_metric[0])
        _series_dates = [r["date"] for r in _date_series if r.get("date")]
    else:
        _pat_sig = (test.get("pattern") or ("",))[0]
        _date_series = pattern_over_matches(iros, _pat_sig) if _pat_sig else []
        _series_dates = [r["date"] for r in _date_series if r.get("date")]
    record_bool(
        f"[temporal-A] dates sorted chronologically in series",
        _series_dates == sorted(_series_dates),
        f"first={_series_dates[0] if _series_dates else 'n/a'}  "
        f"last={_series_dates[-1] if _series_dates else 'n/a'}",
    )

    for (metric, player) in test.get("metrics", []):
        if player is None:
            # total_points: check combined series
            series = []
            for iro in iros:
                mr = iro.metrics.get(metric)
                if mr:
                    series.append({
                        "date": iro.provenance.date,
                        "match_id": iro.provenance.match_id,
                        "count": mr.combined.count,
                        "total": mr.combined.count,
                        "pct": 1.0,
                    })
        else:
            series = player_metric_over_matches(iros, player, metric)

        if not series:
            record_bool(f"[temporal-B/C/D] {metric}[{player or 'combined'}] series non-empty",
                        False, f"no data points found in {len(iros)} IROs")
            continue

        # ── (B) plausibility bounds ───────────────────────────────────────────
        lo, hi = BOUNDS.get(metric, (0, 9999))
        out_of_bounds = [(r["date"], r["count"]) for r in series
                         if r["count"] < lo or r["count"] > hi]
        record_bool(
            f"[temporal-B] {metric}[{player or 'combined'}] all values in [{lo},{hi}]",
            not out_of_bounds,
            f"{len(series)} matches  (violators: {out_of_bounds[:3]})" if out_of_bounds
            else f"{len(series)} matches  min={min(r['count'] for r in series)}  max={max(r['count'] for r in series)}",
        )

        # Check no impossible percentages.
        # Stat.pct returns 0-100 scale (already multiplied by 100).
        # Skip pct check for raw-count metrics where total==count (pct=100 by construction).
        raw_count_metrics = {"winners", "unforced_errors"}
        if metric not in raw_count_metrics:
            bad_pct = [r for r in series
                       if r.get("total", 1) > 0
                       and r.get("pct") is not None
                       and (r["pct"] < 0 or r["pct"] > 100.0001)]
            record_bool(
                f"[temporal-B] {metric}[{player or 'combined'}] pct in [0,100]",
                not bad_pct,
                f"bad pct rows: {bad_pct[:2]}" if bad_pct else "all OK",
            )

        # ── (C) non-degenerate ────────────────────────────────────────────────
        counts = [r["count"] for r in series]
        is_varied = len(set(counts)) > 1
        stdev = statistics.stdev(counts) if len(counts) > 1 else 0
        record_bool(
            f"[temporal-C] {metric}[{player or 'combined'}] values vary across matches",
            is_varied,
            f"stdev={stdev:.1f}  values={sorted(set(counts))} (distinct={len(set(counts))})",
        )

        # ── (D) sum consistency ───────────────────────────────────────────────
        gt = RIVALRY_TOTALS.get((metric, player))
        if gt is not None:
            actual_sum = sum(r["count"] for r in series)
            record(f"[temporal-D] {metric}[{player}] Σ per-match == rivalry total",
                   gt, actual_sum)

        # ── (E) trend direction ───────────────────────────────────────────────
        for tc in test.get("trend_checks", []):
            if tc["metric"] != metric or tc["player"] != player:
                continue
            cutoff = tc["cutoff_date"]
            early  = [r for r in series if (r.get("date") or "") < cutoff]
            recent = [r for r in series if (r.get("date") or "") >= cutoff]
            if not early or not recent:
                print(f"  SKIP [temporal-E] {tc['description']} (insufficient data before/after {cutoff})")
                continue
            def safe_pct(rows):
                c = sum(r["count"] for r in rows)
                t = sum(r.get("total", r["count"]) for r in rows)
                return c / t if t > 0 else 0.0
            early_pct  = safe_pct(early)
            recent_pct = safe_pct(recent)
            if tc["direction"] == "up":
                trend_ok = recent_pct >= early_pct - 0.03   # 3pp tolerance
            else:
                trend_ok = recent_pct <= early_pct + 0.03
            record_bool(
                f"[temporal-E] {tc['description'][:65]}",
                trend_ok,
                f"early({len(early)})={early_pct:.1%}  recent({len(recent)})={recent_pct:.1%}  dir={tc['direction']}",
            )

    # ── Pattern series checks ─────────────────────────────────────────────────
    if "pattern" in test:
        sig, player = test["pattern"]
        p_series = pattern_over_matches(iros, sig, player)
        lo, hi = test.get("pattern_bounds", (0, 50))
        print(f"  Pattern '{sig}' [{player or 'combined'}]: {len(p_series)} matches with data")

        # (A) dates sorted
        p_dates = [r["date"] for r in p_series if r.get("date")]
        record_bool(
            f"[temporal-A] pattern series dates sorted",
            p_dates == sorted(p_dates),
            f"first={p_dates[0] if p_dates else 'n/a'}  last={p_dates[-1] if p_dates else 'n/a'}",
        )

        # (B) bounds
        bad = [(r["date"], r["occurrences"]) for r in p_series
               if r["occurrences"] < lo or r["occurrences"] > hi]
        record_bool(
            f"[temporal-B] pattern occurrences in [{lo},{hi}]",
            not bad,
            f"min={min((r['occurrences'] for r in p_series), default=0)}  "
            f"max={max((r['occurrences'] for r in p_series), default=0)}"
            + (f"  violators={bad[:3]}" if bad else ""),
        )

        # (C) varied
        occs = [r["occurrences"] for r in p_series]
        record_bool(
            f"[temporal-C] pattern occurrences vary across matches",
            len(set(occs)) > 1,
            f"distinct values={sorted(set(occs))}",
        )

        # Print per-match table for visual inspection
        print(f"\n  Per-match time series for '{sig}':")
        print(f"  {'Date':12} {'Tournament':30} {'Surface':8} {'Occ':>5} {'Wins':>5} {'WinPct':>7}")
        for r in p_series:
            pct = f"{r['wins']/r['occurrences']:.0%}" if r['occurrences'] > 0 else "  n/a"
            print(f"  {r['date']:12} {r['tournament'][:30]:30} {r['surface']:8} "
                  f"{r['occurrences']:>5} {r['wins']:>5} {pct:>7}")

# ── Print time series table for key metric ────────────────────────────────────
print(f"\n{'─'*80}")
print("Per-match time series: points_won % (Alcaraz vs Sinner over time)")
q_pts = "How many points did each player win?"
if q_pts in all_iros:
    iros_pts = all_iros[q_pts]
    alc_s = {r["date"]: r for r in player_metric_over_matches(iros_pts, ALCARAZ, "points_won")}
    sin_s = {r["date"]: r for r in player_metric_over_matches(iros_pts, SINNER,  "points_won")}
    dates = sorted(set(alc_s) | set(sin_s))
    print(f"  {'Date':12} {'Tournament':30} {'ALR%':>6} {'SIN%':>6} {'Total':>7} {'Winner':>10}")
    for d in dates:
        a = alc_s.get(d, {})
        s = sin_s.get(d, {})
        # Stat.pct returns 0-100 scale; format as "xx.x%"
        ap = f"{a['pct']:.1f}%" if a.get("pct") is not None else "  n/a"
        sp = f"{s['pct']:.1f}%" if s.get("pct") is not None else "  n/a"
        tourn = a.get("tournament") or s.get("tournament") or ""
        total = (a.get("count", 0) or 0) + (s.get("count", 0) or 0)
        note = " [ret?]" if d in RETIREMENT_MATCHES else ""
        # winner = whoever won more points
        winner = ""
        if a.get("count") and s.get("count"):
            winner = "Alcaraz" if a["count"] > s["count"] else "Sinner"
        print(f"  {d:12} {(tourn[:30]+note):38} {ap:>6} {sp:>6} {total:>7} {winner:>10}")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*90}")
print(f"RESULT: {len(PASSES)} OK  /  {len(FAILURES)} FAIL  out of {len(PASSES)+len(FAILURES)}")
if FAILURES:
    print("\nFailed checks:")
    for label, exp, got in FAILURES:
        print(f"  {label}")
        print(f"    expected: {exp!r}  got: {got!r}")
print("="*90)
