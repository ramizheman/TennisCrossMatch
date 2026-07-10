"""
Cross-match additive consistency test — Level 1.

For each test question, runs the cross-match engine across all rivalry matches,
then verifies:

  (A) REDUCER CONSISTENCY  — reduced.metric[m][p] == Σ iro.metric[m][p]  for all iros
  (B) PER-MATCH GROUND TRUTH — the US Open 2025 F MatchIRO matches known values
      (derived from the 25/25 single-match comparison already validated)
  (C) GROUPED CONSISTENCY   — same algebraic check on every grouped dimension
  (D) PATTERN CONSISTENCY   — pattern.combined == Σ iro.pattern.combined for each sig
  (E) SCOPE COVERAGE        — the correct 20 rivalry matches are loaded

All checks are deterministic because CrossMatchEngine uses plan-reuse across matches.
"""
import os, sys
from dotenv import load_dotenv
load_dotenv()
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from crossmatch import CrossMatchEngine, find_matches, build_match_iro
from crossmatch.iro import Stat, MetricResult, ReducedResult, MatchIRO
from crossmatch.reducer import reduce_iros

SINNER  = "Jannik Sinner"
ALCARAZ = "Carlos Alcaraz"

# Known ground truth for the single match (validated 25/25 in _compare_single_match.py)
USOPEN_GT = {
    # (metric, player) -> expected count
    ("points_won",     SINNER):  89,
    ("points_won",     ALCARAZ): 112,
    ("total_points",   None):    201,
    ("winners",        ALCARAZ): 25,
    ("winners",        SINNER):  15,
    ("unforced_errors", SINNER): 30,
    ("unforced_errors", ALCARAZ):32,
    ("double_faults",  SINNER):   4,
    ("aces",           ALCARAZ): 12,
    ("aces",           SINNER):   2,
    # grouped: shot_type
    ("grouped:shot_type:forehand:winners",   ALCARAZ): 20,
    ("grouped:shot_type:backhand:winners",   ALCARAZ):  5,
    ("grouped:shot_type:forehand:winners",   SINNER):   9,
    ("grouped:shot_type:backhand:winners",   SINNER):   6,
    ("grouped:shot_type:forehand:unforced_errors", SINNER):  20,
    ("grouped:shot_type:backhand:unforced_errors", SINNER):  10,
    ("grouped:shot_type:forehand:unforced_errors", ALCARAZ): 19,
    ("grouped:shot_type:backhand:unforced_errors", ALCARAZ): 13,
    # multistep patterns
    ("pattern:backhand crosscourt -> backhand crosscourt -> backhand down the line:total"): 5,
    ("pattern:backhand crosscourt -> backhand crosscourt -> backhand down the line", ALCARAZ): 3,
    ("pattern:backhand crosscourt -> backhand crosscourt -> backhand down the line", SINNER):  2,
    ("pattern:crosscourt -> crosscourt -> down the line:total"): 15,
    ("pattern:crosscourt -> crosscourt -> down the line", ALCARAZ): 8,
    ("pattern:crosscourt -> crosscourt -> down the line", SINNER):  7,
}

# ── helpers ───────────────────────────────────────────────────────────────────

def metric_count(obj, name, player=None):
    """Read count from a MatchIRO or ReducedResult."""
    mr = obj.metrics.get(name)
    if mr is None:
        return None
    if player:
        st = mr.by_player.get(player)
        return st.count if st else None
    return mr.combined.count

def grouped_count(obj, dim, met, bucket, player=None):
    gr = obj.grouped.get(dim)
    if not gr:
        return None
    bk = gr.buckets.get(bucket)
    if not bk:
        return None
    mr = bk.get(met)
    if not mr:
        return None
    if player:
        st = mr.by_player.get(player)
        return st.count if st else None
    return mr.combined.count

def pattern_count(obj, sig, player=None):
    pr = obj.patterns.get(sig)
    if pr is None:
        return None
    if player:
        st = pr.by_player.get(player)
        return st.occurrences if st else None
    return pr.combined.occurrences

def sum_metric_over_iros(iros, name, player):
    total = 0
    for iro in iros:
        mr = iro.metrics.get(name)
        if mr is None:
            continue
        if player:
            st = mr.by_player.get(player)
            total += st.count if st else 0
        else:
            total += mr.combined.count
    return total

def sum_grouped_over_iros(iros, dim, met, bucket, player):
    total = 0
    for iro in iros:
        gr = iro.grouped.get(dim)
        if not gr:
            continue
        bk = gr.buckets.get(bucket)
        if not bk:
            continue
        mr = bk.get(met)
        if not mr:
            continue
        if player:
            st = mr.by_player.get(player)
            total += st.count if st else 0
        else:
            total += mr.combined.count
    return total

def sum_pattern_over_iros(iros, sig, player):
    total = 0
    for iro in iros:
        pr = iro.patterns.get(sig)
        if pr is None:
            continue
        if player:
            st = pr.by_player.get(player)
            total += st.occurrences if st else 0
        else:
            total += pr.combined.occurrences
    return total

# ── test runner ───────────────────────────────────────────────────────────────

eng = CrossMatchEngine()
# Use eng.run() / eng.analyze() with EXPLICIT player scope so we get exactly
# the 20 rivalry matches without going through the scope planner (which would
# return database-wide scope for generic question strings).
refs = find_matches(players=[SINNER, ALCARAZ], require_all_players=True)
usopen_ref = next(r for r in refs if r.date == "2025-09-07")

PASSES = []
FAILURES = []

def record(label, expected, got):
    if expected is None:
        return  # nothing to check
    status = "OK" if got == expected else "FAIL"
    PASSES.append(label) if status == "OK" else FAILURES.append((label, expected, got))
    print(f"  {status:4} | {label:70} | exp={expected!r:>8}  got={got!r}")

print(f"\n{'='*100}")
print("CROSS-MATCH ADDITIVE CONSISTENCY TEST — Sinner vs Alcaraz rivalry (20 matches)")
print(f"{'='*100}")

# ── TEST QUESTIONS ────────────────────────────────────────────────────────────
TEST_QUESTIONS = [
    {
        "q": "How many aces did each player hit?",
        "metric_checks": [
            ("aces", ALCARAZ),
            ("aces", SINNER),
        ],
    },
    {
        "q": "How many winners did each player hit?",
        "metric_checks": [
            ("winners", ALCARAZ),
            ("winners", SINNER),
        ],
    },
    {
        "q": "How many unforced errors did each player make?",
        "metric_checks": [
            ("unforced_errors", ALCARAZ),
            ("unforced_errors", SINNER),
        ],
    },
    {
        "q": "How many double faults did each player commit?",
        "metric_checks": [
            ("double_faults", SINNER),
            ("double_faults", ALCARAZ),
        ],
    },
    {
        "q": "How many points did each player win?",
        "metric_checks": [
            ("points_won", ALCARAZ),
            ("points_won", SINNER),
            ("total_points", None),
        ],
    },
    {
        "q": "How many forehand vs backhand winners did each player hit?",
        "grouped_checks": [
            ("shot_type", "winners", "forehand", ALCARAZ),
            ("shot_type", "winners", "backhand", ALCARAZ),
            ("shot_type", "winners", "forehand", SINNER),
            ("shot_type", "winners", "backhand", SINNER),
        ],
    },
    {
        "q": "What was each player's unforced error count by shot type?",
        "grouped_checks": [
            ("shot_type", "unforced_errors", "forehand", SINNER),
            ("shot_type", "unforced_errors", "backhand", SINNER),
            ("shot_type", "unforced_errors", "forehand", ALCARAZ),
            ("shot_type", "unforced_errors", "backhand", ALCARAZ),
        ],
    },
    {
        "q": "How many times did backhand crosscourt -> backhand crosscourt -> backhand down the line happen?",
        "pattern_checks": [
            ("backhand crosscourt -> backhand crosscourt -> backhand down the line", None),
            ("backhand crosscourt -> backhand crosscourt -> backhand down the line", ALCARAZ),
            ("backhand crosscourt -> backhand crosscourt -> backhand down the line", SINNER),
        ],
    },
    {
        "q": "How many times did crosscourt -> crosscourt -> down the line pattern happen?",
        "pattern_checks": [
            ("crosscourt -> crosscourt -> down the line", None),
            ("crosscourt -> crosscourt -> down the line", ALCARAZ),
            ("crosscourt -> crosscourt -> down the line", SINNER),
        ],
    },
]

for test in TEST_QUESTIONS:
    q = test["q"]
    print(f"\n{'─'*80}")
    print(f"Q: {q}")

    # Run cross-match directly on the known rivalry refs (bypass scope planner).
    # eng.analyze() uses parallel fan-out with plan reuse.
    reduced, iros = eng.analyze(q, refs, progress=True)

    n_matches = len(iros)
    print(f"  Loaded {n_matches} matches")

    # ── (E) scope coverage ────────────────────────────────────────────────────
    record(
        f"[scope] rivalry loaded {n_matches} matches (expect 20)",
        20, n_matches,
    )

    # ── (A) reducer consistency — metric channel ──────────────────────────────
    for (name, player) in test.get("metric_checks", []):
        reduced_val = metric_count(reduced, name, player)
        manual_sum  = sum_metric_over_iros(iros, name, player)
        record(f"[reducer-metric] {name}[{player or 'combined'}]  reduced==Σiros",
               manual_sum, reduced_val)

    # ── (A) reducer consistency — grouped channel ─────────────────────────────
    for (dim, met, bucket, player) in test.get("grouped_checks", []):
        reduced_val = grouped_count(reduced, dim, met, bucket, player)
        manual_sum  = sum_grouped_over_iros(iros, dim, met, bucket, player)
        record(f"[reducer-grouped] {dim}:{bucket}:{met}[{player}]  reduced==Σiros",
               manual_sum if manual_sum else None, reduced_val)

    # ── (A) reducer consistency — pattern channel ─────────────────────────────
    for (sig, player) in test.get("pattern_checks", []):
        reduced_val = pattern_count(reduced, sig, player)
        manual_sum  = sum_pattern_over_iros(iros, sig, player)
        record(f"[reducer-pattern] {sig[:45]}[{player or 'combined'}]  reduced==Σiros",
               manual_sum if manual_sum else None, reduced_val)

    # ── (B) per-match ground truth pinning — US Open 2025 F ──────────────────
    usopen_iro = next((iro for iro in iros if iro.provenance.date == "2025-09-07"), None)
    if usopen_iro is None:
        print(f"  WARN: US Open 2025 F IRO not found in iros list")
    else:
        for (name, player) in test.get("metric_checks", []):
            gt_key = (name, player)
            gt_val = USOPEN_GT.get(gt_key)
            if gt_val is None:
                continue
            got = metric_count(usopen_iro, name, player)
            record(f"[gt-usopen-metric] {name}[{player or 'combined'}]",
                   gt_val, got)

        for (dim, met, bucket, player) in test.get("grouped_checks", []):
            gt_key = (f"grouped:{dim}:{bucket}:{met}", player)
            gt_val = USOPEN_GT.get(gt_key)
            if gt_val is None:
                continue
            got = grouped_count(usopen_iro, dim, met, bucket, player)
            record(f"[gt-usopen-grouped] {dim}:{bucket}:{met}[{player}]",
                   gt_val, got)

        for (sig, player) in test.get("pattern_checks", []):
            gt_key = (f"pattern:{sig}:total",) if player is None else (f"pattern:{sig}", player)
            gt_val = USOPEN_GT.get(gt_key)
            if gt_val is None:
                continue
            got = pattern_count(usopen_iro, sig, player)
            record(f"[gt-usopen-pattern] {sig[:40]}[{player or 'combined'}]",
                   gt_val, got)

    # ── (C/D) print actual rivalry totals for inspection ─────────────────────
    for (name, player) in test.get("metric_checks", []):
        val = metric_count(reduced, name, player)
        print(f"  >> RIVALRY TOTAL  {name}[{player or 'combined'}] = {val}")

    for (dim, met, bucket, player) in test.get("grouped_checks", []):
        val = grouped_count(reduced, dim, met, bucket, player)
        print(f"  >> RIVALRY TOTAL  {dim}:{bucket}:{met}[{player}] = {val}")

    for (sig, player) in test.get("pattern_checks", []):
        val = pattern_count(reduced, sig, player)
        print(f"  >> RIVALRY TOTAL  {sig[:45]}[{player or 'combined'}] = {val}")

# ── summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*100}")
print(f"RESULT: {len(PASSES)} OK  /  {len(FAILURES)} FAIL  out of {len(PASSES)+len(FAILURES)}")
if FAILURES:
    print("\nFailed checks:")
    for label, exp, got in FAILURES:
        print(f"  {label}")
        print(f"    expected: {exp!r}")
        print(f"    got:      {got!r}")
print("="*100)
