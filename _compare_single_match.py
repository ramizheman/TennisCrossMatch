"""
Cross-match engine vs single-match ground truth comparison.

Match: US Open 2025 F – Jannik Sinner vs Carlos Alcaraz (2025-09-07)
Ground truth sourced from:
  test_50exec_20260410_115659.md
  test_conversation_fixes_20260410_150454.md

For each question the cross-match engine is loaded to that single match
and run through analyze_question_structured -> build_match_iro. We then
pull the relevant metric / pattern / grouped count and compare.
"""
import os, sys
from dotenv import load_dotenv
load_dotenv()
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from crossmatch import CrossMatchEngine, build_match_iro, find_matches

# ── load the single match ────────────────────────────────────────────────────
eng = CrossMatchEngine()
ref = next(r for r in find_matches(players=["Jannik Sinner", "Carlos Alcaraz"],
                                   require_all_players=True)
           if r.date == "2025-09-07")
eng.load_match(ref)
SINNER  = "Jannik Sinner"
ALCARAZ = "Carlos Alcaraz"

def run(q):
    s = eng.agent.analyze_question_structured(q)
    return build_match_iro(s, provenance_overrides=ref.provenance_overrides()), s

def metric(iro, name, player=None):
    mr = iro.metrics.get(name)
    if mr is None:
        return None
    if player:
        st = mr.by_player.get(player)
        return st.count if st else None
    return mr.combined.count

def pattern(iro, sig, player=None):
    pr = iro.patterns.get(sig)
    if pr is None:
        return None
    if player:
        st = pr.by_player.get(player)
        return st.occurrences if st else None
    return pr.combined.occurrences

def grouped_count(iro, dim, met, bucket, player=None):
    gr = iro.grouped.get(dim)
    if not gr:
        return None
    bk = gr.buckets.get(str(bucket))
    if not bk:
        return None
    mr = bk.get(met)
    if not mr:
        return None
    if player:
        st = mr.by_player.get(player)
        return st.count if st else None
    return mr.combined.count

def pct(iro, name, player=None):
    """Return pct field of a ratio metric."""
    mr = iro.metrics.get(name)
    if mr is None:
        return None
    if player:
        st = mr.by_player.get(player)
        return round(st.pct, 1) if st else None
    return round(mr.combined.pct, 1)

results = []

def check(label, q, expected, extractor):
    iro, _ = run(q)
    got = extractor(iro)
    status = "OK" if got == expected else "FAIL"
    results.append((status, label, expected, got))
    print(f"  {status:4} | {label:55} | expected={expected!r:>8}  got={got!r}")

print(f"\n{'='*90}")
print("CROSS-MATCH SINGLE-MATCH COMPARISON — US Open 2025 F (Sinner vs Alcaraz)")
print(f"{'='*90}")

# ── Section A: basic tree metrics ────────────────────────────────────────────
print("\n--- A: Basic tree metrics ---")
# Use the "each player" formulation — routes to player=both, giving the stable
# full-match terminal with player1_wins=89 / player2_wins=112 / total_points=201.
# _add_direct_terminal_metrics now reads these directly from the terminal node.
iro_pts, _ = run("How many points did each player win?")
results.append(("OK" if metric(iro_pts, "points_won", SINNER) == 89 else "FAIL",
                "Sinner points won", 89, metric(iro_pts, "points_won", SINNER)))
results.append(("OK" if metric(iro_pts, "points_won", ALCARAZ) == 112 else "FAIL",
                "Alcaraz points won", 112, metric(iro_pts, "points_won", ALCARAZ)))
results.append(("OK" if metric(iro_pts, "total_points") == 201 else "FAIL",
                "Total points (201)", 201, metric(iro_pts, "total_points")))
for _, label, expected, got in results[-3:]:
    status = "OK" if got == expected else "FAIL"
    print(f"  {status:4} | {label:55} | expected={expected!r:>8}  got={got!r}")
check("Alcaraz winners",         "How many winners did Alcaraz hit?",       25,  lambda i: metric(i, "winners", ALCARAZ))
check("Sinner winners",          "How many winners did Sinner hit?",        15,  lambda i: metric(i, "winners", SINNER))
check("Sinner unforced errors",  "How many unforced errors did Sinner make?", 30, lambda i: metric(i, "unforced_errors", SINNER))
check("Alcaraz unforced errors", "How many unforced errors did Alcaraz make?", 32, lambda i: metric(i, "unforced_errors", ALCARAZ))
check("Sinner double faults",    "How many double faults did Sinner commit?",  4, lambda i: metric(i, "double_faults", SINNER))
check("Alcaraz aces",            "How many aces did Alcaraz hit?",          12,  lambda i: metric(i, "aces", ALCARAZ))
check("Sinner aces",             "How many aces did Sinner hit?",            2,  lambda i: metric(i, "aces", SINNER))

# ── Section A: serve percentage (ratio metric) ───────────────────────────────
print("\n--- A: Serve percentage ---")
check("Alcaraz 1st serve win% (45/54 = 83.3%)",
      "What was Alcaraz's 1st serve win percentage?",
      83.3,
      lambda i: pct(i, "first_serve_win_pct", ALCARAZ))

# ── Section B/SH: grouped / shot-type breakdowns ─────────────────────────────
print("\n--- B/SH: Winners by shot type (grouped) ---")
iro_w, _ = run("How many forehand vs backhand winners did each player hit?")
check_items = [
    ("Alcaraz FH winners (20)",   ALCARAZ, "forehand", 20),
    ("Alcaraz BH winners (5)",    ALCARAZ, "backhand",  5),
    ("Sinner FH winners (9)",     SINNER,  "forehand",  9),
    ("Sinner BH winners (6)",     SINNER,  "backhand",  6),
]
for label, player, bucket, expected in check_items:
    got = grouped_count(iro_w, "shot_type", "winners", bucket, player)
    status = "OK" if got == expected else "FAIL"
    results.append((status, label, expected, got))
    print(f"  {status:4} | {label:55} | expected={expected!r:>8}  got={got!r}")

print("\n--- SH: UEs by shot type (grouped) ---")
iro_ue, _ = run("What was each player's unforced error count by shot type?")
ue_items = [
    ("Sinner FH UEs (20)",    SINNER,  "forehand", 20),
    ("Sinner BH UEs (10)",    SINNER,  "backhand", 10),
    ("Alcaraz FH UEs (19)",   ALCARAZ, "forehand", 19),
    ("Alcaraz BH UEs (13)",   ALCARAZ, "backhand", 13),
]
for label, player, bucket, expected in ue_items:
    got = grouped_count(iro_ue, "shot_type", "unforced_errors", bucket, player)
    status = "OK" if got == expected else "FAIL"
    results.append((status, label, expected, got))
    print(f"  {status:4} | {label:55} | expected={expected!r:>8}  got={got!r}")

# ── Section F: shot patterns ─────────────────────────────────────────────────
print("\n--- F: Shot patterns (multistep) ---")
F05_Q = "How many times did backhand crosscourt -> backhand crosscourt -> backhand down the line happen?"
iro_f05, _ = run(F05_Q)
f05_sigs = list(iro_f05.patterns.keys())
print(f"  (F05 patterns found: {f05_sigs})")
check("F05: bh cc->bh cc->bh dtl total (5)",
      F05_Q, 5,
      lambda i: sum(pr.combined.occurrences for pr in i.patterns.values()))
check("F05: Alcaraz executions (3)",
      F05_Q, 3,
      lambda i: (lambda prs: sum(pr.by_player.get(ALCARAZ, type('',(),{'occurrences':0})()).occurrences
                                  for pr in prs.values()))(i.patterns))
check("F05: Sinner executions (2)",
      F05_Q, 2,
      lambda i: (lambda prs: sum(pr.by_player.get(SINNER, type('',(),{'occurrences':0})()).occurrences
                                  for pr in prs.values()))(i.patterns))

F06_Q = "How many times did crosscourt -> crosscourt -> down the line pattern happen?"
iro_f06, _ = run(F06_Q)
f06_sigs = list(iro_f06.patterns.keys())
print(f"  (F06 patterns found: {f06_sigs})")
check("F06: cc->cc->dtl total (15)",
      F06_Q, 15,
      lambda i: sum(pr.combined.occurrences for pr in i.patterns.values()))
check("F06: Alcaraz executions (8)",
      F06_Q, 8,
      lambda i: (lambda prs: sum(pr.by_player.get(ALCARAZ, type('',(),{'occurrences':0})()).occurrences
                                  for pr in prs.values()))(i.patterns))
check("F06: Sinner executions (7)",
      F06_Q, 7,
      lambda i: (lambda prs: sum(pr.by_player.get(SINNER, type('',(),{'occurrences':0})()).occurrences
                                  for pr in prs.values()))(i.patterns))

# ── Summary ───────────────────────────────────────────────────────────────────
n_ok   = sum(1 for s,*_ in results if s == "OK")
n_fail = sum(1 for s,*_ in results if s == "FAIL")
print(f"\n{'='*90}")
print(f"RESULT: {n_ok} OK  /  {n_fail} FAIL  out of {len(results)}")
if n_fail:
    print("\nFailed checks:")
    for s, label, exp, got in results:
        if s == "FAIL":
            print(f"  {label:55} expected={exp!r}  got={got!r}")
print("="*90)
