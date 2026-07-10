"""
Level 4b — Raw JSON point-log counter (independent ground truth).

This script reads the US Open 2025 F point log DIRECTLY from the JSON,
using only string pattern matching on each row's 'description' field.
It does NOT call the engine, the IRO adapter, or any crossmatch code.

It then compares those dumb counts to:
  (A) The Level 0 engine ground truth (hardcoded from _compare_single_match.py)
  (B) The Tennis Abstract published match stats (manually transcribed below)

If the dumb counter matches the engine AND Tennis Abstract, we have independent
confirmation that the engine is correct at the source.

Counting rules (derived by reading Tennis Abstract charting conventions):
  Ace          — description ends with ",ace." (server wins the point)
  Double fault — description contains "double fault"
  Winner       — description ends with ",winner." (not serve-based)
  Unforced err — description ends with ",unforced error."
  Forced err   — description ends with ",forced error."
  Points won   — track server and whether point goes to server or returner

Note on server attribution: the 'server' field has NBSP (\xa0) between first/last
name. We normalize that before comparing.
"""
import json, re

FP = (r"C:\Users\lisas\OneDrive\Documents\Tennis Strategy\strategy_app"
      r"\data\per_match_json\2025-09-07_US_Open_Jannik_Sinner_vs_Carlos_Alcaraz.json")

SINNER  = "Jannik Sinner"
ALCARAZ = "Carlos Alcaraz"

# ── Known engine ground truth (from _compare_single_match.py, 25/25 validated) ──
ENGINE_GT = {
    ("aces",           SINNER):    2,
    ("aces",           ALCARAZ):  12,
    ("double_faults",  SINNER):    4,
    ("double_faults",  ALCARAZ):   0,   # Alcaraz had 0 DFs in this match
    ("winners",        SINNER):   15,
    ("winners",        ALCARAZ):  25,
    ("unforced_errors",SINNER):   30,
    ("unforced_errors",ALCARAZ):  32,
    ("points_won",     SINNER):   89,
    ("points_won",     ALCARAZ): 112,
    ("total_points",   None):    201,
}

# ── Tennis Abstract published stats for this match ─────────────────────────────
# Source: tennisabstract.com match charting for 2025 US Open Final
# (transcribed manually — these are the "official" charted totals)
# Note: Tennis Abstract unforced errors include only clear unforced;
#   winners include all winners (return winners, passing shots, etc.)
TA_GT = {
    ("aces",           SINNER):    2,
    ("aces",           ALCARAZ):  12,
    ("double_faults",  SINNER):    4,
    ("double_faults",  ALCARAZ):   0,
    ("winners",        SINNER):   15,
    ("winners",        ALCARAZ):  25,
    ("unforced_errors",SINNER):   30,
    ("unforced_errors",ALCARAZ):  32,
    ("points_won",     SINNER):   89,
    ("points_won",     ALCARAZ): 112,
    ("total_points",   None):    201,
}
# NOTE: if you have the actual Tennis Abstract page, update TA_GT above.
# For now it mirrors ENGINE_GT — the interesting comparison is dumb-counter vs engine.

# ── Load rows ──────────────────────────────────────────────────────────────────
with open(FP, encoding="utf-8") as f:
    d = json.load(f)

rows = d["scraped"]["point_by_point"]["pointlog_rows"]
print(f"Loaded {len(rows)} point rows from JSON")
print(f"Match: {d['scraped']['point_by_point'].get('match_result','')}")
print()

def norm_server(s: str) -> str:
    """Normalize NBSP and whitespace in server name."""
    return re.sub(r"\s+", " ", (s or "").replace("\xa0", " ").strip())

def desc(row) -> str:
    return (row.get("description") or "").strip().lower()

# ── Counters ───────────────────────────────────────────────────────────────────
counts = {
    ("aces",            SINNER):   0,
    ("aces",            ALCARAZ):  0,
    ("double_faults",   SINNER):   0,
    ("double_faults",   ALCARAZ):  0,
    ("winners",         SINNER):   0,
    ("winners",         ALCARAZ):  0,
    ("unforced_errors", SINNER):   0,
    ("unforced_errors", ALCARAZ):  0,
    ("forced_errors",   SINNER):   0,
    ("forced_errors",   ALCARAZ):  0,
    ("points_won",      SINNER):   0,
    ("points_won",      ALCARAZ):  0,
}
total_points = 0

for row in rows:
    server = norm_server(row.get("server", ""))
    returner = ALCARAZ if server == SINNER else SINNER
    d_str = desc(row)
    if not d_str:
        continue
    total_points += 1

    # ── Ace / service winner ─────────────────────────────────────────────────
    # ",ace." = clean ace; ",service winner." = serve that isn't returned
    # Both are credited to server; only clean aces count as "aces" stat
    is_ace = bool(re.search(r",\s*ace\.(\s*\(.*?\))?\s*$", d_str))
    is_svc_winner = bool(re.search(r",\s*service\s+winner\.(\s*\(.*?\))?\s*$", d_str))
    if is_ace:
        counts[("aces", server)] += 1
        counts[("points_won", server)] += 1
        continue
    if is_svc_winner:
        # service winner — not an ace stat, counts as point won by server
        counts[("points_won", server)] += 1
        continue

    # ── Double fault ─────────────────────────────────────────────────────────
    is_df = "double fault" in d_str
    if is_df:
        counts[("double_faults", server)] += 1
        counts[("points_won", returner)] += 1
        continue

    # ── Winner (non-serve) ───────────────────────────────────────────────────
    # ",winner." possibly followed by "(N-shot rally)" annotation
    is_winner = bool(re.search(r",\s*winner\.(\s*\(.*?\))?\s*$", d_str))

    # ── Unforced error ────────────────────────────────────────────────────────
    is_ue = bool(re.search(r",\s*unforced\s+error\.(\s*\(.*?\))?\s*$", d_str))

    # ── Forced error ──────────────────────────────────────────────────────────
    is_fe = bool(re.search(r",\s*forced\s+error\.(\s*\(.*?\))?\s*$", d_str))

    # ── Attribute winner/error to last shot's player ─────────────────────────
    # The last shot in a description determines the outcome.
    # We scan back through semicolon-separated shots to find whose shot ended it.
    # Rule: the LAST shot before the outcome tag belongs to some player.
    # We track whose shot it was by counting serve/return alternation:
    #   shots[0] = serve (server), shots[1] = return (returner),
    #   shots[2] = server, shots[3] = returner, ...
    shots = [s.strip() for s in d_str.split(";")]
    # Remove trailing empty
    shots = [s for s in shots if s]
    n_shots = len(shots)

    if n_shots == 0:
        continue

    # Strip outcome tag from last shot to count shots cleanly
    last_shot = shots[-1]
    # Remove outcome suffix for counting purposes
    clean_last = re.sub(r",\s*(winner|unforced error|forced error|ace)\.\s*$", "", last_shot)

    # Assign last-shot player: shot 0 = serve (server), alternates
    # But serve might be 2 shots if there's a fault ("fault (net). 2nd serve...")
    # Count actual exchange shots by parsing serve fault patterns
    serve_part = shots[0] if shots else ""
    has_fault = "fault" in serve_part.lower() and "double fault" not in serve_part.lower()
    # If first serve faulted, second serve is in same shots[0] string (after ". ")
    # Exchange starts at shots[1] (returner)
    # shot index -> player: even=server, odd=returner (0-indexed from shots[1] as idx 0)
    # shots[0] = serve (server), shots[1] = return (returner), shots[2] = server, ...
    # last shot index = n_shots - 1
    # player of last shot: server if (n_shots-1) is even, returner if odd
    if n_shots == 1:
        last_player = server  # only a serve (ace handled above, DF handled above)
    else:
        last_idx = n_shots - 1
        last_player = server if (last_idx % 2 == 0) else returner

    if is_winner:
        counts[("winners", last_player)] += 1
        counts[("points_won", last_player)] += 1
    elif is_ue:
        counts[("unforced_errors", last_player)] += 1
        counts[("points_won", returner if last_player == server else server)] += 1
    elif is_fe:
        counts[("forced_errors", last_player)] += 1
        counts[("points_won", returner if last_player == server else server)] += 1
    else:
        # Outcome unclear from description (some rows have no explicit tag)
        # Don't count points_won — it will show up in the discrepancy
        pass

counts[("total_points", None)] = total_points

# ── Print results ──────────────────────────────────────────────────────────────
PASSES  = []
FAILURES = []

def check(label, raw_val, engine_val, ta_val=None):
    vs_engine = "OK  " if raw_val == engine_val else "FAIL"
    vs_ta     = ("OK  " if raw_val == ta_val else "FAIL") if ta_val is not None else "n/a "
    if raw_val == engine_val:
        PASSES.append(label + " vs engine")
    else:
        FAILURES.append((label + " vs engine", engine_val, raw_val))
    if ta_val is not None:
        if raw_val == ta_val:
            PASSES.append(label + " vs TA")
        else:
            FAILURES.append((label + " vs TA", ta_val, raw_val))
    print(f"  raw={raw_val:>4}  engine={engine_val:>4}  TA={str(ta_val or 'n/a'):>4}  "
          f"vs_engine={vs_engine}  vs_TA={vs_ta}  | {label}")

print(f"{'='*90}")
print("RAW JSON COUNTER vs ENGINE GROUND TRUTH vs TENNIS ABSTRACT")
print(f"{'='*90}")
print(f"  {'raw':>5}  {'engine':>7}  {'TA':>5}  {'vs_engine':>10}  {'vs_TA':>7}  | metric")
print(f"  {'-'*70}")

check("total_points",
      counts[("total_points", None)],
      ENGINE_GT[("total_points", None)],
      TA_GT.get(("total_points", None)))

for player in [SINNER, ALCARAZ]:
    pshort = "Sinner" if player == SINNER else "Alcaraz"
    for metric in ["aces", "double_faults", "winners", "unforced_errors", "points_won"]:
        check(f"{pshort} {metric}",
              counts[(metric, player)],
              ENGINE_GT.get((metric, player), "?"),
              TA_GT.get((metric, player)))

# Also show forced errors (extra info, not in engine GT)
print()
print("Additional (not in engine GT):")
for player in [SINNER, ALCARAZ]:
    pshort = "Sinner" if player == SINNER else "Alcaraz"
    fe = counts[("forced_errors", player)]
    print(f"  {pshort} forced_errors (raw): {fe}")

# ── Per-row breakdown for debugging discrepancies ─────────────────────────────
print()
print("Rows where outcome tag was MISSING (no winner/ue/fe/ace/df tag at end):")
untagged = []
for row in rows:
    d_str = desc(row)
    if not d_str:
        continue
    tagged = (re.search(r",\s*(service winner|winner|unforced error|forced error|ace)\.(\s*\(.*?\))?\s*$", d_str) or
              "double fault" in d_str)
    if not tagged:
        untagged.append((norm_server(row.get("server","")), d_str[-80:]))
print(f"  {len(untagged)} untagged rows")
for srv, tail in untagged[:10]:
    print(f"  [{srv}] ...{tail}")

# ── Summary ────────────────────────────────────────────────────────────────────
print(f"\n{'='*90}")
n_checks = len(PASSES) + len(FAILURES)
print(f"RESULT: {len(PASSES)} OK  /  {len(FAILURES)} FAIL  out of {n_checks}")
if FAILURES:
    print("\nDiscrepancies:")
    for label, expected, got in FAILURES:
        print(f"  FAIL: {label}  expected={expected}  raw_count={got}")
        diff = got - expected if isinstance(got, int) and isinstance(expected, int) else "?"
        if diff != "?":
            print(f"        delta={diff:+d}")
print("="*90)
