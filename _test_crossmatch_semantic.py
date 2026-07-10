"""
Cross-match semantic test — Level 3.

Verifies that the LLM narrator never invents numbers: every number cited in the
prose narrative must appear in the deterministic dossier that was handed to the LLM.

Algorithm
─────────
1. Run a question  →  ReducedResult + MatchIROs
2. Build the dossier (the only data the narrator is allowed to use)
3. Generate the narrative (LLM)
4. Extract every numeric token from the narrative
5. Build a "truth set" of every number in the dossier
6. For each narrative number, check membership in the truth set
   - integers: exact match OR ±1 (LLMs round)
   - percentages: ±0.15pp tolerance
   - very small numbers (1-5) and years (1900-2030) are skipped as incidental
7. Report: % traceable, any untraced numbers flagged as POTENTIAL HALLUCINATIONS

Pass criterion: ≥ 90% of "significant" narrative numbers are traceable.
"""
import os, sys, re, json, statistics
from dotenv import load_dotenv
load_dotenv()

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "agents"))

from crossmatch import CrossMatchEngine, find_matches
from crossmatch.narrator import build_dossier, narrate

SINNER  = "Jannik Sinner"
ALCARAZ = "Carlos Alcaraz"

PASS_THRESHOLD = 0.90   # 90% of significant numbers must be traceable


# ── Number extraction ─────────────────────────────────────────────────────────

def extract_narrative_numbers(text: str):
    """Return list of (kind, value, snippet) for every number token in text.
    kind in ('pct', 'int', 'float').
    snippet is the surrounding 30 characters for debugging.

    Pre-processing: ISO dates (YYYY-MM-DD) are stripped before extraction so
    month/day components aren't flagged as untraced integers."""
    results = []
    text_lower = text.lower()

    # Strip ISO dates like 2025-06-08 before tokenizing — their sub-parts
    # (month=06, day=08) are not analytical numbers from the dossier
    text = re.sub(r'\b\d{4}-\d{2}-\d{2}\b', 'DATE', text)

    # Percentages first  e.g.  "53.7%"  "54%"
    for m in re.finditer(r'\b(\d+(?:\.\d+)?)\s*%', text):
        val = float(m.group(1))
        snip = text[max(0, m.start()-20):m.end()+20].strip()
        results.append(('pct', val, snip))

    # Decimal non-pct  e.g.  "5.05"  (but not if immediately followed by %)
    for m in re.finditer(r'\b(\d+\.\d+)\b', text):
        end_char = text[m.end():m.end()+1]
        if end_char == '%':
            continue  # already captured above
        val = float(m.group(1))
        snip = text[max(0, m.start()-20):m.end()+20].strip()
        results.append(('float', val, snip))

    # Integers  (only if not part of a decimal already captured)
    decimal_spans = {m.start() for m in re.finditer(r'\b\d+\.\d+\b', text)}
    for m in re.finditer(r'\b(\d+)\b', text):
        # skip if this integer is the start of a decimal
        if m.start() in decimal_spans:
            continue
        # skip years
        val = int(m.group(1))
        if 1900 <= val <= 2030:
            continue
        # skip small incidental numbers (1-5) — likely ordinal/prose words
        if val <= 5:
            continue
        # skip if immediately followed by %
        end_char = text[m.end():m.end()+1]
        if end_char == '%':
            continue
        snip = text[max(0, m.start()-20):m.end()+20].strip()
        results.append(('int', val, snip))

    return results


# ── Truth set builder ─────────────────────────────────────────────────────────

def collect_truth_set(dossier: dict):
    """Recursively collect every leaf numeric value from the dossier.
    Returns a set of floats (all numbers canonicalized to float)."""
    truth = set()
    def _walk(obj):
        if isinstance(obj, dict):
            for v in obj.values():
                _walk(v)
        elif isinstance(obj, list):
            for v in obj:
                _walk(v)
        elif isinstance(obj, bool):
            pass
        elif isinstance(obj, (int, float)) and obj is not None:
            truth.add(float(obj))
    _walk(dossier)
    return truth


def is_traceable(kind: str, val: float, truth: set) -> bool:
    """Return True if val appears in truth within allowed tolerance."""
    if kind == 'pct':
        return any(abs(val - t) <= 0.15 for t in truth)
    elif kind == 'float':
        return any(abs(val - t) <= 0.1 for t in truth)
    else:  # int
        return (float(val) in truth or
                float(val - 1) in truth or
                float(val + 1) in truth)


# ── Test questions ─────────────────────────────────────────────────────────────

TEST_QUESTIONS = [
    {
        "q": "How many aces did each player hit?",
        "desc": "Simple metric — raw counts + pct should be easily traceable",
    },
    {
        "q": "How many points did each player win?",
        "desc": "Points won with total denominator — ratio + counts both in dossier",
    },
    {
        "q": "How did the crosscourt to crosscourt to down the line pattern compare between players?",
        "desc": "Pattern metric — occurrences + win%",
    },
    {
        "q": "Compare aces by serve target (wide, body, T) for each player",
        "desc": "Grouped breakdown — bucket counts in dossier",
    },
]

PASSES = []
FAILURES = []

def record_bool(label, condition, detail=""):
    if condition:
        PASSES.append(label)
    else:
        FAILURES.append((label, detail))
    status = "OK  " if condition else "FAIL"
    print(f"  {status} | {label:70} | {detail}")


# ── Load matches once ──────────────────────────────────────────────────────────
print(f"\n{'='*90}")
print("CROSS-MATCH SEMANTIC TEST — Level 3 (narrative grounding)")
print(f"{'='*90}")

eng = CrossMatchEngine()
refs = find_matches(players=[SINNER, ALCARAZ], require_all_players=True)
print(f"  {len(refs)} rivalry matches loaded\n")

for test in TEST_QUESTIONS:
    q = test["q"]
    print(f"{'─'*80}")
    print(f"Q: {q}")
    print(f"   ({test['desc']})")

    # ── Step 1: run deterministic engine ──────────────────────────────────────
    reduced, iros = eng.analyze(q, refs, progress=False)
    print(f"  {len(iros)} IROs built, {len(reduced.skipped)} skipped")

    # ── Step 2: build dossier ─────────────────────────────────────────────────
    dossier = build_dossier(reduced, iros=iros)
    truth = collect_truth_set(dossier)
    print(f"  {len(truth)} unique numbers in dossier truth set")
    record_bool(
        f"[L3-data] reduced has metrics/patterns/grouped",
        bool(reduced.metrics or reduced.patterns or reduced.grouped),
        f"metrics={list(reduced.metrics.keys())}  patterns={len(reduced.patterns)}  grouped={list(reduced.grouped.keys())}",
    )

    # ── Step 3: generate narrative ────────────────────────────────────────────
    print("  Generating narrative (LLM)...")
    narrative = narrate(reduced, q, iros=iros, dossier=dossier)

    record_bool(
        f"[L3-narr] narrative was generated",
        bool(narrative),
        f"{len(narrative)} chars" if narrative else "empty",
    )
    if not narrative:
        print("  (no narrative — skipping number checks)\n")
        continue

    print(f"\n  NARRATIVE:\n{'─'*60}")
    for line in narrative.splitlines():
        print(f"  {line}")
    print(f"{'─'*60}\n")

    # ── Step 4: extract numbers from narrative ────────────────────────────────
    tokens = extract_narrative_numbers(narrative)
    print(f"  Extracted {len(tokens)} numeric tokens from narrative")

    if not tokens:
        record_bool(f"[L3-num] numeric tokens found in narrative", False,
                    "LLM produced no numbers — might be too vague")
        continue

    # ── Step 5: check traceability ────────────────────────────────────────────
    traceable = []
    untraced  = []
    for kind, val, snip in tokens:
        if is_traceable(kind, val, truth):
            traceable.append((kind, val, snip))
        else:
            untraced.append((kind, val, snip))

    trace_pct = len(traceable) / len(tokens) if tokens else 1.0
    pass_trace = trace_pct >= PASS_THRESHOLD

    record_bool(
        f"[L3-trace] {trace_pct:.0%} of numbers traceable to dossier (>={PASS_THRESHOLD:.0%})",
        pass_trace,
        f"{len(traceable)}/{len(tokens)} traceable",
    )

    if untraced:
        print(f"\n  UNTRACED numbers (potential hallucinations):")
        for kind, val, snip in untraced:
            print(f"    [{kind}] {val}  in: '...{snip}...'")
    else:
        print("  All narrative numbers traceable to dossier.")

    # ── Step 6: verify no derived calculations (per-match averages etc.) ───────
    # A derived number would be something like "5.05 aces per match" where 5.05
    # is not in the dossier but is count/n_matches = 101/20. Flag those specifically.
    derived_suspects = []
    n = reduced.n_matches
    for kind, val, snip in untraced:
        if kind == 'float' and n > 1:
            # Check if val ≈ some dossier count / n_matches
            for t in truth:
                if abs(t / n - val) < 0.05:
                    derived_suspects.append((val, f"≈ {t:.0f}/{n} = {t/n:.2f}", snip))
    if derived_suspects:
        print(f"\n  DERIVED VALUES (LLM computed averages — not allowed):")
        for val, formula, snip in derived_suspects:
            print(f"    {val} ≈ {formula}  in: '...{snip}...'")
        record_bool(
            f"[L3-derived] no per-match averages invented by LLM",
            False,
            f"{len(derived_suspects)} suspected derived values",
        )
    else:
        record_bool(
            f"[L3-derived] no per-match averages invented by LLM",
            True,
            "no derived values detected",
        )

    print()


# ── Dossier integrity check ───────────────────────────────────────────────────
# Verify the dossier itself is self-consistent (numbers it contains add up)
print(f"\n{'─'*80}")
print("Dossier integrity: verify n_matches matches IRO count")
q_check = TEST_QUESTIONS[0]["q"]
reduced_c, iros_c = eng.analyze(q_check, refs, progress=False)
doss_c = build_dossier(reduced_c, iros=iros_c)
record_bool(
    "[L3-doss] dossier.n_matches == len(iros)",
    doss_c["n_matches"] == len(iros_c),
    f"dossier={doss_c['n_matches']}  iros={len(iros_c)}",
)
# Verify every player in dossier metrics is a real player in the refs
known_players = {SINNER, ALCARAZ}
doss_players = set()
for mr in doss_c.get("metrics", {}).values():
    for row in mr:
        doss_players.add(row.get("player", ""))
unknown_players = doss_players - known_players
record_bool(
    "[L3-doss] all metric players are known rivalry players",
    not unknown_players,
    f"unknown: {unknown_players}" if unknown_players else f"players: {doss_players}",
)


# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*90}")
print(f"RESULT: {len(PASSES)} OK  /  {len(FAILURES)} FAIL  out of {len(PASSES)+len(FAILURES)}")
if FAILURES:
    print("\nFailed checks:")
    for label, detail in FAILURES:
        print(f"  FAIL: {label}")
        if detail:
            print(f"        {detail}")
print("="*90)
