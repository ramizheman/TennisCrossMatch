# TennisCrossMatch

Cross-match natural language reasoning platform for professional tennis.

Built on top of the deterministic single-match engine from [TennisNL](https://github.com/ramizheman/TennisNL).

## What it does

Ask any natural language question across an entire corpus of matches:

- "How many aces did Alcaraz hit against Sinner on clay?"
- "How has Sinner's double-fault rate changed over the rivalry?"
- "Which serve target (wide / body / T) does Alcaraz use most on break points?"
- "Show the crosscourt → crosscourt → down-the-line pattern frequency per match over time"

All numbers are computed deterministically from point-by-point data. The LLM is used only for scope planning (which matches?) and prose explanation (never for arithmetic).

## Architecture

```
question
  -> scope_planner  (LLM: which players / surface / dates?)
  -> retrieval      (find matching per_match_json files)
  -> parallel fan-out over N matches
       -> each match: load point tree -> deterministic engine -> MatchIRO
  -> reducer        (associative merge of MatchIROs -> ReducedResult)
  -> narrator       (LLM: prose explanation from deterministic numbers only)
```

### Key packages

| Package | Role |
|---|---|
| `crossmatch/iro.py` | Typed intermediate result objects (ratio-safe, player-name keyed) |
| `crossmatch/adapter.py` | Engine structured result → MatchIRO |
| `crossmatch/reducer.py` | Associative merge of MatchIROs + presentation helpers |
| `crossmatch/retrieval.py` | File-based match discovery with player/surface/date filtering |
| `crossmatch/orchestrator.py` | Parallel fan-out (ThreadPoolExecutor) + reduction |
| `crossmatch/scope_planner.py` | LLM-based scope extraction (Gemini) |
| `crossmatch/narrator.py` | Grounded prose from ReducedResult (no invented numbers) |
| `agents/chat_match_questions.py` | Deterministic single-match engine (copied from TennisNL) |

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt

cp .env.example .env
# add your GOOGLE_API_KEY to .env
```

## Data

Expects per-match JSON files at:
```
../Tennis Strategy/strategy_app/data/per_match_json/
```
(sibling directory to this repo, matching the TennisNL project layout)

Override with:
```bash
set TENNIS_DATA_DIR=C:\path\to\your\per_match_json
```

## Quick start

```python
from dotenv import load_dotenv
load_dotenv()

from crossmatch import CrossMatchEngine, find_matches

eng = CrossMatchEngine()
refs = find_matches(players=["Jannik Sinner", "Carlos Alcaraz"], require_all_players=True)
reduced, iros = eng.analyze("How many aces did each player hit?", refs)

from crossmatch.reducer import rank_players_by_metric
print(rank_players_by_metric(reduced, "aces"))
```

## Test suite

```bash
# Level 1 — additive consistency (cross-match total == sum of per-match totals)
python _test_crossmatch_additive.py

# Level 2 — temporal ordering + plausibility bounds
python _test_crossmatch_temporal.py

# Single-match cross-validation (cross-match engine == original single-match engine)
python _compare_single_match.py
```
