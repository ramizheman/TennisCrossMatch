"""
LLM scope planner.

Turns a raw natural-language question into a retrieval SCOPE (which matches to
analyze) + the residual analytic question (what to compute per match). The LLM
is used for INTENT/scope extraction ONLY -- it never counts, filters points, or
aggregates. All math still happens in the deterministic per-match engine.

Output is strict JSON:
{
  "scope_type": "single_match|rivalry|player_career|surface|tournament|season|database",
  "players": ["Full Name", ...],
  "surface": "clay|grass|hard|null",
  "date_from": "YYYY-MM-DD|null",
  "date_to": "YYYY-MM-DD|null",
  "tournament_contains": "text|null",
  "residual_question": "the analytic question to run per match"
}
"""
from __future__ import annotations
import os
import re
import json
import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional

SCOPE_TYPES = {"single_match", "rivalry", "player_career", "surface",
               "tournament", "season", "database"}


@dataclass
class ScopePlan:
    scope_type: str = "database"
    players: List[str] = field(default_factory=list)
    require_all_players: bool = False
    surface: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    tournament_contains: Optional[str] = None
    residual_question: str = ""
    raw: Dict = field(default_factory=dict)

    def find_matches_kwargs(self) -> Dict:
        return {
            "players": self.players or None,
            "require_all_players": self.require_all_players,
            "surface": self.surface,
            "date_from": self.date_from,
            "date_to": self.date_to,
            "tournament_contains": self.tournament_contains,
        }

    def as_dict(self) -> Dict:
        d = dict(self.__dict__)
        d.pop("raw", None)
        return d


_PROMPT = """You are a query SCOPE planner for a tennis analytics engine.
Given a user question, decide WHICH matches must be analyzed and restate the
analytic question. You do NOT answer the question or compute any numbers.

Today's date is {today}.

Return ONLY strict JSON (no prose, no code fence) with these keys:
- "scope_type": one of single_match, rivalry, player_career, surface, tournament, season, database
- "players": array of FULL player names mentioned (expand surnames, e.g. "Alcaraz"->"Carlos Alcaraz", "Djokovic"->"Novak Djokovic", "Sinner"->"Jannik Sinner"). Empty array if none.
- "surface": "clay", "grass", "hard", or null
- "date_from": "YYYY-MM-DD" or null  (e.g. "since 2020" -> "2020-01-01"; "in 2023" -> "2023-01-01")
- "date_to": "YYYY-MM-DD" or null    (e.g. "in 2023" -> "2023-12-31")
- "tournament_contains": a tournament name substring (e.g. "Wimbledon", "Roland Garros") or null
- "residual_question": the analytic question to run on EACH match, with scope words (player names, surface, dates, "career", "rivalry") removed but the METRIC/PATTERN kept. Keep "each player" style wording.

Guidance:
- Two named players compared head-to-head -> "rivalry" (both must be in the match).
- One player across their matches -> "player_career".
- A surface focus ("on clay") -> "surface" (+ players if named).
- A tournament focus -> "tournament".
- A single year/season -> "season" with date_from/date_to.
- No player and no filter -> "database".

Examples:
Q: "How has Sinner's ace rate changed since 2020?"
{{"scope_type":"player_career","players":["Jannik Sinner"],"surface":null,"date_from":"2020-01-01","date_to":null,"tournament_contains":null,"residual_question":"How many aces did each player hit?"}}

Q: "Compare Alcaraz and Sinner on clay"
{{"scope_type":"surface","players":["Carlos Alcaraz","Jannik Sinner"],"surface":"clay","date_from":null,"date_to":null,"tournament_contains":null,"residual_question":"Compare the players' key serve and return stats"}}

Q: "In the Sinner Alcaraz rivalry how many times did forehand to forehand occur"
{{"scope_type":"rivalry","players":["Jannik Sinner","Carlos Alcaraz"],"surface":null,"date_from":null,"date_to":null,"tournament_contains":null,"residual_question":"How many times did forehand -> forehand occur"}}

User question: {question}
JSON:"""


def _extract_json(text: str) -> Optional[Dict]:
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?", "", t).strip()
    t = re.sub(r"```$", "", t).strip()
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _norm_surface(s) -> Optional[str]:
    if not s:
        return None
    s = str(s).strip().lower()
    return s if s in ("clay", "grass", "hard") else None


def plan_scope(question: str, model_name: str = "gemini-2.5-flash",
               llm_json: Optional[Dict] = None) -> ScopePlan:
    """Plan the retrieval scope for a question. Pass llm_json to bypass the LLM
    (useful for tests). Falls back to a database-wide scope on any failure."""
    data = llm_json
    if data is None:
        try:
            import google.generativeai as genai
            api_key = os.getenv("GOOGLE_API_KEY")
            if api_key:
                genai.configure(api_key=api_key)
            prompt = _PROMPT.format(today=datetime.date.today().isoformat(), question=question)
            resp = genai.GenerativeModel(model_name).generate_content(prompt)
            data = _extract_json(getattr(resp, "text", "") or "")
        except Exception as e:
            print(f"[scope-planner] LLM scope parse failed ({e}); defaulting to database scope")
            data = None

    if not isinstance(data, dict):
        return ScopePlan(scope_type="database", residual_question=question, raw={})

    scope_type = str(data.get("scope_type") or "database").strip().lower()
    if scope_type not in SCOPE_TYPES:
        scope_type = "database"
    players = [p.strip() for p in (data.get("players") or []) if str(p).strip()]
    require_all = scope_type in ("rivalry", "single_match") and len(players) >= 2

    return ScopePlan(
        scope_type=scope_type,
        players=players,
        require_all_players=require_all,
        surface=_norm_surface(data.get("surface")),
        date_from=(data.get("date_from") or None),
        date_to=(data.get("date_to") or None),
        tournament_contains=(data.get("tournament_contains") or None),
        residual_question=(data.get("residual_question") or question).strip() or question,
        raw=data,
    )
