"""
Cross-match narrator.

Turns a deterministic ReducedResult into a short analytical prose narrative.
The LLM here is EXPLANATION-ONLY: it is handed the already-computed numbers
(counts, totals, percentages, rankings, per-bucket splits, trends) and asked to
describe them. It must not compute, re-derive, or invent any number. All math
was done by the deterministic engine + reducer; this layer only writes prose.
"""
from __future__ import annotations
import os
import json
from typing import Dict, List, Optional

from .iro import ReducedResult, MatchIRO
from .reducer import (rank_players_by_metric, rank_players_by_pattern,
                      grouped_table, player_metric_over_matches, pattern_over_matches)


def build_dossier(reduced: ReducedResult, scope=None, iros: Optional[List[MatchIRO]] = None,
                  top_n: int = 10) -> Dict:
    """Assemble a compact, purely-factual summary of the deterministic result.
    This is the ONLY thing the narrator LLM is allowed to describe."""
    dossier: Dict = {
        "n_matches": reduced.n_matches,
        "question": reduced.question,
    }
    if scope is not None:
        dossier["scope"] = {
            "scope_type": getattr(scope, "scope_type", None),
            "players": getattr(scope, "players", None),
            "surface": getattr(scope, "surface", None),
            "date_from": getattr(scope, "date_from", None),
            "date_to": getattr(scope, "date_to", None),
            "tournament_contains": getattr(scope, "tournament_contains", None),
        }

    metrics = {}
    for name in reduced.metrics:
        ranked = rank_players_by_metric(reduced, name, by="count")[:top_n]
        if ranked:
            metrics[name] = ranked
    if metrics:
        dossier["metrics"] = metrics

    patterns = {}
    for sig in reduced.patterns:
        ranked = rank_players_by_pattern(reduced, sig, by="occurrences")[:top_n]
        combined = reduced.patterns[sig].combined
        patterns[sig] = {
            "combined": {"occurrences": combined.occurrences, "wins": combined.wins,
                         "win_pct": combined.win_pct},
            "by_player": ranked,
        }
    if patterns:
        dossier["patterns"] = patterns

    grouped = {}
    for dim, gr in reduced.grouped.items():
        dim_metrics = {}
        seen = set()
        for metrics_map in gr.buckets.values():
            seen.update(metrics_map.keys())
        for metric in seen:
            table = grouped_table(reduced, dim, metric)
            if table:
                dim_metrics[metric] = [
                    {"bucket": r["bucket"],
                     "by_player": {p: v["count"] for p, v in r["by_player"].items()},
                     "total": r["combined"]["count"]}
                    for r in table
                ]
        if dim_metrics:
            grouped[dim] = dim_metrics
    if grouped:
        dossier["grouped"] = grouped

    # temporal trends: only meaningful when spanning many matches over time
    if iros and reduced.n_matches >= 3:
        trends = {}
        # metric trend for the leading player of the top metric
        for name in list(metrics.keys())[:2]:
            top_rows = metrics[name]
            if not top_rows:
                continue
            player = top_rows[0]["player"]
            series = player_metric_over_matches(iros, player, name)
            series = [s for s in series if s.get("date")]
            if len(series) >= 3:
                trends.setdefault("metrics", {})[f"{player} | {name}"] = [
                    {"date": s["date"], "count": s["count"], "total": s["total"]}
                    for s in series
                ]
        for sig in list(patterns.keys())[:1]:
            series = [s for s in pattern_over_matches(iros, sig) if s.get("date")]
            if len(series) >= 3:
                trends.setdefault("patterns", {})[sig] = [
                    {"date": s["date"], "occurrences": s["occurrences"], "wins": s["wins"]}
                    for s in series
                ]
        if trends:
            dossier["trends"] = trends

    return dossier


_PROMPT = """You are a professional tennis analyst writing a short, factual summary.

You are given PRE-COMPUTED statistics (a JSON "dossier"). Every number was
computed deterministically from point-by-point data. Your job is ONLY to explain
these numbers in clear analytical prose.

STRICT RULES:
- Use ONLY numbers that appear in the dossier. Never invent, estimate, round to
  new values, or compute new numbers (no ratios/differences that aren't given).
- If you state a percentage or count, it must appear verbatim in the dossier.
- Do not speculate about causes beyond what the data shows; keep it grounded.
- Refer to players by name. Be concise: 1-3 short paragraphs.
- If a "trends" section is present, comment on the direction over time.
- Do not mention "dossier", "JSON", or this prompt.

User question:
{question}

Dossier (authoritative, pre-computed):
{dossier}

Write the analytical summary:"""


def narrate(reduced: ReducedResult, question: str = "", scope=None,
            iros: Optional[List[MatchIRO]] = None,
            model_name: str = "gemini-2.5-flash",
            dossier: Optional[Dict] = None) -> str:
    """Generate a prose narrative of the deterministic ReducedResult.
    Returns "" if there is nothing to describe or the LLM is unavailable.
    Pass a prebuilt `dossier` to bypass assembly (useful for tests)."""
    if dossier is None:
        dossier = build_dossier(reduced, scope=scope, iros=iros)
    has_content = any(k in dossier for k in ("metrics", "patterns", "grouped"))
    if not has_content:
        return ""
    q = question or reduced.question or ""
    prompt = _PROMPT.format(question=q, dossier=json.dumps(dossier, indent=2, default=str))
    try:
        import google.generativeai as genai
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
        resp = genai.GenerativeModel(model_name).generate_content(prompt)
        return (getattr(resp, "text", "") or "").strip()
    except Exception as e:
        print(f"[narrator] LLM narrative failed ({e}); returning empty narrative")
        return ""
