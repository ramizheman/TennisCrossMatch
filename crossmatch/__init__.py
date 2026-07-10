"""
crossmatch: deterministic cross-match aggregation over the single-match engine.

Pipeline:
  NL question
    -> scope (which matches)                      [retrieval.py]
    -> per match: engine.analyze_question_structured()   [chat_match_questions]
    -> MatchIRO                                    [adapter.build_match_iro]
    -> reduce over matches                         [reducer.reduce_iros]
    -> present (rank / trend) + optional LLM explanation

Invariant: ALL numbers come from the point tree (the engine). Neo4j only selects
which matches to analyze. The LLM never counts, filters, or aggregates.
"""
from .iro import (Stat, MetricResult, PatternPlayerStat, PatternResult,
                  GroupedResult, Provenance, MatchIRO, ReducedResult)
from .adapter import build_match_iro, compute_plan_signature
from .reducer import (reduce_iros, rank_players_by_metric, player_metric_over_matches,
                      rank_players_by_pattern, pattern_over_matches, grouped_table)
from .retrieval import find_matches, MatchRef, detect_surface
from .scope_planner import plan_scope, ScopePlan
from .narrator import narrate, build_dossier
from .orchestrator import CrossMatchEngine

__all__ = [
    "Stat", "MetricResult", "PatternPlayerStat", "PatternResult", "GroupedResult",
    "Provenance", "MatchIRO", "ReducedResult",
    "build_match_iro", "compute_plan_signature",
    "reduce_iros", "rank_players_by_metric", "player_metric_over_matches",
    "rank_players_by_pattern", "pattern_over_matches", "grouped_table",
    "find_matches", "MatchRef", "detect_surface",
    "plan_scope", "ScopePlan",
    "narrate", "build_dossier",
    "CrossMatchEngine",
]
