"""
Intermediate Result Object (IRO) for cross-match aggregation.

Design invariants (do NOT violate):
  1. RATIO-SAFE: every quantity is stored as {count, total}, NEVER pre-divided.
     Percentages are derived only at presentation time, after reduction.
  2. RAW COUNTS: counts are integers; merging is pure summation.
  3. PLAYER-NAME KEYED: per-player stats are keyed by the player's real name
     (global identity), not match-local 'player1'/'player2'. This is what makes
     stats mergeable across matches where a player is p1 in one and p2 in another.
  4. PROVENANCE: every MatchIRO records which match it came from, so any
     aggregate is fully traceable and auditable.
  5. Merges are associative + commutative -> order-independent reduction.

The engine (chat_match_questions) never knows about this schema; the adapter
converts the engine's native structured result into these objects.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Stat:
    """A ratio-safe statistic. count = numerator, total = denominator.

    For pure counts (aces, winners), total may be 0/unused; count is the value.
    For ratios (first_serve_pct), both are meaningful and pct = count/total.
    """
    count: int = 0
    total: int = 0

    def merge(self, other: "Stat") -> "Stat":
        return Stat(self.count + other.count, self.total + other.total)

    @property
    def pct(self) -> Optional[float]:
        return round(100.0 * self.count / self.total, 1) if self.total else None

    def as_dict(self) -> Dict:
        d = {"count": self.count, "total": self.total}
        if self.total:
            d["pct"] = self.pct
        return d


@dataclass
class MetricResult:
    """One metric's result: per-player stats (by real name) + combined."""
    by_player: Dict[str, Stat] = field(default_factory=dict)
    combined: Stat = field(default_factory=Stat)

    def add_player(self, player: str, stat: Stat) -> None:
        if not player:
            return
        cur = self.by_player.get(player, Stat())
        self.by_player[player] = cur.merge(stat)

    def merge(self, other: "MetricResult") -> "MetricResult":
        out = MetricResult(combined=self.combined.merge(other.combined))
        out.by_player = {k: Stat(v.count, v.total) for k, v in self.by_player.items()}
        for p, s in other.by_player.items():
            out.add_player(p, s)
        return out

    def as_dict(self) -> Dict:
        return {
            "by_player": {p: s.as_dict() for p, s in self.by_player.items()},
            "combined": self.combined.as_dict(),
        }


@dataclass
class PatternPlayerStat:
    """Per-player tally for an ordered shot pattern.
    occurrences = how many times the player initiated the pattern.
    wins = of those, how many points the initiating player won (0 when the
    source point rows lack a [Point won by:] tag)."""
    occurrences: int = 0
    wins: int = 0

    def merge(self, other: "PatternPlayerStat") -> "PatternPlayerStat":
        return PatternPlayerStat(self.occurrences + other.occurrences, self.wins + other.wins)

    @property
    def win_pct(self):
        return round(100.0 * self.wins / self.occurrences, 1) if self.occurrences else None

    def as_dict(self) -> Dict:
        d = {"occurrences": self.occurrences, "wins": self.wins}
        if self.wins:
            d["win_pct"] = self.win_pct
        return d


@dataclass
class PatternResult:
    """One ordered shot pattern (e.g. 'forehand -> approach -> volley -> winner'),
    tallied per player. Merges by summation across matches."""
    signature: str = ""
    by_player: Dict[str, PatternPlayerStat] = field(default_factory=dict)
    combined: PatternPlayerStat = field(default_factory=PatternPlayerStat)
    player_filter: Optional[str] = None  # set if the query restricted to one player

    def add_player(self, player: str, occurrences: int = 0, wins: int = 0) -> None:
        if not player:
            return
        cur = self.by_player.get(player, PatternPlayerStat())
        self.by_player[player] = cur.merge(PatternPlayerStat(occurrences, wins))

    def merge(self, other: "PatternResult") -> "PatternResult":
        out = PatternResult(
            signature=self.signature or other.signature,
            combined=self.combined.merge(other.combined),
            player_filter=self.player_filter if self.player_filter == other.player_filter else None,
        )
        out.by_player = {k: PatternPlayerStat(v.occurrences, v.wins) for k, v in self.by_player.items()}
        for p, s in other.by_player.items():
            out.add_player(p, s.occurrences, s.wins)
        return out

    def as_dict(self) -> Dict:
        return {
            "signature": self.signature,
            "player_filter": self.player_filter,
            "by_player": {p: s.as_dict() for p, s in self.by_player.items()},
            "combined": self.combined.as_dict(),
        }


@dataclass
class GroupedResult:
    """A grouped-by-dimension breakdown, e.g. serve_target -> {wide,body,t} or
    sets -> {1,2,3,4}. buckets[bucket_label][metric] is a per-player MetricResult.
    Merges bucket-by-bucket, metric-by-metric (summation)."""
    dimension: str = ""
    buckets: Dict[str, Dict[str, MetricResult]] = field(default_factory=dict)

    def add(self, bucket: str, metric: str, player: str, stat: Stat) -> None:
        b = self.buckets.setdefault(str(bucket), {})
        mr = b.setdefault(metric, MetricResult())
        mr.add_player(player, stat)

    def merge(self, other: "GroupedResult") -> "GroupedResult":
        out = GroupedResult(dimension=self.dimension or other.dimension)
        for src in (self, other):
            for bucket, metrics in src.buckets.items():
                dst = out.buckets.setdefault(bucket, {})
                for metric, mr in metrics.items():
                    dst[metric] = dst[metric].merge(mr) if metric in dst else MetricResult().merge(mr)
        # recompute combined for each bucket/metric
        for metrics in out.buckets.values():
            for mr in metrics.values():
                combined = Stat()
                for s in mr.by_player.values():
                    combined = combined.merge(s)
                mr.combined = combined
        return out

    def as_dict(self) -> Dict:
        return {
            "dimension": self.dimension,
            "buckets": {b: {m: r.as_dict() for m, r in metrics.items()}
                        for b, metrics in self.buckets.items()},
        }


@dataclass
class Provenance:
    match_id: str = ""
    player1: str = ""
    player2: str = ""
    date: str = ""
    tournament: str = ""
    surface: str = ""
    match_score: str = ""
    total_points: int = 0
    server_attribution_missing: bool = False

    def as_dict(self) -> Dict:
        return dict(self.__dict__)


@dataclass
class MatchIRO:
    """Deterministic per-match result for one question, ready to reduce."""
    provenance: Provenance = field(default_factory=Provenance)
    question: str = ""
    plan_signature: str = ""
    metrics: Dict[str, MetricResult] = field(default_factory=dict)
    patterns: Dict[str, PatternResult] = field(default_factory=dict)
    grouped: Dict[str, GroupedResult] = field(default_factory=dict)  # dimension -> GroupedResult
    # non-fatal notes (e.g., "no tree op", "grouped-by-set not mapped")
    warnings: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict:
        return {
            "provenance": self.provenance.as_dict(),
            "question": self.question,
            "plan_signature": self.plan_signature,
            "metrics": {m: r.as_dict() for m, r in self.metrics.items()},
            "patterns": {p: r.as_dict() for p, r in self.patterns.items()},
            "grouped": {d: g.as_dict() for d, g in self.grouped.items()},
            "warnings": self.warnings,
        }


@dataclass
class ReducedResult:
    """Cross-match aggregate: metrics merged over N matches, plus coverage."""
    question: str = ""
    metrics: Dict[str, MetricResult] = field(default_factory=dict)
    patterns: Dict[str, PatternResult] = field(default_factory=dict)
    grouped: Dict[str, GroupedResult] = field(default_factory=dict)
    match_ids: List[str] = field(default_factory=list)
    n_matches: int = 0
    skipped: List[Dict] = field(default_factory=list)  # {match_id, reason}
    narrative: str = ""  # optional LLM explanation of the numbers above (explain=True)

    def as_dict(self) -> Dict:
        return {
            "question": self.question,
            "n_matches": self.n_matches,
            "match_ids": self.match_ids,
            "metrics": {m: r.as_dict() for m, r in self.metrics.items()},
            "patterns": {p: r.as_dict() for p, r in self.patterns.items()},
            "grouped": {d: g.as_dict() for d, g in self.grouped.items()},
            "narrative": self.narrative,
            "skipped": self.skipped,
        }
