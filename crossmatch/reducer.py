"""
Cross-match reducer: fold per-match MatchIRO objects into a ReducedResult.

Merge algebra is pure summation on {count, total}, which is associative and
commutative -> the result is independent of match order. Percentages are derived
ONLY here at the end, never stored pre-divided in the IROs.
"""
from __future__ import annotations
from typing import Dict, List, Optional

from .iro import MetricResult, PatternResult, GroupedResult, ReducedResult, MatchIRO


def reduce_iros(iros: List[MatchIRO], question: str = "") -> ReducedResult:
    out = ReducedResult(question=question)
    for iro in iros:
        mid = iro.provenance.match_id or f"{iro.provenance.date}_{iro.provenance.player1}_vs_{iro.provenance.player2}"
        if not iro.metrics and not iro.patterns and not iro.grouped:
            reason = iro.warnings[0] if iro.warnings else "no metrics/patterns produced"
            out.skipped.append({"match_id": mid, "reason": reason})
            continue
        out.match_ids.append(mid)
        for metric, mr in iro.metrics.items():
            if metric in out.metrics:
                out.metrics[metric] = out.metrics[metric].merge(mr)
            else:
                # copy so we never mutate the source IRO
                out.metrics[metric] = MetricResult().merge(mr)
        for sig, pr in iro.patterns.items():
            if sig in out.patterns:
                out.patterns[sig] = out.patterns[sig].merge(pr)
            else:
                out.patterns[sig] = PatternResult(signature=sig).merge(pr)
        for dim, gr in iro.grouped.items():
            if dim in out.grouped:
                out.grouped[dim] = out.grouped[dim].merge(gr)
            else:
                out.grouped[dim] = GroupedResult(dimension=dim).merge(gr)
    out.n_matches = len(out.match_ids)
    return out


def rank_players_by_metric(reduced: ReducedResult, metric: str,
                           by: str = "count", min_total: int = 0,
                           descending: bool = True) -> List[Dict]:
    """Rank players for a metric. by='count' (raw) or 'pct' (count/total)."""
    mr = reduced.metrics.get(metric)
    if not mr:
        return []
    rows = []
    for player, stat in mr.by_player.items():
        if by == "pct":
            if not stat.total or stat.total < max(min_total, 1):
                continue
            key = stat.pct
        else:
            key = stat.count
        rows.append({"player": player, "count": stat.count, "total": stat.total,
                     "pct": stat.pct, "_key": key if key is not None else -1})
    rows.sort(key=lambda r: r["_key"], reverse=descending)
    for r in rows:
        r.pop("_key", None)
    return rows


def rank_players_by_pattern(reduced: ReducedResult, signature: str,
                            by: str = "occurrences", min_occ: int = 1,
                            descending: bool = True) -> List[Dict]:
    """Rank players for a shot pattern. by='occurrences' or 'win_pct'."""
    pr = reduced.patterns.get(signature)
    if not pr:
        return []
    rows = []
    for player, s in pr.by_player.items():
        if s.occurrences < max(min_occ, 1):
            continue
        key = s.win_pct if by == "win_pct" else s.occurrences
        rows.append({"player": player, "occurrences": s.occurrences, "wins": s.wins,
                     "win_pct": s.win_pct, "_key": key if key is not None else -1})
    rows.sort(key=lambda r: r["_key"], reverse=descending)
    for r in rows:
        r.pop("_key", None)
    return rows


def grouped_table(reduced: ReducedResult, dimension: str, metric: str) -> List[Dict]:
    """Flatten a grouped dimension+metric into rows: bucket -> per-player counts.
    e.g. dimension='serve_target', metric='aces' -> [{bucket:'wide', <player>:{count,total,pct}}]."""
    gr = reduced.grouped.get(dimension)
    if not gr:
        return []
    rows = []
    for bucket, metrics in gr.buckets.items():
        mr = metrics.get(metric)
        if not mr:
            continue
        row = {"bucket": bucket, "combined": mr.combined.as_dict()}
        row["by_player"] = {p: s.as_dict() for p, s in mr.by_player.items()}
        rows.append(row)
    rows.sort(key=lambda r: r["bucket"])
    return rows


def pattern_over_matches(iros: List[MatchIRO], signature: str, player: str = None) -> List[Dict]:
    """Per-match time series for a shot pattern (optionally one player)."""
    series = []
    for iro in iros:
        pr = iro.patterns.get(signature)
        if not pr:
            continue
        if player:
            s = pr.by_player.get(player)
            occ, wins = (s.occurrences, s.wins) if s else (0, 0)
        else:
            occ, wins = pr.combined.occurrences, pr.combined.wins
        series.append({
            "match_id": iro.provenance.match_id, "date": iro.provenance.date,
            "tournament": iro.provenance.tournament, "surface": iro.provenance.surface,
            "occurrences": occ, "wins": wins,
        })
    series.sort(key=lambda r: r.get("date") or "")
    return series


def player_metric_over_matches(iros: List[MatchIRO], player: str, metric: str) -> List[Dict]:
    """Per-match time series for one player+metric (for trend questions).
    Sorted by match date when available."""
    series = []
    for iro in iros:
        mr = iro.metrics.get(metric)
        if not mr:
            continue
        stat = mr.by_player.get(player)
        if stat is None:
            continue
        series.append({
            "match_id": iro.provenance.match_id,
            "date": iro.provenance.date,
            "tournament": iro.provenance.tournament,
            "surface": iro.provenance.surface,
            "count": stat.count,
            "total": stat.total,
            "pct": stat.pct,
        })
    series.sort(key=lambda r: r.get("date") or "")
    return series
