"""
Adapter: engine-native structured result  ->  canonical MatchIRO.

Reads ONLY the verified locations in the engine's structured output:

  structured['results'][op_id] = {'type': 'tree', 'analysis': {'results': <node>}}
  <node> chain: filter nodes linked by ['children']; terminal is either
     - a GROUP node with ['branches'][label]['results'] = <leaf>, or
     - a <leaf> directly.
  <leaf> = {'metrics': {m: {count,total}}, 'per_player_metrics': {m: {'player1':{count,total}, 'player2':{count,total}}}, ...}

Per-player extraction maps the leaf's 'player1'/'player2' slots to the match's
REAL player names (from provenance), because a player is p1 in one match and p2
in another. We descend to exactly ONE terminal per op and read only its leaf/
branch-leaf per_player_metrics, so nothing is double-counted.
"""
from __future__ import annotations
import json
import hashlib
from typing import Dict, List, Optional

from .iro import Stat, MetricResult, Provenance, MatchIRO, PatternResult, GroupedResult

_PLAYER_DIMS = {"player", "players"}

_MAX_DEPTH = 64


def compute_plan_signature(plan: Dict) -> str:
    """Stable hash of the execution plan (route/filters/metrics), LLM-free."""
    try:
        ops = plan.get("ops", []) if isinstance(plan, dict) else []
        skeleton = [
            {
                "route": op.get("route"),
                "filters": op.get("filters", {}),
                "metrics": sorted(op.get("metrics", []) or []),
                "group_by": op.get("group_by"),
            }
            for op in ops
        ]
        blob = json.dumps(skeleton, sort_keys=True, default=str)
    except Exception:
        blob = repr(plan)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def _descend_to_terminal(node: Dict) -> Optional[Dict]:
    """Follow the filter chain via ['children'] to the terminal node."""
    seen = 0
    while isinstance(node, dict) and isinstance(node.get("children"), dict):
        node = node["children"]
        seen += 1
        if seen > _MAX_DEPTH:
            break
    return node if isinstance(node, dict) else None


def _collect_leaf_ppm(node: Dict, acc: Optional[Dict] = None) -> Dict:
    """Recursively sum per_player_metrics across ALL leaves under a node.

    Handles arbitrary group nesting (e.g. set -> player). Buckets/branches
    partition the points disjointly, so summation counts each point once.
    Returns {metric: {'player1': [count,total], 'player2': [count,total]}}.
    """
    if acc is None:
        acc = {}
    if not isinstance(node, dict):
        return acc
    branches = node.get("branches")
    if isinstance(branches, dict):
        for br in branches.values():
            res = br.get("results") if isinstance(br, dict) else None
            _collect_leaf_ppm(res if isinstance(res, dict) else br, acc)
        return acc
    ppm = node.get("per_player_metrics")
    if isinstance(ppm, dict):
        for metric, per in ppm.items():
            if not isinstance(per, dict):
                continue
            dst = acc.setdefault(metric, {"player1": [0, 0], "player2": [0, 0]})
            for slot in ("player1", "player2"):
                s = per.get(slot)
                if isinstance(s, dict):
                    dst[slot][0] += int(s.get("count", 0) or 0)
                    dst[slot][1] += int(s.get("total", 0) or 0)
    return acc


def _add_metrics_from_ppm(ppm_agg: Dict, name1: str, name2: str,
                          metrics: Dict[str, MetricResult]) -> None:
    for metric, slots in ppm_agg.items():
        mr = metrics.setdefault(metric, MetricResult())
        mr.add_player(name1, Stat(slots["player1"][0], slots["player1"][1]))
        mr.add_player(name2, Stat(slots["player2"][0], slots["player2"][1]))


def _iter_group_nodes(root: Dict):
    """Yield every GROUP node in the tree (following filter .children and group
    .branches). Works regardless of grouping nesting order."""
    stack = [root]
    while stack:
        n = stack.pop()
        if not isinstance(n, dict):
            continue
        if isinstance(n.get("children"), dict):
            stack.append(n["children"])
        br = n.get("branches")
        if isinstance(br, dict):
            yield n
            for b in br.values():
                res = b.get("results") if isinstance(b, dict) else None
                stack.append(res if isinstance(res, dict) else b)


def _extract_grouped_all(root: Dict, name1: str, name2: str) -> Dict[str, GroupedResult]:
    """Lift EVERY grouped-by-DIMENSION node (serve_target, sets, ...) into
    GroupedResults, keyed by dimension. Handles any nesting order:
      - serve_target outer (buckets are leaves)         -> per-bucket per-player
      - player outer / serve_target inner (one D-group per player) -> summed per bucket
    Skips player groupings (the metric channel covers overall per-player).
    Points are partitioned by each grouping, so summation counts each once."""
    players = {name1.lower(), name2.lower()}
    grouped: Dict[str, GroupedResult] = {}
    for gnode in _iter_group_nodes(root):
        dim = (gnode.get("dimension") or "").strip()
        labels = {str(l).strip().lower() for l in gnode["branches"].keys()}
        if dim.lower() in _PLAYER_DIMS:
            continue
        if labels and labels <= players:
            continue  # branches are the two players -> not a dimension grouping
        gr = grouped.setdefault(dim or "group", GroupedResult(dimension=dim or "group"))
        for label, br in gnode["branches"].items():
            res = br.get("results") if isinstance(br, dict) else None
            ppm = _collect_leaf_ppm(res if isinstance(res, dict) else br)
            for metric, slots in ppm.items():
                gr.add(str(label), metric, name1, Stat(slots["player1"][0], slots["player1"][1]))
                gr.add(str(label), metric, name2, Stat(slots["player2"][0], slots["player2"][1]))
    # combined per bucket/metric
    for gr in grouped.values():
        for metrics in gr.buckets.values():
            for mr in metrics.values():
                combined = Stat()
                for s in mr.by_player.values():
                    combined = combined.merge(s)
                mr.combined = combined
    return {d: g for d, g in grouped.items() if g.buckets}


def _pattern_signature(op_type: str, analysis: Dict) -> str:
    """Canonical 'a -> b -> c' signature for a shot-sequence op."""
    if op_type == "multistep":
        steps = [str(s).strip().lower() for s in (analysis.get("pattern") or []) if str(s).strip()]
        return " -> ".join(steps)
    if op_type == "chain":
        sa = (analysis.get("shot_a") or "").strip().lower()
        sb = (analysis.get("shot_b") or "").strip().lower()
        return f"{sa} -> {sb}" if sb else sa
    if op_type == "consecutive":
        # best-effort: consecutive-shots analysis identifies a repeated shot
        shot = (analysis.get("shot_type") or analysis.get("shot") or "").strip().lower()
        n = analysis.get("threshold") or analysis.get("min_count")
        base = f"{shot} (consecutive)" if shot else "consecutive"
        return f"{base} x{n}" if n else base
    return op_type


def _extract_patterns(results: Dict) -> Dict[str, PatternResult]:
    """Lift shot-sequence ops (multistep/chain/consecutive) into pattern results.

    Per-player attribution uses the initiating player of each instance
    ('initiating_player' for multistep, 'player_a' for chain). 'won' feeds the
    win tally (0 when point rows lack a [Point won by:] tag)."""
    patterns: Dict[str, PatternResult] = {}
    for op_id, op in results.items():
        if not isinstance(op, dict):
            continue
        op_type = op.get("type")
        if op_type not in ("multistep", "chain", "consecutive"):
            continue
        analysis = op.get("analysis") or {}
        if not isinstance(analysis, dict) or analysis.get("error"):
            continue
        sig = _pattern_signature(op_type, analysis)
        if not sig or sig == op_type:
            continue
        instances = analysis.get("matching_points") or analysis.get("chain_matches") or []
        pr = patterns.get(sig) or PatternResult(signature=sig)
        pf = analysis.get("player")
        if pf and pf not in ("both", "all"):
            pr.player_filter = pf

        if instances:
            for inst in instances:
                if not isinstance(inst, dict):
                    continue
                player = inst.get("initiating_player") or inst.get("player_a") or ""
                won = 1 if inst.get("won") else 0
                pr.add_player(player, occurrences=1, wins=won)
        else:
            # no per-instance detail; keep the deterministic total in combined
            total = analysis.get("total_matches")
            if total is None:
                total = analysis.get("total_chain_sequences")
            if total:
                pr.combined = pr.combined.merge(type(pr.combined)(int(total), 0))
        patterns[sig] = pr

    # combined = sum of per-player tallies (when we had per-instance detail)
    for pr in patterns.values():
        if pr.by_player:
            combined = type(pr.combined)()
            for s in pr.by_player.values():
                combined = combined.merge(s)
            pr.combined = combined
    return patterns


def _add_direct_terminal_metrics(terminal: Dict, name1: str, name2: str,
                                 metrics: Dict[str, MetricResult]) -> None:
    """Supplement per_player_metrics with direct fields on the terminal node.

    The engine's tree leaf always stores player1_wins / player2_wins / total_points
    as direct integer fields alongside per_player_metrics. These are the authoritative
    traversal counts (not re-derived from metric conditions) and are more reliable
    for 'points_won' and 'total_points' queries than the per_player_metrics slot,
    which can be under-counted when the plan applies a player filter before the
    metric evaluation step.

    Overwrites any existing 'points_won' / 'total_points' entry only when the
    direct fields are present and self-consistent (p1 + p2 == total).
    """
    p1n = str(terminal.get("player1_name") or name1).strip() or name1
    p2n = str(terminal.get("player2_name") or name2).strip() or name2
    p1w = terminal.get("player1_wins")
    p2w = terminal.get("player2_wins")
    tp  = terminal.get("total_points")

    # points_won: authoritative only when both player slots are present and sum to total
    if isinstance(p1w, int) and isinstance(p2w, int):
        total = p1w + p2w
        if not isinstance(tp, int) or tp == total:  # self-consistent
            mr = MetricResult()
            mr.add_player(p1n, Stat(p1w, total))
            mr.add_player(p2n, Stat(p2w, total))
            mr.combined = Stat(total, total)
            metrics["points_won"] = mr

    # total_points: the combined count is the total, not the per-player sum
    if isinstance(tp, int) and tp > 0:
        # Only trust total_points when it matches p1+p2 (or no player win data)
        if not (isinstance(p1w, int) and isinstance(p2w, int)) or tp == (p1w + p2w):
            mr = MetricResult()
            mr.add_player(p1n, Stat(tp, tp))
            mr.add_player(p2n, Stat(tp, tp))
            mr.combined = Stat(tp, tp)
            metrics["total_points"] = mr


def build_match_iro(structured: Dict, provenance_overrides: Optional[Dict] = None) -> MatchIRO:
    """Convert one engine structured result into a MatchIRO."""
    prov_src = structured.get("provenance", {}) or {}
    if provenance_overrides:
        prov_src = {**prov_src, **{k: v for k, v in provenance_overrides.items() if v is not None}}

    prov = Provenance(
        match_id=str(prov_src.get("match_id") or ""),
        player1=prov_src.get("player1") or "",
        player2=prov_src.get("player2") or "",
        date=prov_src.get("date") or "",
        tournament=prov_src.get("tournament") or "",
        surface=prov_src.get("surface") or "",
        match_score=prov_src.get("match_score") or "",
        total_points=int(prov_src.get("total_points") or 0),
        server_attribution_missing=bool(prov_src.get("server_attribution_missing", False)),
    )

    iro = MatchIRO(
        provenance=prov,
        question=structured.get("question", ""),
        plan_signature=compute_plan_signature(structured.get("plan", {})),
    )

    if prov.server_attribution_missing:
        iro.warnings.append("server_attribution_missing: server-role metrics unreliable for this match")

    results = structured.get("results", {}) or {}
    tree_ops = 0
    for op_id, op in results.items():
        if not isinstance(op, dict):
            continue
        if op.get("type") != "tree":
            continue
        analysis = op.get("analysis") or {}
        root = analysis.get("results")
        terminal = _descend_to_terminal(root) if isinstance(root, dict) else None
        if terminal is None:
            iro.warnings.append(f"op {op_id}: no terminal node")
            continue
        tree_ops += 1
        # metric channel: overall per-player (sums across any grouping/nesting)
        _add_metrics_from_ppm(_collect_leaf_ppm(terminal), prov.player1, prov.player2, iro.metrics)
        # supplement with direct terminal fields (more reliable for points_won / total_points)
        _add_direct_terminal_metrics(terminal, prov.player1, prov.player2, iro.metrics)
        # grouped channel: per-bucket per-player for any dimension grouping (any nesting)
        for dim, gr in _extract_grouped_all(root, prov.player1, prov.player2).items():
            iro.grouped[dim] = iro.grouped[dim].merge(gr) if dim in iro.grouped else gr

    # combined = sum of per-player stats for each metric.
    # Exception: total_points is NOT a per-player sum — both players participated in
    # ALL points, so the combined count equals any single player's count (not 2x).
    for metric_name, mr in iro.metrics.items():
        if metric_name == "total_points" and mr.by_player:
            tp_val = next(iter(mr.by_player.values())).count
            mr.combined = Stat(tp_val, tp_val)
            continue
        combined = Stat()
        for s in mr.by_player.values():
            combined = combined.merge(s)
        mr.combined = combined

    # shot-sequence pattern channel
    iro.patterns = _extract_patterns(results)

    if tree_ops == 0 and not iro.patterns:
        types = sorted({op.get("type") for op in results.values() if isinstance(op, dict)})
        iro.warnings.append(f"no tree/pattern op to aggregate (op types present: {types})")

    return iro
