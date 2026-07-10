"""
Retrieval / scope layer.

Given a scope (players, surface, tour, date range, tournament), return the set of
per-match JSON files to analyze. This is the ONLY place that decides WHICH matches
are in scope; the deterministic engine then computes all numbers from each match's
point tree.

v1 uses the per_match_json directory directly (filename prefilter + match_info
confirmation), which is exactly the corpus the graph was built from. A graph-backed
variant (Cypher scope queries) can be layered on later without changing the IRO/
reducer contract.
"""
from __future__ import annotations
import os
import re
import glob
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Iterable

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PER_MATCH_DIR = os.path.join(
    os.path.dirname(_BASE), "Tennis Strategy", "strategy_app", "data", "per_match_json"
)

_DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})_")

_CLAY = ['roland garros', 'french open', 'rome', 'madrid', 'monte carlo', 'barcelona',
         'hamburg', 'stuttgart', 'estoril', 'umag', 'bastad', 'kitzbuhel', 'gstaad']
_GRASS = ['wimbledon', 'queens', "queen's", 'halle', 'eastbourne', "'s-hertogenbosch",
          'newport', 'mallorca', 'stuttgart grass']


def detect_surface(tournament: str) -> str:
    t = (tournament or "").lower()
    if any(k in t for k in _CLAY):
        return "clay"
    if any(k in t for k in _GRASS):
        return "grass"
    return "hard"


@dataclass
class MatchRef:
    filepath: str
    match_id: str = ""
    player1: str = ""
    player2: str = ""
    date: str = ""
    tournament: str = ""
    surface: str = ""
    tour: str = ""

    def provenance_overrides(self) -> Dict:
        return {
            "match_id": self.match_id,
            "player1": self.player1,
            "player2": self.player2,
            "date": self.date,
            "tournament": self.tournament,
            "surface": self.surface,
        }


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _name_tokens(name: str) -> List[str]:
    """Filename-friendly forms of a player name for prefiltering."""
    n = _norm(name)
    forms = {n, n.replace(" ", "_")}
    parts = n.split()
    if parts:
        forms.add(parts[-1])  # last name
    return [f for f in forms if f]


def _match_id_from(match_info: Dict, basename: str, date: str, p1: str, p2: str) -> str:
    d = match_info.get("date") or date
    a = (match_info.get("player1") or p1 or "").replace(" ", "_")
    b = (match_info.get("player2") or p2 or "").replace(" ", "_")
    if d and a and b:
        return f"{d}_{a}_vs_{b}"
    return os.path.splitext(basename)[0]


def _parse_from_filename(basename: str):
    """Best-effort (date, player1, player2, tournament) from the filename."""
    stem = os.path.splitext(basename)[0]
    date = ""
    m = _DATE_RE.match(stem + "_")
    rest = stem
    if m:
        date = m.group(1)
        rest = stem[len(date) + 1:]
    p1 = p2 = ""
    tournament = rest
    if "_vs_" in rest:
        left, p2 = rest.rsplit("_vs_", 1)
        # left = Tournament_[Round_]Player1 ; player1 is trailing 2 tokens (best-effort)
        toks = left.split("_")
        p1 = " ".join(toks[-2:]) if len(toks) >= 2 else left
        tournament = " ".join(toks[:-2]) if len(toks) >= 2 else ""
        p2 = p2.replace("_", " ")
    return date, p1, p2, tournament.replace("_", " ")


def find_matches(players: Optional[Iterable[str]] = None,
                 require_all_players: bool = True,
                 surface: Optional[str] = None,
                 tour: Optional[str] = None,
                 date_from: Optional[str] = None,
                 date_to: Optional[str] = None,
                 tournament_contains: Optional[str] = None,
                 per_match_dir: str = None,
                 limit: Optional[int] = None,
                 confirm_with_match_info: bool = True) -> List[MatchRef]:
    """Return MatchRefs matching the scope.

    players: filter to matches involving these players. With require_all_players=True
             (rivalry), the match must involve ALL of them; else ANY of them.
    Other filters are applied via match_info when confirm_with_match_info=True,
    else via filename heuristics.
    """
    per_match_dir = per_match_dir or DEFAULT_PER_MATCH_DIR
    files = sorted(glob.glob(os.path.join(per_match_dir, "*.json")))

    player_forms = [ _name_tokens(p) for p in players ] if players else None

    def basename_has_player(basename_l: str, forms: List[str]) -> bool:
        return any(f in basename_l for f in forms)

    refs: List[MatchRef] = []
    for fp in files:
        bn = os.path.basename(fp)
        bn_l = bn.lower()

        # cheap filename prefilter on players
        if player_forms is not None:
            hits = [basename_has_player(bn_l, forms) for forms in player_forms]
            if require_all_players and not all(hits):
                continue
            if not require_all_players and not any(hits):
                continue

        date_fn, p1_fn, p2_fn, tour_fn = _parse_from_filename(bn)
        p1, p2, date, tournament, tour_val = p1_fn, p2_fn, date_fn, tour_fn, ""

        if confirm_with_match_info:
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                mi = data.get("match", {}) or {}
                p1 = mi.get("player1") or p1
                p2 = mi.get("player2") or p2
                date = mi.get("date") or date
                tournament = mi.get("tournament") or tournament
                tour_val = mi.get("tour") or ""
            except Exception:
                pass  # fall back to filename-derived values

        # precise player confirmation (handles filename token ambiguity)
        if player_forms is not None and confirm_with_match_info:
            names_l = f"{_norm(p1)} {_norm(p2)}"
            hits = [any(f in names_l for f in forms) for forms in player_forms]
            if require_all_players and not all(hits):
                continue
            if not require_all_players and not any(hits):
                continue

        surf = detect_surface(tournament)
        if surface and surf != surface.lower():
            continue
        if tour and tour_val and tour_val.upper() != tour.upper():
            continue
        if tournament_contains and tournament_contains.lower() not in _norm(tournament):
            continue
        if date_from and date and date < date_from:
            continue
        if date_to and date and date > date_to:
            continue

        refs.append(MatchRef(
            filepath=fp,
            match_id=_match_id_from({"date": date, "player1": p1, "player2": p2}, bn, date, p1, p2),
            player1=p1, player2=p2, date=date, tournament=tournament,
            surface=surf, tour=tour_val,
        ))
        if limit and len(refs) >= limit:
            break

    refs.sort(key=lambda r: r.date or "")
    return refs
