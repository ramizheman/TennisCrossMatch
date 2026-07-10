"""
Cross-match orchestrator.

    question + scope
      -> retrieval.find_matches  (WHICH matches)
      -> for each match: load point data (NO embeddings) into the engine
      -> engine.analyze_question_structured  (deterministic tree math)
      -> adapter.build_match_iro
      -> reducer.reduce_iros  (associative merge)
      -> ReducedResult (+ per-match IROs for trend/audit)

The engine computes ALL numbers from the point tree. Embeddings/FAISS are NOT
built during fan-out (narrative context degrades gracefully), so per-match cost
is just: load points + one intent-parse + deterministic traversal.
"""
from __future__ import annotations
import os
import re
import sys
import json
import copy
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Iterable, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(_ROOT, "agents") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "agents"))

# Serialize concurrent agent initializations (sentence-transformer model load).
_worker_init_lock = threading.Lock()

from .iro import MatchIRO, ReducedResult
from .adapter import build_match_iro
from .reducer import reduce_iros, rank_players_by_metric, player_metric_over_matches
from .retrieval import find_matches, MatchRef
from .scope_planner import plan_scope, ScopePlan
from .narrator import narrate


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").replace("\u00a0", " ").strip().lower())


class CrossMatchEngine:
    def __init__(self, agent=None, llm_provider: str = "gemini", model: str = "gemini-2.5-flash"):
        self._llm_provider = llm_provider
        self._model = model
        if agent is None:
            from chat_match_questions import TennisChatAgentEmbeddingQALocal
            agent = TennisChatAgentEmbeddingQALocal(llm_provider=llm_provider)
            try:
                agent.model = model
            except Exception:
                pass
        self.agent = agent

    def _make_worker_agent(self):
        """Create a fresh agent instance for a parallel worker thread.

        Worker agents share nothing with the main agent. Initialization is
        serialized via _worker_init_lock so multiple threads don't try to load
        the sentence-transformer model simultaneously.
        """
        from chat_match_questions import TennisChatAgentEmbeddingQALocal
        with _worker_init_lock:
            agent = TennisChatAgentEmbeddingQALocal(llm_provider=self._llm_provider)
            try:
                agent.model = self._model
            except Exception:
                pass
        return agent

    # ---- per-match loading (point data only, no embeddings) -----------------
    def _prep_rows(self, rows: List[Dict], p1: str, p2: str) -> List[Dict]:
        n1, n2 = _norm(p1), _norm(p2)

        def resolve_returner(server_norm: str) -> str:
            if server_norm == n1:
                return p2
            if server_norm == n2:
                return p1
            # last-name fallback
            ln = server_norm.split()[-1] if server_norm else ""
            if ln and ln in n1:
                return p2
            if ln and ln in n2:
                return p1
            return ""

        out = []
        for i, r in enumerate(rows):
            server = (r.get("server") or "").replace("\u00a0", " ").strip()
            desc = r.get("description") or ""
            if not desc and r.get("formatted"):
                fm = r["formatted"]
                desc = fm.split(":", 1)[1].strip() if ":" in fm else fm
            out.append({
                "point_number": r.get("point_number", i + 1),
                "server": server,
                "returner": resolve_returner(_norm(server)),
                "sets": r.get("sets", ""),
                "games": r.get("games", ""),
                "points": r.get("points", ""),
                "description": desc,
            })
        return out

    def load_match(self, ref: MatchRef) -> Tuple[str, str, int]:
        with open(ref.filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        mi = data.get("match", {}) or {}
        p1 = mi.get("player1") or ref.player1
        p2 = mi.get("player2") or ref.player2
        pb = (data.get("scraped") or {}).get("point_by_point") or {}
        rows = pb.get("pointlog_rows") or []
        prepped = self._prep_rows(rows, p1, p2)

        a = self.agent
        a.player1 = p1
        a.player2 = p2
        a.match_id = ref.match_id
        a.tournament = mi.get("tournament", "")
        a.match_score = pb.get("match_result", "") or ""
        # avoid stale narrative retrieval from a previously-loaded match
        a.chunks = []
        for attr in ("index", "embeddings", "chunk_embeddings"):
            if hasattr(a, attr):
                try:
                    setattr(a, attr, None)
                except Exception:
                    pass
        a._load_point_by_point_from_json(prepped)
        self._inject_winner_tags(a)
        return p1, p2, len(prepped)

    @staticmethod
    def _inject_winner_tags(agent) -> None:
        """Append '[Point won by: X]' to each loaded point's text.

        _enrich_point_data already computes point['point_winner'] (via the same
        _determine_point_winner used to GENERATE the NL tags). But the shot-
        sequence analyzers (multistep/chain) read the winner from the TAG in the
        text, which the raw JSON descriptions lack. Injecting the tag here makes
        the JSON fan-out path text-identical to the NL path, so pattern win% and
        winner/tag-based filters compute instead of silently returning 0.
        """
        for p in getattr(agent, "point_by_point", None) or []:
            w = (p.get("point_winner") or "").strip()
            if not w:
                continue
            txt = p.get("point_text", "") or ""
            if "[point won by:" in txt.lower():
                continue
            tag = f" [Point won by: {w}]"
            p["point_text"] = txt + tag
            if "[point won by:" not in (p.get("description", "") or "").lower():
                p["description"] = (p.get("description", "") or "") + tag

    # ---- fan-out + reduce ---------------------------------------------------
    def analyze(self, question: str, refs: List[MatchRef], progress: bool = True,
                reuse_plan: bool = True,
                max_workers: int = 6) -> Tuple[ReducedResult, List[MatchIRO]]:
        """Fan out a question over the match set and reduce.

        reuse_plan=True (default): compute the execution plan ONCE via one LLM
        call on the main agent, then pass an identical deep-copy to every worker.
        This makes results deterministic and costs one LLM call regardless of
        match count.

        max_workers: number of concurrent worker threads. Each worker has its
        own isolated agent instance (no shared mutable state). With 6 workers
        a 20-match rivalry runs in ~25-35s vs ~90s sequential. Workers are lazy-
        initialized on first use and reused across subsequent analyze() calls on
        the same thread (via thread-local storage inside the executor).
        """
        if not refs:
            return reduce_iros([], question=question), []

        iros: List[MatchIRO] = []
        skipped: List[Dict] = []
        t0 = time.time()

        # ── Step 1: compute plan once (1 LLM call) ───────────────────────────
        cached_plan = None
        if reuse_plan:
            self.load_match(refs[0])
            cached_plan = self.agent._plan_question(question)
            if progress:
                ops = [f"{o.get('route')}:{o.get('metrics') or o.get('chain_logic') or ''}"
                       for o in cached_plan[0].get("ops", [])]
                print(f"[plan] parsed once -> ops={ops}")

        # ── Step 2: thread-local worker engines ───────────────────────────────
        # Each worker thread gets its own CrossMatchEngine so agents never
        # share mutable state (player1/player2/point_by_point).
        _tlocal = threading.local()

        def _worker_engine() -> "CrossMatchEngine":
            if not hasattr(_tlocal, "engine"):
                _tlocal.engine = CrossMatchEngine(agent=self._make_worker_agent())
            return _tlocal.engine

        def _process_one(ref: MatchRef):
            eng = _worker_engine()
            _, _, npts = eng.load_match(ref)
            if cached_plan is not None:
                plan, cls = cached_plan
                structured = eng.agent.analyze_question_structured(
                    question,
                    plan=copy.deepcopy(plan),
                    classification=copy.deepcopy(cls),
                )
            else:
                structured = eng.agent.analyze_question_structured(question)
            iro = build_match_iro(structured, provenance_overrides=ref.provenance_overrides())
            return iro, npts

        # ── Step 3: parallel fan-out ──────────────────────────────────────────
        n_workers = min(max_workers, len(refs))
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            future_to_ref = {pool.submit(_process_one, ref): ref for ref in refs}
            done = 0
            for future in as_completed(future_to_ref):
                done += 1
                ref = future_to_ref[future]
                try:
                    iro, npts = future.result()
                    iros.append(iro)
                    if progress:
                        ch = list(iro.metrics.keys()) or list(iro.patterns.keys())
                        print(f"[{done}/{len(refs)}] {ref.match_id} ({npts} pts) -> {ch}")
                except Exception as e:
                    skipped.append({"match_id": ref.match_id,
                                    "reason": f"{type(e).__name__}: {e}"})
                    if progress:
                        print(f"[{done}/{len(refs)}] {ref.match_id} SKIPPED: {e}")

        reduced = reduce_iros(iros, question=question)
        for s in skipped:
            reduced.skipped.append(s)
        if progress:
            elapsed = time.time() - t0
            print(f"[fan-out] {len(iros)} ok, {len(skipped)} skipped "
                  f"in {elapsed:.1f}s ({n_workers} workers)")
        return reduced, iros

    def run(self, question: str,
            players: Optional[Iterable[str]] = None,
            require_all_players: bool = True,
            surface: Optional[str] = None,
            tour: Optional[str] = None,
            date_from: Optional[str] = None,
            date_to: Optional[str] = None,
            tournament_contains: Optional[str] = None,
            limit: Optional[int] = None,
            progress: bool = True,
            explain: bool = False) -> Tuple[ReducedResult, List[MatchIRO]]:
        refs = find_matches(players=players, require_all_players=require_all_players,
                            surface=surface, tour=tour, date_from=date_from, date_to=date_to,
                            tournament_contains=tournament_contains, limit=limit)
        if progress:
            print(f"[scope] {len(refs)} matches in scope")
        reduced, iros = self.analyze(question, refs, progress=progress)
        if explain:
            reduced.narrative = narrate(reduced, question=question, iros=iros,
                                        model_name=self._narrator_model())
        return reduced, iros

    def _narrator_model(self) -> str:
        return getattr(self.agent, "model_25_flash", None) or getattr(self.agent, "model", None) or "gemini-2.5-flash"

    def ask(self, question: str, max_matches: int = 60, progress: bool = True,
            explain: bool = True) -> Tuple[ReducedResult, List[MatchIRO], ScopePlan]:
        """Question-only entrypoint: LLM plans the scope, then fan-out + reduce.

        The LLM only decides WHICH matches (scope) and restates the analytic
        question; all numbers still come from the per-match point tree. With
        explain=True a prose narrative is generated over the deterministic
        numbers (explanation only). Returns (reduced, iros, scope_plan).
        """
        scope = plan_scope(question, model_name=self._narrator_model())
        if progress:
            print(f"[scope-planner] {scope.as_dict()}")
        refs = find_matches(limit=max_matches, **scope.find_matches_kwargs())
        if progress:
            print(f"[scope] type={scope.scope_type} -> {len(refs)} matches "
                  f"(cap {max_matches}); running: {scope.residual_question!r}")
        reduced, iros = self.analyze(scope.residual_question, refs, progress=progress)
        if explain:
            reduced.narrative = narrate(reduced, question=question, scope=scope, iros=iros,
                                        model_name=self._narrator_model())
        return reduced, iros, scope
