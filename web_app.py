#!/usr/bin/env python3
"""
TennisCrossMatch — Web UI

FastAPI backend + single-page frontend for cross-match natural language Q&A.
Run:  python web_app.py
Then open http://127.0.0.1:7861
"""
from __future__ import annotations

import asyncio
import os
import sys
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

load_dotenv()

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if os.path.join(_ROOT, "agents") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "agents"))

from crossmatch import CrossMatchEngine, find_matches
from crossmatch.reducer import (
    rank_players_by_metric,
    rank_players_by_pattern,
    grouped_table,
    player_metric_over_matches,
)

app = FastAPI(title="TennisCrossMatch")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class AppState:
    engine: Optional[CrossMatchEngine] = None


state = AppState()


def _get_engine() -> CrossMatchEngine:
    if state.engine is None:
        state.engine = CrossMatchEngine()
    return state.engine


def _serialize_reduced(reduced, iros, scope=None, elapsed_s=None):
    metrics_out = {}
    for name in reduced.metrics:
        metrics_out[name] = rank_players_by_metric(reduced, name, by="count")

    patterns_out = {}
    for sig in reduced.patterns:
        patterns_out[sig] = {
            "combined": reduced.patterns[sig].combined.as_dict(),
            "by_player": rank_players_by_pattern(reduced, sig, by="occurrences"),
        }

    grouped_out = {}
    for dim, gr in reduced.grouped.items():
        dim_metrics = {}
        seen = set()
        for m in gr.buckets.values():
            seen.update(m.keys())
        for metric in seen:
            table = grouped_table(reduced, dim, metric)
            if table:
                dim_metrics[metric] = table
        if dim_metrics:
            grouped_out[dim] = dim_metrics

    matches = []
    for iro in sorted(iros, key=lambda x: x.provenance.date or ""):
        p = iro.provenance
        matches.append({
            "match_id": p.match_id,
            "date": p.date,
            "tournament": p.tournament,
            "surface": p.surface,
            "player1": p.player1,
            "player2": p.player2,
            "metrics": {m: r.as_dict() for m, r in iro.metrics.items()},
        })

    return {
        "question": reduced.question,
        "n_matches": reduced.n_matches,
        "narrative": reduced.narrative or "",
        "metrics": metrics_out,
        "patterns": patterns_out,
        "grouped": grouped_out,
        "matches": matches,
        "skipped": reduced.skipped,
        "scope": scope.as_dict() if scope is not None else None,
        "elapsed_s": elapsed_s,
    }


@app.on_event("startup")
async def startup():
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _get_engine)


@app.get("/api/health")
async def health():
    return {"ok": True, "ready": state.engine is not None}


@app.post("/api/preview")
async def preview_scope(req: Request):
    """Preview how many matches a manual scope would hit."""
    body = await req.json()
    players = [p.strip() for p in body.get("players", []) if p and str(p).strip()]
    surface = (body.get("surface") or "").strip() or None
    date_from = (body.get("date_from") or "").strip() or None
    date_to = (body.get("date_to") or "").strip() or None
    tournament = (body.get("tournament") or "").strip() or None
    require_all = bool(body.get("require_all_players", True))
    limit = body.get("limit")
    try:
        limit = int(limit) if limit not in (None, "", 0) else None
    except (TypeError, ValueError):
        limit = None

    def _run():
        return find_matches(
            players=players or None,
            require_all_players=require_all,
            surface=surface,
            date_from=date_from,
            date_to=date_to,
            tournament_contains=tournament,
            limit=limit,
        )

    loop = asyncio.get_event_loop()
    refs = await loop.run_in_executor(None, _run)
    return JSONResponse({
        "n_matches": len(refs),
        "matches": [
            {
                "date": r.date,
                "tournament": r.tournament,
                "surface": r.surface,
                "player1": r.player1,
                "player2": r.player2,
            }
            for r in refs[:40]
        ],
        "truncated": len(refs) > 40,
    })


@app.post("/api/ask")
async def ask(req: Request):
    """Run a cross-match question.

    mode='auto'   — LLM scope planner (eng.ask)
    mode='manual' — explicit filters (eng.run)
    """
    import time
    body = await req.json()
    question = (body.get("question") or "").strip()
    if not question:
        return JSONResponse({"error": "Question is empty."}, status_code=400)

    mode = (body.get("mode") or "manual").strip().lower()
    explain = bool(body.get("explain", True))
    max_matches = int(body.get("max_matches") or 60)

    players = [p.strip() for p in body.get("players", []) if p and str(p).strip()]
    surface = (body.get("surface") or "").strip() or None
    date_from = (body.get("date_from") or "").strip() or None
    date_to = (body.get("date_to") or "").strip() or None
    tournament = (body.get("tournament") or "").strip() or None
    require_all = bool(body.get("require_all_players", True))

    def _run():
        eng = _get_engine()
        t0 = time.time()
        if mode == "auto":
            reduced, iros, scope = eng.ask(
                question, max_matches=max_matches, progress=False, explain=explain
            )
            return reduced, iros, scope, time.time() - t0
        # Manual scope — more reliable when players are named in filters
        if not players and not surface and not date_from and not tournament:
            # Fall back to auto if no filters provided
            reduced, iros, scope = eng.ask(
                question, max_matches=max_matches, progress=False, explain=explain
            )
            return reduced, iros, scope, time.time() - t0
        reduced, iros = eng.run(
            question,
            players=players or None,
            require_all_players=require_all,
            surface=surface,
            date_from=date_from,
            date_to=date_to,
            tournament_contains=tournament,
            limit=max_matches,
            progress=False,
            explain=explain,
        )
        return reduced, iros, None, time.time() - t0

    try:
        loop = asyncio.get_event_loop()
        reduced, iros, scope, elapsed = await loop.run_in_executor(None, _run)
    except Exception as e:
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)

    if reduced.n_matches == 0 and not iros:
        return JSONResponse({
            "error": "No matches found for this scope. Try different players, surface, or dates.",
            "n_matches": 0,
            "scope": scope.as_dict() if scope else None,
        }, status_code=404)

    return JSONResponse(_serialize_reduced(reduced, iros, scope=scope, elapsed_s=round(elapsed, 1)))


# ── Frontend ──────────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>TennisCrossMatch</title>
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=Syne:wght@600;700;800&display=swap" rel="stylesheet" />
<style>
  :root {
    --bg0: #08110e;
    --bg1: #0e1c16;
    --panel: rgba(18, 36, 28, 0.92);
    --panel2: #152820;
    --line: rgba(184, 230, 58, 0.14);
    --line2: rgba(232, 240, 230, 0.08);
    --text: #eef3ea;
    --muted: #8fa294;
    --dim: #5d7264;
    --accent: #b8e63a;
    --accent-ink: #10200f;
    --clay: #d97845;
    --warn: #e8b84a;
    --err: #ef6b6b;
    --ok: #7dce7a;
    --radius: 14px;
    --shadow: 0 18px 50px rgba(0,0,0,.45);
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    min-height: 100vh;
    font-family: "DM Sans", sans-serif;
    color: var(--text);
    background:
      radial-gradient(1200px 600px at 10% -10%, rgba(184,230,58,.12), transparent 55%),
      radial-gradient(900px 500px at 100% 0%, rgba(217,120,69,.10), transparent 50%),
      linear-gradient(165deg, var(--bg0), var(--bg1) 45%, #0a1612);
    display: grid;
    grid-template-columns: 340px 1fr;
  }

  /* ── Sidebar ── */
  #sidebar {
    border-right: 1px solid var(--line2);
    background: var(--panel);
    backdrop-filter: blur(12px);
    display: flex;
    flex-direction: column;
    min-height: 100vh;
    position: sticky;
    top: 0;
  }

  .brand {
    padding: 28px 24px 20px;
    border-bottom: 1px solid var(--line2);
  }

  .brand-mark {
    font-family: Syne, sans-serif;
    font-weight: 800;
    font-size: 1.55rem;
    letter-spacing: -0.03em;
    line-height: 1.05;
  }

  .brand-mark span { color: var(--accent); }

  .brand p {
    margin-top: 8px;
    color: var(--muted);
    font-size: 0.88rem;
    line-height: 1.4;
  }

  .side-body {
    padding: 18px 20px 24px;
    overflow-y: auto;
    flex: 1;
  }

  .label {
    display: block;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--dim);
    margin: 14px 0 7px;
  }

  .field, .select {
    width: 100%;
    background: var(--panel2);
    border: 1px solid var(--line2);
    color: var(--text);
    border-radius: 10px;
    padding: 10px 12px;
    font: inherit;
    font-size: 0.92rem;
    outline: none;
    transition: border-color .15s;
  }

  .field:focus, .select:focus { border-color: var(--accent); }

  .row2 {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
  }

  .check {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 12px;
    font-size: 0.9rem;
    color: var(--muted);
  }

  .check input { accent-color: var(--accent); width: 15px; height: 15px; }

  .mode-toggle {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 6px;
    margin-top: 4px;
  }

  .mode-btn {
    border: 1px solid var(--line2);
    background: transparent;
    color: var(--muted);
    border-radius: 999px;
    padding: 8px 10px;
    font: inherit;
    font-size: 0.82rem;
    font-weight: 600;
    cursor: pointer;
  }

  .mode-btn.active {
    background: var(--accent);
    color: var(--accent-ink);
    border-color: var(--accent);
  }

  .btn-ghost {
    width: 100%;
    margin-top: 12px;
    background: transparent;
    border: 1px solid var(--line);
    color: var(--accent);
    border-radius: 10px;
    padding: 10px;
    font: inherit;
    font-weight: 600;
    cursor: pointer;
  }

  .btn-ghost:hover { background: rgba(184,230,58,.08); }

  .preview-box {
    margin-top: 12px;
    padding: 10px 12px;
    border-radius: 10px;
    background: rgba(184,230,58,.06);
    border: 1px solid var(--line);
    font-size: 0.85rem;
    color: var(--muted);
    display: none;
  }

  .preview-box.show { display: block; }
  .preview-box strong { color: var(--accent); font-weight: 700; }

  .examples { margin-top: 22px; }
  .chip {
    display: block;
    width: 100%;
    text-align: left;
    margin-bottom: 8px;
    background: var(--panel2);
    border: 1px solid var(--line2);
    color: var(--text);
    border-radius: 10px;
    padding: 10px 12px;
    font: inherit;
    font-size: 0.84rem;
    line-height: 1.35;
    cursor: pointer;
    transition: border-color .15s, transform .12s;
  }
  .chip:hover { border-color: var(--accent); transform: translateX(2px); }

  /* ── Main ── */
  #main {
    display: flex;
    flex-direction: column;
    min-height: 100vh;
    min-width: 0;
  }

  .topbar {
    padding: 22px 32px 10px;
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 16px;
  }

  .topbar h2 {
    font-family: Syne, sans-serif;
    font-size: 1.15rem;
    font-weight: 700;
  }

  .status {
    font-size: 0.8rem;
    color: var(--dim);
  }
  .status.ready { color: var(--ok); }

  #feed {
    flex: 1;
    overflow-y: auto;
    padding: 8px 32px 24px;
  }

  .empty {
    margin-top: 12vh;
    max-width: 560px;
  }

  .empty h3 {
    font-family: Syne, sans-serif;
    font-size: clamp(2rem, 4vw, 2.8rem);
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1.05;
    margin-bottom: 12px;
  }

  .empty h3 em {
    font-style: normal;
    color: var(--accent);
  }

  .empty p {
    color: var(--muted);
    font-size: 1.05rem;
    line-height: 1.5;
    max-width: 42ch;
  }

  .card {
    background: var(--panel);
    border: 1px solid var(--line2);
    border-radius: var(--radius);
    padding: 20px 22px;
    margin-bottom: 16px;
    box-shadow: var(--shadow);
    animation: rise .35s ease;
  }

  @keyframes rise {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: none; }
  }

  .q-card {
    border-left: 3px solid var(--accent);
  }

  .q-card .meta {
    font-size: 0.75rem;
    color: var(--dim);
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 6px;
  }

  .q-card .text {
    font-size: 1.08rem;
    font-weight: 600;
  }

  .narrative {
    font-size: 1.02rem;
    line-height: 1.65;
    white-space: pre-wrap;
  }

  .stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 10px;
    margin-top: 14px;
  }

  .stat {
    background: var(--panel2);
    border-radius: 12px;
    padding: 12px 14px;
    border: 1px solid var(--line2);
  }

  .stat .k {
    font-size: 0.72rem;
    color: var(--dim);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 4px;
  }

  .stat .v {
    font-family: Syne, sans-serif;
    font-size: 1.45rem;
    font-weight: 700;
    color: var(--accent);
    line-height: 1.1;
  }

  .stat .s {
    font-size: 0.82rem;
    color: var(--muted);
    margin-top: 2px;
  }

  .section-h {
    font-family: Syne, sans-serif;
    font-size: 0.95rem;
    margin: 18px 0 10px;
  }

  table.tbl {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
  }

  table.tbl th, table.tbl td {
    text-align: left;
    padding: 8px 10px;
    border-bottom: 1px solid var(--line2);
  }

  table.tbl th {
    color: var(--dim);
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }

  .pill {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    background: rgba(184,230,58,.12);
    color: var(--accent);
  }

  .pill.clay { background: rgba(217,120,69,.18); color: #f0a57a; }
  .pill.grass { background: rgba(125,206,122,.15); color: #9fdc9c; }
  .pill.hard { background: rgba(120,160,220,.15); color: #a8c4ef; }

  .err-card {
    border-color: rgba(239,107,107,.35);
    color: #ffb4b4;
  }

  .loading {
    display: flex;
    align-items: center;
    gap: 12px;
    color: var(--muted);
  }

  .spinner {
    width: 18px; height: 18px;
    border: 2px solid var(--line2);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin .7s linear infinite;
  }

  @keyframes spin { to { transform: rotate(360deg); } }

  .composer {
    padding: 16px 32px 28px;
    border-top: 1px solid var(--line2);
    background: linear-gradient(180deg, transparent, rgba(8,17,14,.85) 30%);
  }

  .composer-box {
    display: flex;
    gap: 10px;
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 10px;
    box-shadow: var(--shadow);
  }

  .composer-box textarea {
    flex: 1;
    resize: none;
    border: none;
    outline: none;
    background: transparent;
    color: var(--text);
    font: inherit;
    font-size: 1rem;
    min-height: 52px;
    max-height: 140px;
    padding: 10px 12px;
  }

  .send {
    align-self: flex-end;
    background: var(--accent);
    color: var(--accent-ink);
    border: none;
    border-radius: 12px;
    padding: 12px 18px;
    font: inherit;
    font-weight: 700;
    cursor: pointer;
    white-space: nowrap;
  }

  .send:disabled {
    opacity: 0.45;
    cursor: not-allowed;
  }

  .hint {
    margin-top: 8px;
    font-size: 0.78rem;
    color: var(--dim);
  }

  @media (max-width: 900px) {
    body { grid-template-columns: 1fr; }
    #sidebar {
      position: relative;
      min-height: auto;
      border-right: none;
      border-bottom: 1px solid var(--line2);
    }
    #feed, .topbar, .composer { padding-left: 18px; padding-right: 18px; }
  }
</style>
</head>
<body>
<aside id="sidebar">
  <div class="brand">
    <div class="brand-mark">Tennis<span>Cross</span>Match</div>
    <p>Ask across any set of matches. Numbers come from the point tree — not the LLM.</p>
  </div>
  <div class="side-body">
    <div class="label">Scope mode</div>
    <div class="mode-toggle">
      <button class="mode-btn active" id="modeManual" type="button">Manual filters</button>
      <button class="mode-btn" id="modeAuto" type="button">Auto (LLM)</button>
    </div>

    <div id="manualFields">
      <label class="label" for="p1">Player 1</label>
      <input class="field" id="p1" placeholder="e.g. Jannik Sinner" value="Jannik Sinner" />

      <label class="label" for="p2">Player 2 (optional)</label>
      <input class="field" id="p2" placeholder="e.g. Carlos Alcaraz" value="Carlos Alcaraz" />

      <label class="check"><input type="checkbox" id="requireAll" checked /> Rivalry (both players in match)</label>

      <label class="label" for="surface">Surface</label>
      <select class="select" id="surface">
        <option value="">Any</option>
        <option value="clay">Clay</option>
        <option value="grass">Grass</option>
        <option value="hard">Hard</option>
      </select>

      <div class="label">Date range</div>
      <div class="row2">
        <input class="field" id="dateFrom" type="text" placeholder="YYYY-MM-DD" />
        <input class="field" id="dateTo" type="text" placeholder="YYYY-MM-DD" />
      </div>

      <label class="label" for="tournament">Tournament contains</label>
      <input class="field" id="tournament" placeholder="e.g. Wimbledon" />

      <label class="label" for="maxMatches">Max matches</label>
      <input class="field" id="maxMatches" type="number" value="60" min="1" max="200" />

      <button class="btn-ghost" id="previewBtn" type="button">Preview matches in scope</button>
      <div class="preview-box" id="previewBox"></div>
    </div>

    <div class="examples">
      <div class="label">Try asking</div>
      <button class="chip" type="button" data-q="How many aces did each player hit?">How many aces did each player hit?</button>
      <button class="chip" type="button" data-q="How many winners did each player hit?">How many winners did each player hit?</button>
      <button class="chip" type="button" data-q="How many points did each player win?">How many points did each player win?</button>
      <button class="chip" type="button" data-q="Compare aces by serve target for each player">Compare aces by serve target for each player</button>
      <button class="chip" type="button" data-q="How many times did crosscourt -> crosscourt -> down the line happen?">crosscourt → crosscourt → down the line</button>
    </div>
  </div>
</aside>

<main id="main">
  <div class="topbar">
    <h2>Cross-match analysis</h2>
    <div class="status" id="status">Starting engine…</div>
  </div>

  <div id="feed">
    <div class="empty" id="emptyState">
      <h3>Ask any tennis question.<br /><em>Across the whole corpus.</em></h3>
      <p>No match picker. Set a rivalry or surface on the left, then ask. Deterministic counts; LLM only explains.</p>
    </div>
  </div>

  <div class="composer">
    <div class="composer-box">
      <textarea id="question" rows="2" placeholder="Ask a cross-match question…"></textarea>
      <button class="send" id="sendBtn" type="button">Ask</button>
    </div>
    <div class="hint">Tip: Manual filters are more reliable than Auto when you know the players.</div>
  </div>
</main>

<script>
  let mode = "manual";
  let busy = false;

  const feed = document.getElementById("feed");
  const empty = document.getElementById("emptyState");
  const statusEl = document.getElementById("status");
  const qEl = document.getElementById("question");
  const sendBtn = document.getElementById("sendBtn");
  const previewBox = document.getElementById("previewBox");

  document.getElementById("modeManual").onclick = () => setMode("manual");
  document.getElementById("modeAuto").onclick = () => setMode("auto");

  function setMode(m) {
    mode = m;
    document.getElementById("modeManual").classList.toggle("active", m === "manual");
    document.getElementById("modeAuto").classList.toggle("active", m === "auto");
    document.getElementById("manualFields").style.opacity = m === "auto" ? "0.45" : "1";
  }

  document.querySelectorAll(".chip").forEach(btn => {
    btn.onclick = () => {
      qEl.value = btn.dataset.q;
      qEl.focus();
    };
  });

  async function checkHealth() {
    try {
      const r = await fetch("/api/health");
      const j = await r.json();
      if (j.ready) {
        statusEl.textContent = "Engine ready";
        statusEl.classList.add("ready");
      } else {
        statusEl.textContent = "Loading engine…";
        setTimeout(checkHealth, 1500);
      }
    } catch {
      statusEl.textContent = "Server offline";
      setTimeout(checkHealth, 2000);
    }
  }
  checkHealth();

  function scopePayload() {
    const players = [];
    const p1 = document.getElementById("p1").value.trim();
    const p2 = document.getElementById("p2").value.trim();
    if (p1) players.push(p1);
    if (p2) players.push(p2);
    return {
      mode,
      players,
      require_all_players: document.getElementById("requireAll").checked,
      surface: document.getElementById("surface").value,
      date_from: document.getElementById("dateFrom").value.trim(),
      date_to: document.getElementById("dateTo").value.trim(),
      tournament: document.getElementById("tournament").value.trim(),
      max_matches: parseInt(document.getElementById("maxMatches").value || "60", 10),
      explain: true,
    };
  }

  document.getElementById("previewBtn").onclick = async () => {
    previewBox.classList.add("show");
    previewBox.innerHTML = "Scanning matches…";
    try {
      const r = await fetch("/api/preview", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(scopePayload()),
      });
      const j = await r.json();
      if (!r.ok) throw new Error(j.error || "Preview failed");
      const sample = (j.matches || []).slice(0, 5).map(m =>
        `${m.date} · ${m.tournament}`
      ).join("<br/>");
      previewBox.innerHTML = `<strong>${j.n_matches}</strong> matches in scope` +
        (sample ? `<br/><br/>${sample}${j.truncated ? "<br/>…" : ""}` : "");
    } catch (e) {
      previewBox.innerHTML = e.message;
    }
  };

  function clearEmpty() {
    if (empty) empty.remove();
  }

  function addCard(html, cls="") {
    clearEmpty();
    const d = document.createElement("div");
    d.className = "card " + cls;
    d.innerHTML = html;
    feed.appendChild(d);
    d.scrollIntoView({ behavior: "smooth", block: "start" });
    return d;
  }

  function surfPill(s) {
    const t = (s || "").toLowerCase();
    const cls = t === "clay" ? "clay" : t === "grass" ? "grass" : "hard";
    return `<span class="pill ${cls}">${s || "—"}</span>`;
  }

  function renderResult(j) {
    const metricsHtml = Object.entries(j.metrics || {}).map(([name, rows]) => {
      const cells = (rows || []).map(r => `
        <div class="stat">
          <div class="k">${name.replace(/_/g," ")} · ${r.player}</div>
          <div class="v">${r.count}</div>
          <div class="s">${r.total ? (r.pct != null ? r.pct + "%" : "of " + r.total) : "count"}</div>
        </div>`).join("");
      return cells;
    }).join("");

    let patternsHtml = "";
    const pats = Object.entries(j.patterns || {});
    if (pats.length) {
      patternsHtml = `<div class="section-h">Patterns</div>` + pats.map(([sig, pr]) => {
        const c = pr.combined || {};
        const rows = (pr.by_player || []).map(r =>
          `<tr><td>${r.player}</td><td>${r.occurrences}</td><td>${r.wins}</td><td>${r.win_pct != null ? r.win_pct + "%" : "—"}</td></tr>`
        ).join("");
        return `<p style="color:var(--muted);margin-bottom:8px"><strong style="color:var(--text)">${sig}</strong>
          · ${c.occurrences || 0} occurrences · ${c.wins || 0} wins</p>
          <table class="tbl"><thead><tr><th>Player</th><th>Occ</th><th>Wins</th><th>Win%</th></tr></thead>
          <tbody>${rows}</tbody></table>`;
      }).join("");
    }

    let matchesHtml = "";
    if ((j.matches || []).length) {
      const rows = j.matches.map(m =>
        `<tr><td>${m.date || ""}</td><td>${m.tournament || ""}</td><td>${surfPill(m.surface)}</td>
         <td>${m.player1 || ""} vs ${m.player2 || ""}</td></tr>`
      ).join("");
      matchesHtml = `<div class="section-h">Matches analyzed (${j.n_matches})</div>
        <table class="tbl"><thead><tr><th>Date</th><th>Tournament</th><th>Surface</th><th>Players</th></tr></thead>
        <tbody>${rows}</tbody></table>`;
    }

    const scopeBits = [];
    if (j.scope) {
      if (j.scope.scope_type) scopeBits.push(j.scope.scope_type);
      if ((j.scope.players || []).length) scopeBits.push(j.scope.players.join(" · "));
      if (j.scope.surface) scopeBits.push(j.scope.surface);
    }
    const meta = [
      `${j.n_matches} matches`,
      j.elapsed_s != null ? `${j.elapsed_s}s` : null,
      scopeBits.join(" · ") || null,
    ].filter(Boolean).join(" · ");

    addCard(`
      <div class="meta" style="font-size:.75rem;color:var(--dim);text-transform:uppercase;letter-spacing:.07em;margin-bottom:8px">${meta}</div>
      ${j.narrative ? `<div class="narrative">${escapeHtml(j.narrative)}</div>` : `<div class="narrative" style="color:var(--muted)">No narrative generated — numbers below.</div>`}
      ${metricsHtml ? `<div class="stats-grid">${metricsHtml}</div>` : ""}
      ${patternsHtml}
      ${matchesHtml}
      ${(j.skipped || []).length ? `<p style="margin-top:12px;color:var(--warn);font-size:.85rem">${j.skipped.length} match(es) skipped</p>` : ""}
    `);
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;")
      .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }

  async function ask() {
    const question = qEl.value.trim();
    if (!question || busy) return;
    busy = true;
    sendBtn.disabled = true;

    addCard(`
      <div class="meta">Question</div>
      <div class="text">${escapeHtml(question)}</div>
    `, "q-card");

    const loading = addCard(`<div class="loading"><div class="spinner"></div> Analyzing across matches…</div>`);

    try {
      const payload = { ...scopePayload(), question };
      const r = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const j = await r.json();
      loading.remove();
      if (!r.ok) {
        addCard(`<div class="narrative">${escapeHtml(j.error || "Request failed")}</div>`, "err-card");
      } else {
        renderResult(j);
      }
    } catch (e) {
      loading.remove();
      addCard(`<div class="narrative">${escapeHtml(e.message)}</div>`, "err-card");
    } finally {
      busy = false;
      sendBtn.disabled = false;
      qEl.focus();
    }
  }

  sendBtn.onclick = ask;
  qEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      ask();
    }
  });
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(HTML)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "7861"))
    print(f"\n  TennisCrossMatch UI → http://127.0.0.1:{port}\n")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
