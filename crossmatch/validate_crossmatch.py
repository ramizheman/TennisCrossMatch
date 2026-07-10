"""Cross-match regression validation on the Sinner-Alcaraz rivalry.

Five channels, all numbers from each match's point tree (reducer only sums):
  1. METRIC   aces               -> Alcaraz 101 / Sinner 93 ; US Open F 12 / 2
  2. PATTERN  forehand->forehand -> US Open F Sinner 40 / Alcaraz 51
     + WIN%    (needs [Point won by:] tag injected on the JSON fan-out path)
  3. GROUPED  aces by serve_target -> bucket sums == overall ; US Open F 12 / 2
  4. NARRATIVE explanation is grounded (headline numbers echoed, none fabricated)
  5. SCOPE    LLM scope planner routes the rivalry question to 20 matches

Run:  python -m crossmatch.validate_crossmatch   (from the Tennis NL repo root)
"""
import os, sys, re, json
from collections import defaultdict
from dotenv import load_dotenv
load_dotenv()
# repo root is the parent of the crossmatch package dir
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from crossmatch import (CrossMatchEngine, rank_players_by_metric,
                        rank_players_by_pattern, grouped_table,
                        build_dossier, narrate, plan_scope)

RIVALRY = ["Jannik Sinner", "Carlos Alcaraz"]


def _manual_metric(iros, metric):
    m = defaultdict(int)
    for iro in iros:
        r = iro.metrics.get(metric)
        if r:
            for p, s in r.by_player.items():
                m[p] += s.count
    return m


def _manual_pattern(iros, sig):
    m = defaultdict(int)
    for iro in iros:
        r = iro.patterns.get(sig)
        if r:
            for p, s in r.by_player.items():
                m[p] += s.occurrences
    return m


def main():
    engine = CrossMatchEngine()
    ok = True

    # ---- 1) METRIC channel: aces ----
    reduced, iros = engine.run("How many aces did each player hit?",
                               players=RIVALRY, require_all_players=True)
    print("\n=== 1) METRIC: aces across the rivalry ===")
    for row in rank_players_by_metric(reduced, "aces", by="count"):
        print(f"  {row['player']:16} {row['count']:>4} aces / {row['total']:>4} ({row['pct']}%)")
    man = _manual_metric(iros, "aces")
    for p, s in reduced.metrics["aces"].by_player.items():
        good = man[p] == s.count
        ok = ok and good
        print(f"  check {p:16} reducer={s.count} manual={man[p]} {'OK' if good else 'MISMATCH'}")
    uso = next((i for i in iros if i.provenance.date == "2025-09-07"), None)
    if uso:
        a = uso.metrics["aces"].by_player
        good = (a.get("Carlos Alcaraz").count == 12) and (a.get("Jannik Sinner").count == 2)
        ok = ok and good
        print(f"  US Open F aces Alcaraz/Sinner = 12/2 ? {'OK' if good else 'MISMATCH'}")

    # ---- 4) NARRATIVE: grounded explanation of the aces result ----
    print("\n=== 4) NARRATIVE: grounded explanation (aces) ===")
    dossier = build_dossier(reduced, iros=iros)
    text = narrate(reduced, question="How many aces did each player hit in the rivalry?",
                   iros=iros, dossier=dossier)
    allowed = set(int(x) for x in re.findall(r"\b\d+\b", json.dumps(dossier, default=str)))
    fabricated = [int(x) for x in re.findall(r"\b\d+\b", text) if int(x) >= 10 and int(x) not in allowed]
    aces = {r["player"]: r["count"] for r in rank_players_by_metric(reduced, "aces")}
    good = bool(text) and all(str(v) in text for v in aces.values()) and not fabricated
    ok = ok and good
    print("  " + (text[:220].replace("\n", " ") + ("..." if len(text) > 220 else "")))
    print(f"  non-empty + headline counts echoed + no fabricated ints? "
          f"{'OK' if good else 'FAIL ' + str(fabricated)}")

    # ---- 2) PATTERN channel: forehand -> forehand (occurrences + win%) ----
    sig = "forehand -> forehand"
    reduced_p, iros_p = engine.run("How many times did forehand -> forehand occur",
                                   players=RIVALRY, require_all_players=True)
    print("\n=== 2) PATTERN: forehand -> forehand across the rivalry ===")
    for row in rank_players_by_pattern(reduced_p, sig, by="occurrences"):
        print(f"  {row['player']:16} occ={row['occurrences']:>4} wins={row['wins']:>4} win%={row['win_pct']}")
    manp = _manual_pattern(iros_p, sig)
    rp = reduced_p.patterns.get(sig)
    if rp:
        for p, s in rp.by_player.items():
            good = manp[p] == s.occurrences
            ok = ok and good
            print(f"  check {p:16} reducer={s.occurrences} manual={manp[p]} {'OK' if good else 'MISMATCH'}")
    usp = next((i for i in iros_p if i.provenance.date == "2025-09-07"), None)
    if usp and usp.patterns.get(sig):
        bp = usp.patterns[sig].by_player
        s_sin, s_alc = bp.get("Jannik Sinner"), bp.get("Carlos Alcaraz")
        good = (s_sin.occurrences == 40) and (s_alc.occurrences == 51)
        ok = ok and good
        print(f"  US Open F fh->fh Sinner/Alcaraz = 40/51 ? {'OK' if good else 'MISMATCH'}")
        # win% now populated from injected [Point won by:] tag (was 0 before)
        win_good = (s_sin.wins == 23) and (s_alc.wins == 28) and s_sin.win_pct == 57.5 and s_alc.win_pct == 54.9
        ok = ok and win_good
        print(f"  US Open F fh->fh wins Sinner/Alcaraz = 23/28 (57.5%/54.9%) ? "
              f"{'OK' if win_good else 'MISMATCH ' + str((s_sin.wins, s_alc.wins, s_sin.win_pct, s_alc.win_pct))}")

    # ---- 3) GROUPED channel: aces by serve_target ----
    print("\n=== 3) GROUPED: aces by serve_target across the rivalry ===")
    reduced_g, iros_g = engine.run("How many aces did each player hit to each serve target?",
                                   players=RIVALRY, require_all_players=True)
    bucket_sum = defaultdict(int)
    for r in grouped_table(reduced_g, "serve_target", "aces"):
        parts = ", ".join(f"{p}={v['count']}" for p, v in r["by_player"].items())
        print(f"  serve_target={r['bucket']:5} total={r['combined']['count']:>3} | {parts}")
        for p, v in r["by_player"].items():
            bucket_sum[p] += v["count"]
    overall = {r["player"]: r["count"] for r in rank_players_by_metric(reduced_g, "aces")}
    for p, tot in overall.items():
        good = bucket_sum.get(p, 0) == tot
        ok = ok and good
        print(f"  check {p:16} buckets_sum={bucket_sum.get(p,0)} overall={tot} {'OK' if good else 'MISMATCH'}")
    usg = next((i for i in iros_g if i.provenance.date == "2025-09-07"), None)
    if usg and "serve_target" in usg.grouped:
        per = defaultdict(int)
        for metrics in usg.grouped["serve_target"].buckets.values():
            mr = metrics.get("aces")
            if mr:
                for p, s in mr.by_player.items():
                    per[p] += s.count
        good = per.get("Carlos Alcaraz") == 12 and per.get("Jannik Sinner") == 2
        ok = ok and good
        print(f"  US Open F serve_target ace sums Alcaraz/Sinner = 12/2 ? {'OK' if good else 'MISMATCH'}")

    # ---- 5) SCOPE planner routing ----
    print("\n=== 5) SCOPE planner ===")
    sp = plan_scope("How many aces did each player hit in the Sinner Alcaraz rivalry?")
    good = sp.scope_type == "rivalry" and sp.require_all_players and len(sp.players) == 2
    ok = ok and good
    print(f"  scope={sp.scope_type} players={sp.players} require_all={sp.require_all_players} "
          f"residual={sp.residual_question!r} {'OK' if good else 'MISMATCH'}")

    print("\nALL CROSS-CHECKS PASSED" if ok else "\nCROSS-CHECK FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
