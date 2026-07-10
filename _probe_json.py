import json

fp = r"C:\Users\lisas\OneDrive\Documents\Tennis Strategy\strategy_app\data\per_match_json\2025-09-07_US_Open_Jannik_Sinner_vs_Carlos_Alcaraz.json"
with open(fp, encoding="utf-8") as f:
    d = json.load(f)

pts = d["scraped"]["point_by_point"]
rows = pts["pointlog_rows"]
print("total rows:", len(rows))
print("total_points field:", pts.get("total_points"))
print("match_result:", pts.get("match_result"))
print()

# Show first 5 rows
for i, row in enumerate(rows[:5]):
    srv = row.get("server", "")
    desc = row.get("description", "")
    print(f"Row {i}: server={srv!r}")
    print(f"        desc  ={desc[:120]!r}")
    print()

# Sample aces
aces = [r for r in rows if "ace" in (r.get("description") or "").lower()]
print("Ace rows found:", len(aces))
for r in aces[:5]:
    print(" ", r.get("server", ""), "|", (r.get("description") or "")[:100])
