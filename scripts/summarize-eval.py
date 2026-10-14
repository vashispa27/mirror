#!/usr/bin/env python3
"""Summarize a saved eval report against the dataset's scenario labels."""

import json
import sys
from collections import Counter, defaultdict

run_path, dataset_path = sys.argv[1], sys.argv[2]

report = json.load(open(run_path, encoding="utf-8"))
print("top-level keys:", sorted(report.keys()))

rows = [json.loads(line) for line in open(dataset_path, encoding="utf-8") if line.strip()]
by_query = {r["query"]: r for r in rows}

# Aggregate table
agg = report.get("aggregate") or report.get("modes") or {}
if isinstance(agg, dict) and agg and isinstance(next(iter(agg.values())), dict):
    print(f"\n{'mode':<14}{'succ@5':>8}{'recall':>8}{'mrr':>8}{'ndcg':>8}")
    for mode, m in sorted(agg.items()):
        print(f"{mode:<14}{m.get('success_at_k', 0):>8.4f}{m.get('recall_at_k', 0):>8.4f}"
              f"{m.get('mrr_at_k', 0):>8.4f}{m.get('ndcg_at_k', 0):>8.4f}")

# Per-scenario hit/miss for the best hybrid mode
mode = "hybrid_dbsf"
hits = defaultdict(int)
total = defaultdict(int)
for q in report.get("per_query", []):
    scenario = by_query.get(q["query"], {}).get("scenario", "base")
    total[scenario] += 1
    if q.get("modes", {}).get(mode, {}).get("success_at_k", 0) >= 1.0:
        hits[scenario] += 1

print(f"\nper-scenario success@5 ({mode}):")
print(f"{'scenario':<28}{'hit/total':>12}{'rate':>8}")
for scenario in sorted(total):
    print(f"{scenario:<28}{hits[scenario]:>6}/{total[scenario]:<5}"
          f"{hits[scenario] / total[scenario]:>8.2f}")

# Which scenarios carry the misses
misses = Counter({s: total[s] - hits[s] for s in total})
print("\ntotal misses by scenario:")
for scenario, count in misses.most_common():
    if count:
        print(f"  {scenario:<28}{count}")
