"""Verify the k=-1 forced-answer medians: is the baseline/incentive gap real, or a parse
artifact? Prints the stored sample texts and the full distribution per condition."""
import json, sys
from pathlib import Path
import numpy as np

p = Path(sys.argv[1], "forced_answer_k-1.json")
d = json.loads(p.read_text())
print(f"threshold = {d['threshold']:,}   count/condition = {d['count']}")

for cond, r in d["conditions"].items():
    print(f"\n=== {cond} ===")
    print(f"  parsed {r['n_parsed']}/{r['n']}   P(>thr)={r['p_above_threshold']:.3f}   "
          f"median={r['median']:,.0f}   rel_dev={r['median_rel_deviation']:+.3f}")
    print("  sample raw completions the parser saw:")
    for t in r.get("sample_texts", []):
        print(f"    | {t}")
