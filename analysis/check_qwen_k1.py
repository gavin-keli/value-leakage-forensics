"""Is Qwen's 550-billion baseline median a real model output or a parse artifact?

Re-parses the stored completions and prints the value distribution plus the raw text, so
the number in the write-up can be checked rather than trusted.
"""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "analysis")
from forced_answer import parse_number

d = json.loads(Path(sys.argv[1], "forced_answer_k-1_v3.json").read_text())
print(f"threshold = {d['threshold']:,}")
for cond, r in d["conditions"].items():
    print(f"\n=== {cond} ===")
    print(f"  n_parsed={r['n_parsed']}  truncated={r.get('n_truncated_dropped', '?')}  "
          f"median={r['median']:,.0f}  P(>thr)={r['p_above_threshold']:.3f}")
    print("  raw completions stored (first 3):")
    for t in r.get("sample_texts", []):
        v = parse_number(t)
        print(f"    parsed={v if v is None else format(v, ',.0f')}")
        print(f"      | {t[:150]}")
