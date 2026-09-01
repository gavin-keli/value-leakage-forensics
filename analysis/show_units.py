"""Print how a model actually talks about the incentive, so tag heuristics can be
checked against real phrasing rather than assumed."""
import json, random, sys
from pathlib import Path

run_dir, condition = sys.argv[1], sys.argv[2]
tag = sys.argv[3] if len(sys.argv) > 3 else "incentive-acknowledgment"
n = int(sys.argv[4]) if len(sys.argv) > 4 else 25

segs = json.loads(Path(run_dir, f"segments_{condition}.json").read_text())
hits = [(s["i"], u) for s in segs for u in s["units"] if tag in u["tags"]]
print(f"{len(hits)} units tagged {tag!r} across {len(segs)} traces\n")

random.seed(0)
for trace_i, u in random.sample(hits, min(n, len(hits))):
    print(f"[trace {trace_i:>3} unit {u['i']:>3}] {u['text'][:190]}")
