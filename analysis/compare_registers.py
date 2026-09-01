"""Show how each model handles the same moment: the first time it engages the bet.

Prints a contiguous window of units around the first incentive-acknowledgment, so the
reasoning register is visible as flow rather than as cherry-picked sentences.
"""
import json, sys
from pathlib import Path

run_dir, cond = sys.argv[1], sys.argv[2]
n_traces = int(sys.argv[3]) if len(sys.argv) > 3 else 2
window = int(sys.argv[4]) if len(sys.argv) > 4 else 6

segs = json.loads(Path(run_dir, f"segments_{cond}.json").read_text())

shown = 0
for s in segs:
    units = s["units"]
    first = next((u["i"] for u in units
                  if "incentive-acknowledgment" in u["tags"]), None)
    if first is None or first < 2:
        continue
    lo, hi = max(0, first - 1), min(len(units), first + window)
    print(f"\n{'='*78}\ntrace {s['i']} — units {lo}..{hi-1} of {len(units)} "
          f"(first incentive mention at {first})\n{'='*78}")
    for u in units[lo:hi]:
        tags = ",".join(t.replace("incentive-acknowledgment", "INC")
                         .replace("disavowal", "DISAVOWAL")
                         .replace("parameter-selection", "param")
                         .replace("threshold-comparison", "thr")
                        for t in u["tags"])
        mark = f"  [{tags}]" if tags else ""
        print(f"  {u['i']:>4}: {u['text'][:150]}{mark}")
    shown += 1
    if shown >= n_traces:
        break
