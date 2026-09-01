"""How much disavowal material does each model actually have, per condition?

Determines whether a disavowal test is even runnable on a given model.
"""
import json, glob, os
from pathlib import Path

for run_dir in sorted(glob.glob("runs/local-*")):
    print(f"\n=== {os.path.basename(run_dir)} ===")
    for cond in ("baseline", "below_good", "above_good"):
        p = Path(run_dir, f"segments_{cond}.json")
        if not p.exists():
            continue
        segs = json.loads(p.read_text())
        n_units = sum(1 for s in segs for u in s["units"] if "disavowal" in u["tags"])
        n_traces = sum(1 for s in segs
                       if any("disavowal" in u["tags"] for u in s["units"]))
        print(f"  {cond:<11} {n_units:>5} disavowal units across "
              f"{n_traces:>3}/{len(segs)} traces  "
              f"({n_units/max(1,len(segs)):.2f} per trace)")
