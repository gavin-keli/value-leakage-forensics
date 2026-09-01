"""Trajectory-judge null rate by condition, across every run.

If the incentive conditions lose systematically more rollouts than baseline, MRF is
computed over a non-random subset and the missingness is itself a result.
"""
import json, glob, os
from pathlib import Path

print(f"{'run':<38} {'baseline':>10} {'below':>10} {'above':>10}   n_kept basis")
for d in sorted(glob.glob("runs/*")):
    p = Path(d, "trajectories.json")
    if not p.exists():
        continue
    t = json.loads(p.read_text())
    cells = []
    for c in ("baseline", "below_good", "above_good"):
        vals = t.get(c) or []
        if not vals:
            cells.append("     -")
            continue
        nulls = sum(1 for v in vals if v is None)
        cells.append(f"{nulls/len(vals):>9.0%}")
    print(f"{os.path.basename(d):<38} {cells[0]:>10} {cells[1]:>10} {cells[2]:>10}")
