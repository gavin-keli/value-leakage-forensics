"""Offline diagnosis — no API calls.

Why did the retry pass produce no output for some runs? And are some judge nulls actually
FAILED SAMPLES rather than unparseable traces? sample.py records a failed rollout as
{"i": i, "error": ...} with no 'reasoning' key at all.
"""
import json, glob, os
from pathlib import Path

CONDITIONS = ("baseline", "below_good", "above_good")

print(f"{'run':<40} {'cond':<11} {'rows':>5} {'err_rows':>9} {'no_reasoning':>13} "
      f"{'nulls':>6} {'nulls_that_are_err':>19}")
for d in sorted(glob.glob("runs/*")):
    tpath = Path(d, "trajectories.json")
    if not tpath.exists():
        continue
    traj = json.loads(tpath.read_text())
    for c in CONDITIONS:
        cpath = Path(d, f"{c}.json")
        if not cpath.exists():
            continue
        rows = json.loads(cpath.read_text())["rows"]
        vals = traj.get(c) or []
        err_rows = [r["i"] for r in rows if "error" in r]
        no_reason = [r["i"] for r in rows if "reasoning" not in r]
        nulls = {i for i, v in enumerate(vals) if v is None}
        nulls_err = len(nulls & set(err_rows))
        if err_rows or no_reason or nulls:
            print(f"{os.path.basename(d):<40} {c:<11} {len(rows):>5} "
                  f"{len(err_rows):>9} {len(no_reason):>13} {len(nulls):>6} "
                  f"{nulls_err:>19}")
