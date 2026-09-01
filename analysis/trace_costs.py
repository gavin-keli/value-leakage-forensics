"""How expensive is each candidate trace to resample? Cost scales with the tokens a
continuation must still generate, i.e. with trace length."""
import json, sys
from pathlib import Path

run_dir = sys.argv[1]
plan_file = sys.argv[2] if len(sys.argv) > 2 else None

for cond in ("below_good", "above_good"):
    segs = json.loads(Path(run_dir, f"segments_{cond}.json").read_text())
    rows = json.loads(Path(run_dir, f"{cond}.json").read_text())["rows"]
    lens = [(s["i"], s["n_units"], len(rows[s["i"]]["reasoning"])) for s in segs]
    ok = [x for x in lens if x[1] >= 12]
    ok_sorted = sorted(ok, key=lambda x: x[2])
    print(f"\n{cond}: {len(ok)} usable traces")
    print(f"  chars: min={ok_sorted[0][2]:,} median={ok_sorted[len(ok_sorted)//2][2]:,} "
          f"max={ok_sorted[-1][2]:,}")
    print(f"  currently selected (first 3 by index): "
          f"{[(i, u, c) for i, u, c in ok[:3]]}")
    print(f"  shortest 3: {[(i, u, c) for i, u, c in ok_sorted[:3]]}")
    sel_cost = sum(c for _, _, c in ok[:3])
    short_cost = sum(c for _, _, c in ok_sorted[:3])
    print(f"  total chars: selected={sel_cost:,}  shortest={short_cost:,}  "
          f"speedup={sel_cost/max(1,short_cost):.1f}x")
