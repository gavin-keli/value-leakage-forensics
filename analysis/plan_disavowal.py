"""Targeted position plan for THE disavowal test.

The general plan spreads positions across all tags and a uniform grid; after the budget
cuts that left zero disavowal units, which is precisely the claim Track A exists to test:
Qwen3.5 asserts honesty 5.2-5.6 times per trace — does asserting it change what it answers?

So: pick the traces richest in disavowal units, take disavowal positions, and pair each
with a matched untagged control at similar depth and length. Controls are essential —
without them a null on disavowal is indistinguishable from "nothing matters here".

  python analysis/plan_disavowal.py --run_dir runs/<name> --out analysis/positions_dis.json
"""

import json
from pathlib import Path

import fire
import numpy as np

CONDS = ("below_good", "above_good")


def main(run_dir: str, out: str, n_traces: int = 2, per_trace: int = 5,
         max_chars: int = 26000):
    run_path = Path(run_dir)
    plan = {"run_dir": run_dir, "purpose": "disavowal causal test", "conditions": {}}

    for cond in CONDS:
        segs = json.loads((run_path / f"segments_{cond}.json").read_text())
        rows = json.loads((run_path / f"{cond}.json").read_text())["rows"]

        cands = []
        for s in segs:
            n_dis = sum(1 for u in s["units"] if "disavowal" in u["tags"])
            chars = len(rows[s["i"]]["reasoning"])
            if n_dis >= 2 and chars <= max_chars:      # cost cap: these are long traces
                cands.append((n_dis, -chars, s))
        # richest in disavowal first, then cheapest
        cands.sort(key=lambda c: (-c[0], -c[1]))
        chosen = [c[2] for c in cands[:n_traces]]

        traces_out = []
        for s in chosen:
            units = s["units"]
            dis = [u["i"] for u in units if "disavowal" in u["tags"] and u["i"] >= 1]
            if len(dis) > per_trace:
                idx = np.unique(np.linspace(0, len(dis) - 1, per_trace).astype(int))
                dis = [dis[int(i)] for i in idx]

            controls = []
            for k in dis:
                target = len(units[k]["text"])
                best, score = None, None
                for d in (-3, -2, 2, 3):
                    j = k + d
                    if j < 1 or j >= len(units) or units[j]["tags"]:
                        continue
                    sc = abs(len(units[j]["text"]) - target)
                    if score is None or sc < score:
                        best, score = j, sc
                if best is not None and best not in controls and best not in dis:
                    controls.append(int(best))

            positions = sorted(set(dis) | set(controls))
            traces_out.append({
                "trace": s["i"], "n_units": s["n_units"],
                "positions": positions,
                "apriori": positions,          # every position here is a-priori by design
                "exploratory": [],
                "disavowal": sorted(dis), "controls": sorted(controls),
            })
            print(f"  {cond} trace {s['i']}: {len(dis)} disavowal + "
                  f"{len(controls)} controls, {s['n_units']} units, "
                  f"{len(rows[s['i']]['reasoning']):,} chars")

        plan["conditions"][cond] = traces_out

    total = sum(len(t["positions"]) for c in plan["conditions"].values() for t in c)
    print(f"\ntotal positions: {total}  (<= {2 * total} unique prefixes)")
    Path(out).write_text(json.dumps(plan, indent=2, default=str))
    print(f"saved {out}")


if __name__ == "__main__":
    fire.Fire(main)
