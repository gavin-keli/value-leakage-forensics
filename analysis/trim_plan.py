"""Trim a position plan to a wall-clock budget, spreading the cut evenly.

Takes every position if it fits, otherwise keeps an evenly-spaced subset per trace so
coverage stays uniform across the trace rather than clustering at one end. A-priori and
exploratory membership is preserved for whatever survives.

  python analysis/trim_plan.py --plan X.json --out Y.json --cap 12
"""

import json
from pathlib import Path

import fire
import numpy as np


def main(plan: str, out: str, cap: int = 12, conditions: str | None = None):
    p = json.loads(Path(plan).read_text())
    wanted = ([c.strip() for c in conditions.split(",")] if conditions
              else list(p["conditions"]))
    total = 0
    for cond in list(p["conditions"]):
        if cond not in wanted:
            del p["conditions"][cond]
            continue
        for t in p["conditions"][cond]:
            pos = t["positions"]
            if len(pos) > cap:
                idx = np.unique(np.linspace(0, len(pos) - 1, cap).astype(int))
                pos = [pos[int(i)] for i in idx]
                t["positions"] = pos
                t["apriori"] = sorted(set(pos) & set(t.get("apriori", [])))
                t["exploratory"] = sorted(set(pos) & set(t.get("exploratory", [])))
            total += len(pos)
        print(f"{cond}: {sum(len(t['positions']) for t in p['conditions'][cond])} "
              f"positions across {len(p['conditions'][cond])} traces")
    Path(out).write_text(json.dumps(p, indent=2, default=str))
    print(f"total {total} positions (<= {2 * total} prefixes)\nsaved {out}")


if __name__ == "__main__":
    fire.Fire(main)
