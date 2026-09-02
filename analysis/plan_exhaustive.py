"""Build an EXHAUSTIVE position plan: every unit of every selected trace.

For short traces this is both cheaper and cleaner than sampling positions:

  cheaper  taking every position makes the prefixes fully nested, so a trace with n units
           needs n+1 generation sets rather than 2n for scattered points.
  cleaner  there is no position selection at all, so the a-priori / exploratory ledger
           distinction disappears — nothing here can be selected on an outcome, and no
           winner's-curse caveat applies to the result.

Unit 0 is skipped: resampling position k needs a prefix ending at k-1, and for k=0 that is
the empty trace, which is the forced-answer probe rather than a counterfactual.

  python analysis/plan_exhaustive.py --run_dir runs/<name> --out analysis/positions_x.json
"""

import json
from pathlib import Path

import fire

CONDS = ("below_good", "above_good")


def main(run_dir: str, out: str, n_traces: int = 100, min_units: int = 3):
    run_path = Path(run_dir)
    plan = {"run_dir": run_dir, "selection": "exhaustive (all positions, no sampling)",
            "conditions": {}}
    total = 0

    for cond in CONDS:
        segs = json.loads((run_path / f"segments_{cond}.json").read_text())
        usable = [s for s in segs if s["n_units"] >= min_units][:n_traces]
        traces = []
        for s in usable:
            positions = [u["i"] for u in s["units"] if u["i"] >= 1]
            if not positions:
                continue
            traces.append({
                "trace": s["i"], "n_units": s["n_units"],
                "positions": positions,
                "apriori": positions,     # exhaustive: every position is a-priori
                "exploratory": [],
            })
            total += len(positions)
        plan["conditions"][cond] = traces
        n_pos = sum(len(t["positions"]) for t in traces)
        prefixes = sum(len(t["positions"]) + 1 for t in traces)
        print(f"{cond}: {len(traces)} traces, {n_pos} positions, "
              f"~{prefixes} unique prefixes (nested)")

    Path(out).write_text(json.dumps(plan, indent=2, default=str))
    print(f"\ntotal positions: {total}")
    print(f"saved {out}")


if __name__ == "__main__":
    fire.Fire(main)
