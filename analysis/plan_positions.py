"""Freeze the step-5 position plan BEFORE any counterfactual result is inspected.

Two ledgers, kept separate on purpose:

  apriori      uniform grid + every hypothesis-relevant tagged unit + matched untagged
               controls. Chosen without reference to any outcome, so these carry the
               hypothesis tests and the p-values.
  exploratory  largest |dP| positions from the forced-answer screen. Selected BECAUSE of
               an outcome, so effect sizes here are subject to winner's curse and are
               reported as "the largest effects we found", with no significance claimed.

  python analysis/plan_positions.py --run_dir runs/<name> --out analysis/positions_B.json
"""

import json
from pathlib import Path

import fire
import numpy as np

CONDS = ("below_good", "above_good")
TAGS_OF_INTEREST = ("disavowal", "threshold-comparison", "parameter-selection",
                    "directional-sensitivity-check", "incentive-acknowledgment")


def matched_control(units, k, used):
    """An untagged neighbour within +-2 of similar length — controls the position-in-trace
    confound the paper flags in its Discussion."""
    target = len(units[k]["text"])
    best, best_score = None, None
    for d in (-2, -1, 1, 2):
        j = k + d
        if j < 1 or j >= len(units) or j in used or units[j]["tags"]:
            continue
        score = abs(len(units[j]["text"]) - target)
        if best_score is None or score < best_score:
            best, best_score = j, score
    return best


def main(run_dir: str, out: str, n_traces: int = 6, grid: int = 10,
         max_tagged: int = 8, n_exploratory: int = 4, per_trace_cap: int = 22):
    run_path = Path(run_dir)
    screen = json.loads((run_path / "screen.json").read_text())
    rng = np.random.default_rng(0)

    plan = {"run_dir": run_dir, "ledger": "frozen before any counterfactual result",
            "conditions": {}}
    summary = {}

    for cond in CONDS:
        segs = json.loads((run_path / f"segments_{cond}.json").read_text())
        segs = [s for s in segs if s["n_units"] >= 12][:n_traces]

        # exploratory nominations from the screen, per trace
        byt = {}
        for r in screen["conditions"][cond]:
            if r["trace"] is not None and r["p_above"] is not None:
                byt.setdefault(r["trace"], []).append(r)
        jumps = {}
        for t, rs in byt.items():
            rs.sort(key=lambda r: r["k"])
            js = [(abs(cur["p_above"] - prev["p_above"]), cur["k"])
                  for prev, cur in zip(rs, rs[1:])]
            js.sort(reverse=True)
            jumps[t] = [k for _, k in js[:n_exploratory]]

        traces_out, n_ap, n_ex = [], 0, 0
        for s in segs:
            units, n = s["units"], s["n_units"]
            apriori = {int(x) for x in
                       np.unique(np.linspace(1, n - 1, min(grid, n - 1)).astype(int))}

            tagged = [u["i"] for u in units
                      if any(t in u["tags"] for t in TAGS_OF_INTEREST) and u["i"] >= 1]
            if len(tagged) > max_tagged:            # spread, don't take the first N
                idx = np.unique(np.linspace(0, len(tagged) - 1, max_tagged).astype(int))
                tagged = [tagged[int(i)] for i in idx]
            apriori |= set(tagged)

            controls = set()
            for k in tagged:
                c = matched_control(units, k, apriori | controls)
                if c is not None:
                    controls.add(int(c))
            apriori |= controls

            explor = {int(k) for k in jumps.get(s["i"], []) if k >= 1} - apriori

            keep = sorted(apriori)
            if len(keep) > per_trace_cap:
                sel = np.unique(np.linspace(0, len(keep) - 1, per_trace_cap).astype(int))
                keep = [keep[int(i)] for i in sel]
            positions = sorted(set(keep) | explor)

            traces_out.append({
                "trace": s["i"], "n_units": n,
                "positions": positions,
                "apriori": sorted(set(keep)),
                "exploratory": sorted(explor),
                "tagged": sorted(set(tagged) & set(keep)),
                "controls": sorted(controls & set(keep)),
            })
            n_ap += len(keep)
            n_ex += len(explor)

        plan["conditions"][cond] = traces_out
        total_pos = sum(len(t["positions"]) for t in traces_out)
        summary[cond] = {"traces": len(traces_out), "positions": total_pos,
                         "apriori": n_ap, "exploratory": n_ex}
        print(f"{cond}: {len(traces_out)} traces, {total_pos} positions "
              f"({n_ap} a-priori, {n_ex} exploratory)")

    plan["summary"] = summary
    total = sum(v["positions"] for v in summary.values())
    print(f"\ntotal positions: {total}")
    print(f"unique prefixes to generate (<= 2 per position): <= {2 * total}")

    Path(out).write_text(json.dumps(plan, indent=2, default=str))
    print(f"saved {out}")


if __name__ == "__main__":
    fire.Fire(main)
