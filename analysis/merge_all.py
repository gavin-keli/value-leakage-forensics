"""Merge several counterfactual result files, and their position plans, into one dataset.

Track A's positions were collected in two passes (an initial trimmed plan, then the
positions that trim had dropped) across three machines. Both passes used identical settings
— same model, same R, same threshold — which this verifies before merging rather than
assuming. The position ledgers are merged too, so the a-priori / exploratory distinction
survives into the combined analysis.

  python analysis/merge_all.py --run_dir runs/<name> \
      --files counterfactual_merged.json,counterfactual_below_extra.json,counterfactual_above_extra.json \
      --plans analysis/positions_A.json,analysis/positions_full_missing.json \
      --out_cf counterfactual_all.json --out_plan analysis/positions_all.json
"""

import json
from pathlib import Path

import fire


def main(run_dir: str, files: str, plans: str, out_cf: str, out_plan: str):
    run_path = Path(run_dir)

    merged, base = {}, None
    for name in files.split(","):
        p = run_path / name.strip()
        if not p.exists():
            print(f"  skip (missing): {p.name}")
            continue
        d = json.loads(p.read_text())
        if base is None:
            base = {k: d.get(k) for k in ("threshold", "model", "family", "R",
                                          "cos_threshold")}
        else:
            for k, v in base.items():
                if d.get(k) != v:
                    raise SystemExit(f"SETTING MISMATCH in {p.name}: {k} "
                                     f"{d.get(k)!r} != {v!r}")
        for cond, recs in d.get("conditions", {}).items():
            seen = {(r["trace"], r["k"]) for r in merged.get(cond, [])}
            added = [r for r in recs if (r["trace"], r["k"]) not in seen]
            merged.setdefault(cond, []).extend(added)
            print(f"  {p.name:<42} {cond:<11} +{len(added):>3} "
                  f"(dupes skipped: {len(recs) - len(added)})")

    out = dict(base or {}, conditions=merged)
    (run_path / out_cf).write_text(json.dumps(out, indent=2, default=str))
    print(f"\nmerged positions: { {c: len(v) for c, v in merged.items()} }")
    print(f"saved {run_path / out_cf}")

    # merge the position ledgers so a-priori / exploratory labels survive
    plan_out = {"conditions": {}}
    for name in plans.split(","):
        p = Path(name.strip())
        if not p.exists():
            print(f"  skip plan (missing): {p}")
            continue
        d = json.loads(p.read_text())
        for cond, traces in d.get("conditions", {}).items():
            bucket = plan_out["conditions"].setdefault(cond, {})
            for t in traces:
                e = bucket.setdefault(t["trace"], {"trace": t["trace"],
                                                   "n_units": t["n_units"],
                                                   "positions": [], "apriori": [],
                                                   "exploratory": []})
                for key in ("positions", "apriori", "exploratory"):
                    e[key] = sorted(set(e[key]) | set(t.get(key, [])))
    plan_out["conditions"] = {c: list(v.values()) for c, v in plan_out["conditions"].items()}
    Path(out_plan).write_text(json.dumps(plan_out, indent=2, default=str))
    tot = {c: sum(len(t["positions"]) for t in v)
           for c, v in plan_out["conditions"].items()}
    print(f"merged plan positions: {tot}")
    print(f"saved {out_plan}")


if __name__ == "__main__":
    fire.Fire(main)
