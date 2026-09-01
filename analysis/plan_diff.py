"""Emit a position plan containing only what has NOT already been resampled.

The Track A plan was cut twice to fit a wall-clock budget (per_trace_cap 22 -> 8), which
dropped most positions on the selected traces. Now that a third GPU is available we can
add the dropped ones back — but re-running the completed positions would waste hours, so
this diffs a full plan against the results already on disk and keeps only the remainder.

Same traces as the completed run, so prefix caching still helps and the results merge.

  python analysis/plan_diff.py --full analysis/positions_A_full.json \
      --run_dir runs/<name> --done counterfactual.json,counterfactual_above.json \
      --out analysis/positions_A_missing.json
"""

import json
from pathlib import Path

import fire


def main(full: str, run_dir: str, out: str, done: str = "counterfactual.json"):
    run_path = Path(run_dir)
    plan = json.loads(Path(full).read_text())

    completed = {}          # cond -> {(trace, k)}
    for name in done.split(","):
        p = run_path / name.strip()
        if not p.exists():
            print(f"  (no {p.name} — treating as nothing done there)")
            continue
        data = json.loads(p.read_text())
        for cond, recs in data.get("conditions", {}).items():
            for r in recs:
                completed.setdefault(cond, set()).add((r["trace"], r["k"]))
        print(f"  {p.name}: "
              f"{ {c: len(v) for c, v in data.get('conditions', {}).items()} }")

    out_plan = {"run_dir": run_dir, "purpose": "positions dropped by earlier budget cuts",
                "conditions": {}}
    total = 0
    for cond, traces in plan["conditions"].items():
        seen = completed.get(cond, set())
        kept = []
        for t in traces:
            missing = [k for k in t["positions"] if (t["trace"], k) not in seen]
            if not missing:
                continue
            kept.append({
                "trace": t["trace"], "n_units": t["n_units"],
                "positions": sorted(missing),
                "apriori": sorted(set(missing) & set(t.get("apriori", []))),
                "exploratory": sorted(set(missing) & set(t.get("exploratory", []))),
            })
            total += len(missing)
        out_plan["conditions"][cond] = kept
        print(f"{cond}: {sum(len(t['positions']) for t in kept)} missing positions "
              f"across {len(kept)} traces (already done: {len(seen)})")

    Path(out).write_text(json.dumps(out_plan, indent=2, default=str))
    print(f"\ntotal missing: {total}  (<= {2 * total} unique prefixes)")
    print(f"saved {out}")


if __name__ == "__main__":
    fire.Fire(main)
