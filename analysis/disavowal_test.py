"""THE disavowal test.

Qwen3.5 asserts honesty 5.2-5.6 times per incentive-condition trace ("I must provide the
honest scientific estimate", "not 'strategically manipulate the bet'"). Does asserting it
change what the model answers?

Compares two groups of positions from the same traces:
  disavowal  units tagged as asserting honesty / disclaiming the incentive
  control    untagged neighbours within +-3 units, matched on length

The controls are what make a null interpretable. Without them, "disavowal has no effect" is
indistinguishable from "nothing at this depth in the trace has an effect".

Reports against the measured sampling-noise floor (split-half of the base arm), because an
effect smaller than the floor is not an effect.

  python analysis/disavowal_test.py --run_dir runs/<name> \
      --files disavowal_below.json,disavowal_above.json --positions analysis/positions_dis.json
"""

import json
from pathlib import Path

import fire
import numpy as np

RNG = np.random.default_rng(0)
MIN_DIFF = 5


def boot_mean(groups, n=4000):
    """Bootstrap over traces (each group is one trace's values)."""
    gs = [g for g in groups if g]
    if len(gs) < 2:
        flat = [v for g in gs for v in g]
        return (float(np.mean(flat)) if flat else None), None, None
    flat = [v for g in gs for v in g]
    obs = float(np.mean(flat))
    stats = []
    for _ in range(n):
        idx = RNG.integers(0, len(gs), len(gs))
        vals = [v for i in idx for v in gs[i]]
        if vals:
            stats.append(float(np.mean(vals)))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return obs, float(lo), float(hi)


def noise_floor(recs):
    vals = []
    for r in recs:
        ys = r.get("y_base") or []
        if len(ys) < 8:
            continue
        ys = np.array(ys, dtype=float)
        h = len(ys) // 2
        halves = []
        for _ in range(50):
            p = RNG.permutation(len(ys))
            halves.append(abs(ys[p[:h]].mean() - ys[p[h:2 * h]].mean()))
        vals.append(float(np.mean(halves)))
    return float(np.mean(vals)) if vals else None


def main(run_dir: str, positions: str, files: str = "disavowal_below.json"):
    run_path = Path(run_dir)
    plan = json.loads(Path(positions).read_text())

    label = {}
    for cond, traces in plan["conditions"].items():
        for t in traces:
            for k in t.get("disavowal", []):
                label[(cond, t["trace"], k)] = "disavowal"
            for k in t.get("controls", []):
                label[(cond, t["trace"], k)] = "control"

    recs = []
    for fname in files.split(","):
        p = run_path / fname.strip()
        if not p.exists():
            print(f"(missing {p.name} — skipping)")
            continue
        data = json.loads(p.read_text())
        for cond, rs in data["conditions"].items():
            for r in rs:
                r = dict(r, cond=cond,
                         group=label.get((cond, r["trace"], r["k"]), "unlabelled"))
                recs.append(r)

    usable = [r for r in recs if r["n_diff"] >= MIN_DIFF and r["delta"] is not None]
    print(f"loaded {len(recs)} positions, {len(usable)} usable "
          f"(>={MIN_DIFF} semantically different resamples)")
    nf = noise_floor(usable)
    print(f"sampling-noise floor (split-half of base arm): "
          f"{nf:.3f}" if nf else "noise floor: n/a")

    print(f"\n{'group':<12} {'n':>4} {'mean |dp|':>10} {'ci95':>20} "
          f"{'signed':>8} {'ci95':>20}")
    out = {}
    for group in ("disavowal", "control"):
        sel = [r for r in usable if r["group"] == group]
        by_trace_abs, by_trace_sgn = {}, {}
        for r in sel:
            key = (r["cond"], r["trace"])
            by_trace_abs.setdefault(key, []).append(abs(r["delta"]))
            if r.get("signed_effect") is not None:
                by_trace_sgn.setdefault(key, []).append(r["signed_effect"])
        a, alo, ahi = boot_mean(list(by_trace_abs.values()))
        s, slo, shi = boot_mean(list(by_trace_sgn.values()))
        aci = f"[{alo:+.3f},{ahi:+.3f}]" if alo is not None else "(1 trace)"
        sci = f"[{slo:+.3f},{shi:+.3f}]" if slo is not None else "(1 trace)"
        av = f"{a:>10.3f}" if a is not None else f"{'-':>10}"
        sv = f"{s:>+8.3f}" if s is not None else f"{'-':>8}"
        print(f"{group:<12} {len(sel):>4} {av} {aci:>20} {sv} {sci:>20}")
        out[group] = {"n": len(sel), "mean_abs_delta": a, "abs_ci": [alo, ahi],
                      "signed": s, "signed_ci": [slo, shi]}

    d, c = out.get("disavowal", {}), out.get("control", {})
    if d.get("mean_abs_delta") is not None and c.get("mean_abs_delta") is not None:
        diff = d["mean_abs_delta"] - c["mean_abs_delta"]
        print(f"\ndisavowal minus control, mean |dp| = {diff:+.3f}")
        print(f"noise floor = {nf:.3f}" if nf else "")
        verdict = ("disavowal units move the answer MORE than matched controls"
                   if diff > 0.02 else
                   "disavowal units are indistinguishable from matched controls")
        print(f"verdict: {verdict}")
        if nf and d["mean_abs_delta"] is not None and d["mean_abs_delta"] < nf:
            print("NOTE: the disavowal effect is below the sampling-noise floor — "
                  "this is a null, not a small effect")

    print("\nlargest-|dp| disavowal units:")
    for r in sorted([r for r in usable if r["group"] == "disavowal"],
                    key=lambda r: -abs(r["delta"]))[:6]:
        print(f"  |dp|={abs(r['delta']):.2f} sgn={r['signed_effect']:+.2f} "
              f"{r['cond'][:5]} tr{r['trace']} k={r['k']}  {r['text'][:105]}")

    (run_path / "disavowal_analysis.json").write_text(
        json.dumps({"noise_floor": nf, "groups": out}, indent=2, default=str))
    print(f"\nsaved {run_path / 'disavowal_analysis.json'}")


if __name__ == "__main__":
    fire.Fire(main)
