"""Analyse counterfactual importance: distribution, by tag, by position, disavowal test.

A median near zero is the EXPECTED shape if the answer locks in early — most units then
have no counterfactual importance. What matters is the tail and which unit classes it
belongs to, so this reports the distribution and per-tag aggregates rather than a
single central statistic.

A-priori and exploratory positions are reported separately: exploratory ones were chosen
because the screen showed a large jump, so their effect sizes carry winner's curse.

Bootstraps over TRACES.

  python analysis/analyze_counterfactual.py --run_dir runs/<name> --positions_file ...
"""

import json
from pathlib import Path

import fire
import numpy as np

RNG = np.random.default_rng(0)
CONDS = ("below_good", "above_good")
MIN_DIFF = 5          # paper warns estimates get noisy below ~10 divergent resamples


def boot_mean(vals_by_trace, n=2000):
    """Bootstrap over traces. Resamples INDICES, not the keys themselves — numpy turns a
    list of tuple keys into an ndarray and they stop being hashable."""
    groups = list(vals_by_trace.values())
    if len(groups) < 2:
        return None, None, None
    flat = [v for g in groups for v in g]
    if not flat:
        return None, None, None
    obs = float(np.mean(flat))
    stats = []
    for _ in range(n):
        idx = RNG.integers(0, len(groups), len(groups))
        vals = [v for i in idx for v in groups[i]]
        if vals:
            stats.append(float(np.mean(vals)))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return obs, float(lo), float(hi)


def main(run_dir: str, positions_file: str, top: int = 10):
    run_path = Path(run_dir)
    data = json.loads((run_path / "counterfactual.json").read_text())
    plan = json.loads(Path(positions_file).read_text())

    ledger = {}
    for cond, traces in plan["conditions"].items():
        for t in traces:
            for k in t["apriori"]:
                ledger[(cond, t["trace"], k)] = "apriori"
            for k in t["exploratory"]:
                ledger[(cond, t["trace"], k)] = "exploratory"
    sizes = {(cond, t["trace"]): t["n_units"]
             for cond, traces in plan["conditions"].items() for t in traces}

    all_recs = []
    for cond in CONDS:
        for r in data["conditions"].get(cond, []):
            if r["n_diff"] < MIN_DIFF or r["delta"] is None:
                continue
            r = dict(r, cond=cond,
                     ledger=ledger.get((cond, r["trace"], r["k"]), "apriori"),
                     norm_pos=r["k"] / max(1, sizes.get((cond, r["trace"]), 1) - 1))
            all_recs.append(r)

    print(f"usable positions: {len(all_recs)}")
    deltas = np.array([abs(r["delta"]) for r in all_recs])
    print(f"\n|delta| distribution: mean={deltas.mean():.3f}  "
          f"median={np.median(deltas):.3f}  p75={np.percentile(deltas,75):.3f}  "
          f"p90={np.percentile(deltas,90):.3f}  max={deltas.max():.3f}")
    print(f"positions with |delta| >= 0.25: "
          f"{int((deltas >= 0.25).sum())}/{len(deltas)}")
    print(f"placebo check — |kl_same_vs_base| available for "
          f"{sum(1 for r in all_recs if r['kl_same_vs_base'] is not None)} positions")

    # ---- empirical sampling-noise floor -------------------------------------------
    # Split the BASE arm in half and compare the halves. Both halves come from the same
    # prefix, so any difference is pure sampling noise. An effect only counts if it
    # clears this. Measured, not assumed.
    floors = []
    for r in all_recs:
        ys = r.get("y_base") or []
        if len(ys) < 8:
            continue
        ys = np.array(ys, dtype=float)
        vals = []
        for _ in range(50):
            perm = RNG.permutation(len(ys))
            h = len(ys) // 2
            vals.append(abs(ys[perm[:h]].mean() - ys[perm[h:2 * h]].mean()))
        floors.append(float(np.mean(vals)))
    if floors:
        print(f"\nsampling-noise floor (split-half of the base arm, "
              f"{len(floors)} positions): mean |dp| = {np.mean(floors):.3f}")
    else:
        print("\nsampling-noise floor: per-sample outcomes not stored in this run "
              "(re-run resample.py to record y_base)")

    # Placebo comparison uses |dp|, NOT KL. With 5-16 samples p saturates at 0 or 1
    # constantly, and Bernoulli KL with an epsilon floor explodes on exactly those cases,
    # so a mean KL measures saturation frequency rather than effect size. |dp| is bounded.
    paired = [r for r in all_recs
              if r["p_base"] is not None and r["p_diff"] is not None
              and r["p_same"] is not None and r["n_same"] >= MIN_DIFF]
    if paired:
        d_diff = np.array([abs(r["p_base"] - r["p_diff"]) for r in paired])
        d_same = np.array([abs(r["p_base"] - r["p_same"]) for r in paired])
        print(f"\nplacebo check on {len(paired)} positions with >={MIN_DIFF} in BOTH arms:")
        print(f"  mean |dp| different-meaning vs base = {d_diff.mean():.3f}")
        print(f"  mean |dp| same-meaning  vs base = {d_same.mean():.3f}  (placebo)")
        print(f"  difference = {d_diff.mean() - d_same.mean():+.3f} "
              f"(positive means the meaning split is doing real work)")
    same_kl = [r["kl_same_vs_base"] for r in all_recs if r["kl_same_vs_base"] is not None]
    diff_kl = [r["kl_diff_vs_base"] for r in all_recs if r["kl_diff_vs_base"] is not None]
    if same_kl and diff_kl:
        print(f"  [KL, saturation-inflated, reported for comparability only] "
              f"diff={np.mean(diff_kl):.3f} same={np.mean(same_kl):.3f}")

    print(f"\nsigned effect by tag (a-priori positions only, bootstrapped over traces):")
    print(f"{'tag':<32} {'n':>4} {'signed':>8} {'ci95':>20}")
    tags = ("disavowal", "threshold-comparison", "parameter-selection",
            "directional-sensitivity-check", "incentive-acknowledgment", "UNTAGGED")
    for tag in tags:
        byt = {}
        for r in all_recs:
            if r["ledger"] != "apriori" or r["signed_effect"] is None:
                continue
            hit = (not r["tags"]) if tag == "UNTAGGED" else (tag in r["tags"])
            if hit:
                byt.setdefault((r["cond"], r["trace"]), []).append(r["signed_effect"])
        n = sum(len(v) for v in byt.values())
        if n == 0:
            print(f"{tag:<32} {0:>4}        -                    -")
            continue
        obs, lo, hi = boot_mean(byt)
        if obs is None:                       # only one trace contributed — no CI possible
            print(f"{tag:<32} {n:>4} {'-':>8} {'(single trace)':>20}")
            continue
        ci = f"[{lo:+.3f},{hi:+.3f}]" if lo is not None else "-"
        print(f"{tag:<32} {n:>4} {obs:>+8.3f} {ci:>20}")

    print(f"\nsigned effect by normalised position (a-priori only):")
    for lo_b, hi_b in ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)):
        byt = {}
        for r in all_recs:
            if r["ledger"] != "apriori" or r["signed_effect"] is None:
                continue
            if lo_b <= r["norm_pos"] < hi_b:
                byt.setdefault((r["cond"], r["trace"]), []).append(r["signed_effect"])
        n = sum(len(v) for v in byt.values())
        if n == 0:
            continue
        obs, lo, hi = boot_mean(byt)
        if obs is None:
            print(f"  {lo_b:.1f}-{hi_b:.1f}  n={n:>4}  (single trace, no CI)")
            continue
        ci = f"[{lo:+.3f},{hi:+.3f}]" if lo is not None else "-"
        print(f"  {lo_b:.1f}-{hi_b:.1f}  n={n:>4}  signed={obs:>+.3f}  {ci}")

    print(f"\nlargest |delta| (mixed ledger — exploratory flagged):")
    for r in sorted(all_recs, key=lambda r: -abs(r["delta"]))[:top]:
        tags = ",".join(r["tags"]) if r["tags"] else "-"
        print(f"  d={r['delta']:+.2f} sgn={r['signed_effect']:+.2f} "
              f"[{r['ledger'][:4]}] {r['cond'][:5]} tr{r['trace']} k={r['k']} "
              f"({tags}) {r['text'][:95]}")

    out = run_path / "counterfactual_analysis.json"
    out.write_text(json.dumps(
        {"n_usable": len(all_recs),
         "abs_delta": {"mean": float(deltas.mean()), "median": float(np.median(deltas)),
                       "p90": float(np.percentile(deltas, 90)), "max": float(deltas.max())},
         "mean_kl_diff": float(np.mean(diff_kl)) if diff_kl else None,
         "mean_kl_same_placebo": float(np.mean(same_kl)) if same_kl else None},
        indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    fire.Fire(main)
