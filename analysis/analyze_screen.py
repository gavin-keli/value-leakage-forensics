"""Analyse the forced-answer screen and nominate positions for step 5.

Aggregates on NORMALISED position (k / n_units of that trace), never raw index: traces
here run 21-909 units, so pooling by index mixes "middle of a short trace" with "early in
a long one".

Bootstraps over TRACES, not over samples — the R samples at one position are correlated,
and treating them as independent would inflate significance.

  python analysis/analyze_screen.py --run_dir runs/<name>
"""

import json
from pathlib import Path

import fire
import numpy as np

RNG = np.random.default_rng(0)
CONDS = ("below_good", "above_good")


def load(run_path: Path):
    screen = json.loads((run_path / "screen.json").read_text())
    sizes = {}
    for cond in CONDS:
        segs = json.loads((run_path / f"segments_{cond}.json").read_text())
        sizes[cond] = {s["i"]: s["n_units"] for s in segs}
    return screen, sizes


def curve(records, sizes, bins):
    """mean P per normalised-position bin, and the per-trace values for bootstrapping."""
    per_trace = {}
    for r in records:
        if r["trace"] is None or r["p_above"] is None:
            continue
        n = sizes.get(r["trace"], 0)
        if n <= 1:
            continue
        x = r["k"] / (n - 1)
        b = min(int(x * bins), bins - 1)
        per_trace.setdefault(r["trace"], {}).setdefault(b, []).append(r["p_above"])
    means = {}
    for t, byb in per_trace.items():
        for b, vals in byb.items():
            means.setdefault(b, []).append(float(np.mean(vals)))
    return means


def boot_gap(a_means, b_means, b, n=2000):
    a, bb = a_means.get(b, []), b_means.get(b, [])
    if len(a) < 2 or len(bb) < 2:
        return None, None, None
    obs = float(np.mean(a) - np.mean(bb))
    stats = [float(np.mean(RNG.choice(a, len(a))) - np.mean(RNG.choice(bb, len(bb))))
             for _ in range(n)]
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return obs, float(lo), float(hi)


def main(run_dir: str, bins: int = 10, top: int = 12):
    run_path = Path(run_dir)
    screen, sizes = load(run_path)
    recs = {c: screen["conditions"][c] for c in CONDS}

    anchors = {c: next(r for r in recs[c] if r["k"] == -1) for c in CONDS}
    print("k = -1 (no reasoning at all):")
    for c in CONDS:
        print(f"  {c:<11} P(>thr)={anchors[c]['p_above']:.3f}  "
              f"median={anchors[c]['median']:,.0f}")
    print(f"  pre-reasoning gap = "
          f"{anchors['above_good']['p_above'] - anchors['below_good']['p_above']:+.3f}")

    curves = {c: curve(recs[c], sizes[c], bins) for c in CONDS}
    print(f"\nP(>thr) by normalised position ({bins} bins), bootstrapped over traces:")
    print(f"{'bin':>5} {'below':>8} {'above':>8} {'gap':>8} {'ci95':>20} {'n_tr':>5}")
    divergence = None
    for b in range(bins):
        lo_b = curves["below_good"].get(b, [])
        hi_b = curves["above_good"].get(b, [])
        if not lo_b or not hi_b:
            continue
        obs, lo, hi = boot_gap(curves["above_good"], curves["below_good"], b)
        ci = f"[{lo:+.3f},{hi:+.3f}]" if lo is not None else "-"
        star = ""
        if lo is not None and lo > 0:
            star = " *"
            if divergence is None:
                divergence = b
        print(f"{b/bins:>5.1f} {np.mean(lo_b):>8.3f} {np.mean(hi_b):>8.3f} "
              f"{(obs if obs is not None else float('nan')):>+8.3f} {ci:>20} "
              f"{min(len(lo_b), len(hi_b)):>5}{star}")
    if divergence is not None:
        print(f"\nDIVERGENCE: conditions separate (CI excludes 0) from normalised "
              f"position {divergence/bins:.1f}")
    else:
        print("\nDIVERGENCE: no bin where the gap's CI excludes 0")

    print(f"\nLargest single-step jumps (screen-driven nominations, EXPLORATORY):")
    for cond in CONDS:
        byt = {}
        for r in recs[cond]:
            if r["trace"] is not None and r["p_above"] is not None:
                byt.setdefault(r["trace"], []).append(r)
        jumps = []
        for t, rs in byt.items():
            rs.sort(key=lambda r: r["k"])
            for prev, cur in zip(rs, rs[1:]):
                jumps.append((cur["p_above"] - prev["p_above"], t, cur))
        jumps.sort(key=lambda j: -abs(j[0]))
        print(f"\n  {cond}:")
        for d, t, r in jumps[:top // 2]:
            tags = ",".join(r["tags"]) if r["tags"] else "-"
            print(f"    dP={d:+.2f} trace {t:>3} k={r['k']:>4} [{tags}] {r['text'][:110]}")

    out = {"anchors": anchors, "divergence_bin": divergence, "bins": bins}
    (run_path / "screen_analysis.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nsaved {run_path / 'screen_analysis.json'}")


if __name__ == "__main__":
    fire.Fire(main)
