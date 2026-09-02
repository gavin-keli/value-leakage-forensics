"""Does counterfactual importance degrade with REMAINING reasoning, not total length?

A8 measured the placebo separation once per run and compared runs of different trace
length. But the mechanism it proposes is about how much reasoning remains AFTER the
intervention — total length is only a proxy. Remaining length is recoverable for every
position already resampled (segment offsets + trace length), so the curve can be measured
directly, pooled over ~1,300 positions, with no new generation.

  placebo separation = mean |dp|(different-meaning vs base) - mean |dp|(same-meaning vs base)

Positive means the meaning-split discriminates; <= 0 means it does not.

The pooled curve is confounded — positions with little remaining come mostly from short
traces — so the load-bearing output is the WITHIN-CONFIG breakdown: inside one long-trace
run, do late positions behave like short traces?

  python analysis/horizon_curve.py --specs "label=run_dir:cf_file, ..."
"""

import json
from pathlib import Path

import fire
import numpy as np

RNG = np.random.default_rng(0)
CONDS = ("below_good", "above_good")
MIN_ARM = 5
BINS = [(0, 500), (500, 2000), (2000, 8000), (8000, 20000), (20000, 10**9)]
BIN_LABELS = ["<0.5k", "0.5-2k", "2-8k", "8-20k", ">20k"]


def load_positions(run_dir: str, cf_file: str, label: str):
    """Every usable position with its remaining-character count."""
    p = Path(run_dir)
    cf = json.loads((p / cf_file).read_text())
    out = []
    for cond in CONDS:
        recs = cf.get("conditions", {}).get(cond)
        if not recs:
            continue
        segs_path, rows_path = p / f"segments_{cond}.json", p / f"{cond}.json"
        if not (segs_path.exists() and rows_path.exists()):
            continue
        segs = {s["i"]: s for s in json.loads(segs_path.read_text())}
        rows = {r["i"]: r for r in json.loads(rows_path.read_text())["rows"]}
        for r in recs:
            s = segs.get(r["trace"])
            row = rows.get(r["trace"])
            if s is None or row is None or r["k"] >= len(s["units"]):
                continue
            # both arms must be populated for the placebo comparison to mean anything
            if (r.get("n_diff", 0) < MIN_ARM or r.get("n_same", 0) < MIN_ARM
                    or r.get("p_base") is None or r.get("p_diff") is None
                    or r.get("p_same") is None):
                continue
            total = len(row.get("reasoning") or "")
            remaining = max(0, total - s["units"][r["k"]]["end"])
            out.append({
                "label": label, "cond": cond, "trace": r["trace"], "k": r["k"],
                "remaining": remaining, "total": total,
                "d_diff": abs(r["p_base"] - r["p_diff"]),
                "d_same": abs(r["p_base"] - r["p_same"]),
            })
    return out


def boot_sep(items, n=3000):
    """Placebo separation with a CI bootstrapped over traces, not positions."""
    if not items:
        return None, None, None, 0
    groups = {}
    for it in items:
        groups.setdefault((it["label"], it["cond"], it["trace"]), []).append(it)
    keys = list(groups)
    obs = float(np.mean([it["d_diff"] for it in items])
                - np.mean([it["d_same"] for it in items]))
    if len(keys) < 2:
        return obs, None, None, len(items)
    stats = []
    for _ in range(n):
        idx = RNG.integers(0, len(keys), len(keys))
        pick = [it for i in idx for it in groups[keys[i]]]
        if pick:
            stats.append(float(np.mean([x["d_diff"] for x in pick])
                               - np.mean([x["d_same"] for x in pick])))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return obs, float(lo), float(hi), len(items)


def rankdata(x):
    """Average-tied ranks, so no scipy dependency is introduced."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), float)
    ranks[order] = np.arange(1, len(x) + 1)
    x = np.asarray(x, float)
    for v in np.unique(x):
        m = x == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return ranks


def spearman(items):
    """Rank correlation of remaining length vs per-position separation.

    Uses every position rather than binning them, which matters at n=29.
    p is a trace-permutation test, so positions from one trace stay together.
    """
    if len(items) < 8:
        return None, None, len(items)
    rem = np.array([it["remaining"] for it in items], float)
    sep = np.array([it["d_diff"] - it["d_same"] for it in items], float)
    rr, rs = rankdata(rem), rankdata(sep)
    obs = float(np.corrcoef(rr, rs)[0, 1])
    traces = {}
    for j, it in enumerate(items):
        traces.setdefault((it["cond"], it["trace"]), []).append(j)
    blocks = list(traces.values())
    if len(blocks) < 4:
        return obs, None, len(items)
    null = []
    for _ in range(2000):
        perm = np.empty(len(items), float)
        shuffled = RNG.permutation(len(blocks))
        # move whole traces, preserving each trace's internal ordering
        src = [j for b in shuffled for j in blocks[b]]
        perm[np.array([j for b in blocks for j in b])] = rs[np.array(src)]
        null.append(abs(float(np.corrcoef(rr, perm)[0, 1])))
    p = float((np.sum(np.array(null) >= abs(obs)) + 1) / (len(null) + 1))
    return obs, p, len(items)


def median_split(items):
    """Separation below vs above the median remaining length, CI on the gap."""
    if len(items) < 12:
        return None
    med = float(np.median([it["remaining"] for it in items]))
    lo = [it for it in items if it["remaining"] <= med]
    hi = [it for it in items if it["remaining"] > med]
    if len(lo) < 4 or len(hi) < 4:
        return None
    s_lo, _, _, _ = boot_sep(lo, n=1)
    s_hi, _, _, _ = boot_sep(hi, n=1)
    groups = {}
    for it in items:
        groups.setdefault((it["cond"], it["trace"]), []).append(it)
    keys = list(groups)
    diffs = []
    for _ in range(3000):
        idx = RNG.integers(0, len(keys), len(keys))
        pick = [it for i in idx for it in groups[keys[i]]]
        a = [it for it in pick if it["remaining"] <= med]
        b = [it for it in pick if it["remaining"] > med]
        if len(a) >= 3 and len(b) >= 3:
            diffs.append((np.mean([x["d_diff"] for x in a]) - np.mean([x["d_same"] for x in a]))
                         - (np.mean([x["d_diff"] for x in b]) - np.mean([x["d_same"] for x in b])))
    if not diffs:
        return None
    ci = np.percentile(diffs, [2.5, 97.5])
    return med, len(lo), s_lo, len(hi), s_hi, float(np.mean(diffs)), float(ci[0]), float(ci[1])


def table(items, title):
    print(f"\n{title}")
    print(f"{'remaining':>10} {'n':>6} {'diff':>7} {'same':>7} {'separation':>11} {'ci95':>18}")
    for (lo_b, hi_b), lab in zip(BINS, BIN_LABELS):
        sel = [it for it in items if lo_b <= it["remaining"] < hi_b]
        if not sel:
            continue
        obs, lo, hi, n = boot_sep(sel)
        ci = f"[{lo:+.3f},{hi:+.3f}]" if lo is not None else "(1 trace)"
        print(f"{lab:>10} {n:>6} {np.mean([s['d_diff'] for s in sel]):>7.3f} "
              f"{np.mean([s['d_same'] for s in sel]):>7.3f} {obs:>+11.3f} {ci:>18}")


def main(specs: str, out: str = "analysis/horizon_curve.json"):
    all_items = []
    for spec in specs.split(","):
        label, rest = spec.split("=", 1)
        run_dir, cf_file = rest.split(":", 1)
        items = load_positions(run_dir.strip(), cf_file.strip(), label.strip())
        all_items.extend(items)
        print(f"{label.strip():<22} {len(items):>5} usable positions "
              f"(median remaining {np.median([i['remaining'] for i in items]):,.0f} chars)"
              if items else f"{label.strip():<22} none")

    table(all_items, "=== POOLED across configs (confounded: short-remaining comes "
                     "mostly from short traces) ===")

    labels = list(dict.fromkeys(i["label"] for i in all_items))
    for label in labels:
        sub = [i for i in all_items if i["label"] == label]
        if len(sub) >= 25:
            table(sub, f"=== WITHIN {label} (breaks the confound) ===")

    print("\n=== rank correlation: remaining length vs per-position separation ===")
    print("negative rho = separation shrinks as more reasoning remains (horizon account)")
    print(f"{'config':<22} {'n':>5} {'rho':>8} {'p(perm)':>9}")
    for label in labels + ["ALL POOLED"]:
        sub = all_items if label == "ALL POOLED" else [i for i in all_items if i["label"] == label]
        rho, p, n = spearman(sub)
        if rho is None:
            print(f"{label:<22} {n:>5}  (too few)")
        else:
            print(f"{label:<22} {n:>5} {rho:>+8.3f} "
                  f"{('%.3f' % p) if p is not None else '   n/a':>9}")

    print("\n=== within-config median split on remaining length ===")
    print(f"{'config':<22} {'median':>8} {'n_lo':>5} {'sep_lo':>8} {'n_hi':>5} {'sep_hi':>8} "
          f"{'lo-hi':>8} {'ci95':>18}")
    for label in labels:
        sub = [i for i in all_items if i["label"] == label]
        r = median_split(sub)
        if r is None:
            print(f"{label:<22}   (too few)")
            continue
        med, n_lo, s_lo, n_hi, s_hi, gap, lo, hi = r
        print(f"{label:<22} {med:>8,.0f} {n_lo:>5} {s_lo:>+8.3f} {n_hi:>5} {s_hi:>+8.3f} "
              f"{gap:>+8.3f} {f'[{lo:+.3f},{hi:+.3f}]':>18}")

    # persist, so the report cites a file rather than a console transcript
    def bins_for(items):
        rows = []
        for (lo_b, hi_b), lab in zip(BINS, BIN_LABELS):
            sel = [it for it in items if lo_b <= it["remaining"] < hi_b]
            if not sel:
                continue
            obs, lo, hi, n = boot_sep(sel)
            rows.append({"bin": lab, "n": n,
                         "mean_abs_dp_different": float(np.mean([s["d_diff"] for s in sel])),
                         "mean_abs_dp_same": float(np.mean([s["d_same"] for s in sel])),
                         "separation": obs, "ci95": [lo, hi]})
        return rows

    payload = {
        "metric": "placebo separation = mean|dp|(different-meaning) - mean|dp|(same-meaning)",
        "variable": "characters of reasoning remaining after the resampled unit",
        "n_positions": len(all_items),
        "pooled_bins": bins_for(all_items),
        "by_config": {},
    }
    for label in labels:
        sub = [i for i in all_items if i["label"] == label]
        rho, p, n = spearman(sub)
        ms = median_split(sub)
        payload["by_config"][label] = {
            "n": len(sub),
            "median_remaining": float(np.median([i["remaining"] for i in sub])),
            "bins": bins_for(sub),
            "spearman_rho": rho, "spearman_p": p,
            "median_split": None if ms is None else {
                "median": ms[0], "n_low": ms[1], "sep_low_remaining": ms[2],
                "n_high": ms[3], "sep_high_remaining": ms[4],
                "gap_low_minus_high": ms[5], "ci95": [ms[6], ms[7]]},
        }
    rho, p, _ = spearman(all_items)
    payload["pooled_spearman_rho"], payload["pooled_spearman_p"] = rho, p
    Path(out).write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    fire.Fire(main)
