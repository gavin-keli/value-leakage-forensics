"""Forensic re-analysis of the shipped value-leakage runs.

Pure function of the run dirs — no API calls, no sampling. Adds what the
shipped analysis does not have:

  1. uncertainty on MRF (bootstrap CI + label-permutation p-value)
  2. an outcome metric that does not depend on the trajectory judge:
     P(final answer > threshold) per condition
  3. start-side stratification (the motivated-stopping prediction)
  4. landing positions near the threshold
  5. the report step: last in-CoT number vs the final answer
  6. reasoning effort (chars, number of floated estimates) by condition

  python analysis/forensics.py --runs_root runs --out analysis/forensics.json
"""

import json
from pathlib import Path

import fire
import numpy as np

from value_leakage.plot import DRIFT_WINDOW, N_GRID, resample, valid

CONDITIONS = ("baseline", "below_good", "above_good")
RNG = np.random.default_rng(0)
N_BOOT = 4000
N_PERM = 4000


def per_rollout_drift(trajectories: list, threshold: float) -> np.ndarray:
    """The per-rollout quantity MRF takes the median of, kept unaggregated.

    Mirrors plot.drift exactly, including its outlier_factor=None — the shipped
    drift path does NOT apply the [thr/10, thr*10] filter that the curves apply.
    """
    kept = valid(trajectories, threshold, outlier_factor=None)
    w = max(1, int(round(N_GRID * DRIFT_WINDOW)))
    out = []
    for t in kept:
        g = resample(t)
        out.append((g[-w:].mean() - g[:w].mean()) / threshold)
    return np.asarray(out, dtype=float)


def boot_ci(fn, *samples, n: int = N_BOOT, alpha: float = 0.05) -> tuple:
    """Percentile bootstrap over rollouts, resampling each sample independently."""
    stats = []
    for _ in range(n):
        drawn = [s[RNG.integers(0, len(s), len(s))] if len(s) else s for s in samples]
        stats.append(fn(*drawn))
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def perm_p(a: np.ndarray, b: np.ndarray, n: int = N_PERM) -> float:
    """Two-sided label-permutation p-value for median(a) - median(b)."""
    obs = abs(np.median(a) - np.median(b))
    pool = np.concatenate([a, b])
    na = len(a)
    hits = 0
    for _ in range(n):
        RNG.shuffle(pool)
        if abs(np.median(pool[:na]) - np.median(pool[na:])) >= obs:
            hits += 1
    return (hits + 1) / (n + 1)


def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    """Wilson score interval — behaves at the extremes where normal-approx does not."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return float(centre - half), float(centre + half)


def analyse_run(run_dir: Path) -> dict:
    threshold = json.loads((run_dir / "threshold.json").read_text())["threshold"]
    trajectories = json.loads((run_dir / "trajectories.json").read_text())
    estimates = json.loads((run_dir / "estimates.json").read_text())
    config = json.loads((run_dir / "config.json").read_text())

    out = {"model": config.get("model", run_dir.name), "threshold": threshold,
           "conditions": {}}

    # ---- 1. MRF with uncertainty -------------------------------------------
    drifts = {c: per_rollout_drift(trajectories.get(c, []), threshold)
              for c in CONDITIONS}
    a, b = drifts["above_good"], drifts["below_good"]
    mrf = float(np.median(a) - np.median(b)) if len(a) and len(b) else None
    out["mrf"] = {
        "value": mrf,
        "n_above": int(len(a)), "n_below": int(len(b)),
        "ci95": boot_ci(lambda x, y: np.median(x) - np.median(y), a, b)
                if len(a) and len(b) else None,
        "perm_p": perm_p(a, b) if len(a) and len(b) else None,
    }

    # ---- 2. judge-independent outcome: P(final answer > threshold) ----------
    for c in CONDITIONS:
        finals = [e for e in (estimates.get(c) or []) if e is not None]
        k = sum(1 for e in finals if e > threshold)
        n = len(finals)
        rec = {
            "n_parsed": n,
            "unknown_rate": 1 - n / len(estimates.get(c) or [1]),
            "p_above_threshold": k / n if n else None,
            "p_above_ci95": wilson(k, n),
            "median_final": float(np.median(finals)) if finals else None,
        }
        out["conditions"][c] = rec

    fa = np.array([1.0 if e > threshold else 0.0
                   for e in (estimates.get("above_good") or []) if e is not None])
    fb = np.array([1.0 if e > threshold else 0.0
                   for e in (estimates.get("below_good") or []) if e is not None])
    out["p_gap"] = {
        "definition": "P(final > threshold | above_good) - P(final > threshold | below_good)",
        "value": float(fa.mean() - fb.mean()) if len(fa) and len(fb) else None,
        "ci95": boot_ci(lambda x, y: x.mean() - y.mean(), fa, fb)
                if len(fa) and len(fb) else None,
    }

    # ---- 3. start-side stratification (motivated-stopping prediction) -------
    strata = {}
    for c in ("below_good", "above_good"):
        kept = valid(trajectories.get(c, []), threshold, outlier_factor=None)
        d = per_rollout_drift(trajectories.get(c, []), threshold)
        starts = np.array([t[0] for t in kept], dtype=float)
        for side, mask in (("start_above", starts > threshold),
                           ("start_below", starts <= threshold)):
            sel = d[mask]
            favoured = (c == "above_good" and side == "start_above") or \
                       (c == "below_good" and side == "start_below")
            strata[f"{c}|{side}"] = {
                "n": int(len(sel)),
                "median_drift": float(np.median(sel)) if len(sel) else None,
                "already_favoured_at_start": favoured,
                "median_n_estimates": float(np.median(
                    [len(t) for t, m in zip(kept, mask) if m])) if len(sel) else None,
            }
    out["start_side"] = strata

    # ---- 4. landing positions ----------------------------------------------
    for c in CONDITIONS:
        finals = np.array([e for e in (estimates.get(c) or []) if e is not None],
                          dtype=float)
        if not len(finals):
            continue
        rel = (finals - threshold) / threshold
        out["conditions"][c].update({
            "frac_within_5pct_of_threshold": float(np.mean(np.abs(rel) <= 0.05)),
            "frac_exactly_at_threshold": float(np.mean(finals == threshold)),
            "median_rel_deviation": float(np.median(rel)),
        })

    # ---- 5. the report step: last in-CoT number vs final answer -------------
    for c in CONDITIONS:
        traj = trajectories.get(c) or []
        est = estimates.get(c) or []
        deltas = []
        for t, e in zip(traj, est):
            if isinstance(t, list) and len(t) and e is not None and t[-1]:
                deltas.append((e - t[-1]) / threshold)
        out["conditions"][c]["report_shift"] = {
            "definition": "(final answer - last in-CoT estimate)/threshold",
            "n": len(deltas),
            "median": float(np.median(deltas)) if deltas else None,
            "frac_nonzero": float(np.mean([d != 0 for d in deltas])) if deltas else None,
        }

    # ---- 6. reasoning effort ------------------------------------------------
    for c in CONDITIONS:
        path = run_dir / f"{c}.json"
        if not path.exists():
            continue
        rows = json.loads(path.read_text())["rows"]
        lens = [len(r.get("reasoning") or "") for r in rows]
        traj = trajectories.get(c) or []
        n_est = [len(t) for t in traj if isinstance(t, list)]
        out["conditions"][c].update({
            "median_cot_chars": float(np.median(lens)) if lens else None,
            "median_n_estimates": float(np.median(n_est)) if n_est else None,
        })

    return out


def main(runs_root: str = "runs", out: str = "analysis/forensics.json"):
    root = Path(runs_root)
    results = []
    for d in sorted(root.iterdir()):
        if not (d / "trajectories.json").exists():
            continue
        try:
            results.append(analyse_run(d))
        except Exception as exc:  # a malformed run should not kill the sweep
            print(f"{d.name}: SKIPPED ({type(exc).__name__}: {exc})")

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))

    def fmt(v, spec, width):
        """Shipped runs have no judged estimates for the incentive conditions, so several
        of these are legitimately absent — print a dash rather than crashing."""
        return f"{v:{spec}}" if v is not None else "-".rjust(width)

    print(f"\n{'model':<26} {'MRF':>8} {'ci95':>18} {'p':>7} "
          f"{'P(>thr) bel':>12} {'P(>thr) abv':>12} {'gap':>8}")
    for r in results:
        m, g = r["mrf"], r["p_gap"]
        ci = m["ci95"]
        below = r["conditions"].get("below_good", {})
        above = r["conditions"].get("above_good", {})
        ci_s = f"[{ci[0]:>+7.4f},{ci[1]:>+7.4f}]" if ci else "-".rjust(18)
        print(f"{r['model']:<26} {fmt(m['value'], '+8.4f', 8)} {ci_s} "
              f"{fmt(m['perm_p'], '7.3f', 7)} "
              f"{fmt(below.get('p_above_threshold'), '12.2f', 12)} "
              f"{fmt(above.get('p_above_threshold'), '12.2f', 12)} "
              f"{fmt(g['value'], '+8.3f', 8)}")
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    fire.Fire(main)
