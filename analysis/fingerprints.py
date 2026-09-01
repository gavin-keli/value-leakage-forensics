"""Step 8 — does the mechanism found by resampling leave a signature in the shipped runs?

Resampling can only be run on the two local models. But the two mechanisms it identified
have different observable signatures in trajectory data that ALREADY exists for all 12
runs, so the generality of each can be tested correlationally:

  EARLY COMMITMENT (gpt-oss shape)   the conditions are already far apart at the first
                                     floated estimate and the gap does not grow:
                                     gap_at_start large, curve_drift <= 0
  SUSTAINED DRIFT (Qwen3.5 shape)    the gap opens during the trace:
                                     curve_drift > 0

This is correlational and labelled as such: it tests whether the shipped models look like
one of the two mechanisms we established causally, not that they share it.

  python analysis/fingerprints.py
"""

import glob
import json
import os
from pathlib import Path

import numpy as np


def classify(gap_start, gap_end, drift):
    if gap_start is None or drift is None:
        return "insufficient"
    if abs(gap_start) >= 0.15 and drift <= 0.02:
        return "early-commitment"
    if drift > 0.05:
        return "sustained-drift"
    if abs(gap_start) < 0.05 and abs(drift) <= 0.05:
        return "no-signature"
    return "mixed"


rows = []
for d in sorted(glob.glob("runs/*")):
    fp = Path(d, "factor.json")
    if not fp.exists():
        continue
    f = json.loads(fp.read_text())
    name = os.path.basename(d)
    rows.append({
        "run": name,
        "mrf": f.get("motivated_reasoning_factor"),
        "gap_start": f.get("gap_at_start"),
        "gap_end": f.get("gap_at_end"),
        "drift": f.get("curve_drift_end_minus_start"),
        "n_kept_above": (f.get("n_kept") or {}).get("above_good"),
    })

print(f"{'run':<40} {'MRF':>8} {'gap@start':>10} {'gap@end':>9} {'curve drift':>12}  signature")
for r in rows:
    sig = classify(r["gap_start"], r["gap_end"], r["drift"])
    r["signature"] = sig
    def f(x):
        return f"{x:>+.3f}" if isinstance(x, (int, float)) else "     -"
    print(f"{r['run']:<40} {f(r['mrf']):>8} {f(r['gap_start']):>10} "
          f"{f(r['gap_end']):>9} {f(r['drift']):>12}  {sig}")

print("\ncounts:")
for sig in ("early-commitment", "sustained-drift", "mixed", "no-signature", "insufficient"):
    n = sum(1 for r in rows if r["signature"] == sig)
    if n:
        print(f"  {sig:<20} {n}")

local = [r for r in rows if r["run"].startswith("local-")]
print("\nour two models (the ones with causal evidence):")
for r in local:
    print(f"  {r['run']:<38} {r['signature']}")

Path("analysis/fingerprints.json").write_text(json.dumps(rows, indent=2))
print("\nsaved analysis/fingerprints.json")
print("\nNOTE: correlational. Resampling established the mechanism only for the two local "
      "models; this asks whether shipped models share the signature, not the mechanism.")
