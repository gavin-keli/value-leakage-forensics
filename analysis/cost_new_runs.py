"""Cost the judging for the three new runs, from measured sizes not guesses.

Two judge passes per run:
  estimates     reads the VISIBLE ANSWER (short)            -> 1 call per rollout
  trajectories  reads the FULL REASONING TRACE (long)       -> 1 call per rollout

Baselines exist on disk, so their sizes are measured. The two incentive conditions are not
sampled yet; their length is projected from the inflation factor observed in the completed
runs (incentive traces run ~1.2-1.3x baseline).
"""
import json
import sys
from pathlib import Path

import numpy as np

PROMPT_OVERHEAD = 1900        # judge template minus the {llm_text} slot, chars
CHARS_PER_TOKEN = 3.6
IN_PER_M = 2.00               # Sonnet 5
OUT_PER_M = 10.00
OUT_TOKENS_EST = 120          # comma-separated integer list / tagged number
INFLATION = 1.25              # incentive-condition trace length vs baseline

total_in = total_out = total_calls = 0.0
print(f"{'run':<26} {'pass':<13} {'calls':>6} {'in-tok':>11} {'$in':>8} {'$out':>7} {'$tot':>8}")

for run_dir in sys.argv[1:]:
    p = Path(run_dir, "baseline.json")
    if not p.exists():
        print(f"{Path(run_dir).name:<26} MISSING baseline.json")
        continue
    rows = json.loads(p.read_text())["rows"]
    n = len(rows)
    ans = float(np.mean([len(r.get("content") or "") for r in rows]))
    rea = float(np.mean([len(r.get("reasoning") or "") for r in rows]))

    for label, chars_per_call, calls in (
            ("estimates", ans + PROMPT_OVERHEAD, n * 3),
            ("trajectories", rea * ((1 + 2 * INFLATION) / 3) + PROMPT_OVERHEAD, n * 3)):
        in_tok = calls * chars_per_call / CHARS_PER_TOKEN
        out_tok = calls * OUT_TOKENS_EST
        d_in = in_tok / 1e6 * IN_PER_M
        d_out = out_tok / 1e6 * OUT_PER_M
        print(f"{Path(run_dir).name:<26} {label:<13} {calls:>6} {in_tok:>11,.0f} "
              f"{d_in:>8.2f} {d_out:>7.2f} {d_in+d_out:>8.2f}")
        total_in += in_tok
        total_out += out_tok
        total_calls += calls

print("-" * 82)
d_in, d_out = total_in / 1e6 * IN_PER_M, total_out / 1e6 * OUT_PER_M
print(f"{'TOTAL':<26} {'':<13} {total_calls:>6.0f} {total_in:>11,.0f} "
      f"{d_in:>8.2f} {d_out:>7.2f} {d_in+d_out:>8.2f}")
print(f"\n(estimates pass covers all 3 conditions; trajectory input projected with a "
      f"{INFLATION}x inflation on the two incentive conditions)")
