"""Recompute every number in the write-up's §1 model table straight from the artifacts."""
import json, sys
from pathlib import Path
import numpy as np

CONDS = ("baseline", "below_good", "above_good")

for run_dir in sys.argv[1:]:
    p = Path(run_dir)
    cfg = json.loads((p / "config.json").read_text())
    thr = json.loads((p / "threshold.json").read_text())
    print(f"\n=== {cfg.get('model')} ({p.name}) ===")
    print(f"  model_id        : {cfg.get('model_id')}")
    print(f"  threshold       : {thr['threshold']:,}  "
          f"(n_valid {thr.get('n_valid')}/{thr.get('n_baseline')})")
    print(f"  max_tokens      : {cfg.get('target_max_tokens')}")

    med_chars, n_trunc, n_unclosed, counts = {}, 0, 0, {}
    for c in CONDS:
        data = json.loads((p / f"{c}.json").read_text())
        rows = data["rows"]
        counts[c] = len(rows)
        lens = [len(r.get("reasoning") or "") for r in rows]
        med_chars[c] = float(np.median(lens))
        n_trunc += sum(1 for r in rows if r.get("finish_reason") == "length")
        n_unclosed += sum(1 for r in rows if not (r.get("content") or "").strip())
        if c == "baseline":
            print(f"  max_model_len   : (run with) see launch cmd; prompt+gen cap "
                  f"{data.get('max_tokens')}")

    print(f"  rollouts        : {len(CONDS)} x {counts['baseline']}  "
          f"(truncated {n_trunc}, empty-answer {n_unclosed})")
    print(f"  median reasoning: baseline {med_chars['baseline']:,.0f} | "
          f"below {med_chars['below_good']:,.0f} | above {med_chars['above_good']:,.0f} chars")
    print(f"    -> range for the table: "
          f"{min(med_chars.values()):,.0f}-{max(med_chars.values()):,.0f}")
