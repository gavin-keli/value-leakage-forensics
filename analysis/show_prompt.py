"""Print the rendered system prompt actually sent to each model, and trace-length stats."""
import json, sys
from pathlib import Path
import numpy as np

for run_dir in sys.argv[1:]:
    d = json.loads(Path(run_dir, "above_good.json").read_text())
    print(f"\n=== {Path(run_dir).name} ===")
    print("--- rendered prompt, first 420 chars ---")
    print(repr(d["rendered_prompt"][:420]))
    toks = [r["usage"]["completion_tokens"] for r in d["rows"]]
    print(f"completion tokens: median {int(np.median(toks)):,}  "
          f"min {min(toks):,}  max {max(toks):,}  cap was {d['max_tokens']:,}")
    print(f"hit the cap: {sum(1 for r in d['rows'] if r['finish_reason']=='length')}/{len(toks)}")
