"""Merge Track A's two condition files, produced on separate GPUs.

copenhagen ran below_good -> counterfactual.json
oslo       ran above_good -> counterfactual_above.json

The conditions are analytically independent (each is scored against its own base arm), so
splitting them across hosts changes nothing except wall clock. Verifies the shared
settings match before merging, because a silent mismatch in R or threshold would make the
signed comparison meaningless.
"""

import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
primary = json.loads((run_dir / "counterfactual.json").read_text())
extra_path = run_dir / "counterfactual_above.json"
if not extra_path.exists():
    sys.exit(f"missing {extra_path}")
extra = json.loads(extra_path.read_text())

for field in ("threshold", "model", "family", "R", "cos_threshold"):
    if primary.get(field) != extra.get(field):
        sys.exit(f"MISMATCH on {field}: {primary.get(field)!r} vs {extra.get(field)!r}")

before = {k: len(v) for k, v in primary["conditions"].items()}
for cond, recs in extra["conditions"].items():
    if cond in primary["conditions"]:
        print(f"note: {cond} already present with {len(primary['conditions'][cond])} "
              f"records; overwriting with {len(recs)} from the second host")
    primary["conditions"][cond] = recs

out = run_dir / "counterfactual_merged.json"
out.write_text(json.dumps(primary, indent=2, ensure_ascii=False, default=str))
print(f"before: {before}")
print(f"after:  { {k: len(v) for k, v in primary['conditions'].items()} }")
print(f"settings verified identical (R={primary['R']}, threshold={primary['threshold']:,})")
print(f"saved {out}")
