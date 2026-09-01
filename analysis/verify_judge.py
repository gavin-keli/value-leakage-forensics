"""Did every judge call actually complete, or did some fail on the API?

judge.py conflates two very different nulls:
  - the rollout was never judged (API exception, or empty source text) -> recoverable
  - the judge answered but the answer did not parse (UNKNOWN / NONE / nonconforming)
    -> a real property of the trace, not a failure

It prints 'idx N: ExceptionType' for the first kind, so the log is the only place that
distinguishes them. This checks both the artifacts and the log.
"""

import json
import re
from pathlib import Path

import fire

CONDITIONS = ("baseline", "below_good", "above_good")


def main(run_dir: str, log: str | None = None):
    run_path = Path(run_dir)
    print(f"=== {run_dir} ===")

    for kind in ("estimates", "trajectories"):
        path = run_path / f"{kind}.json"
        if not path.exists():
            print(f"{kind}: MISSING")
            continue
        data = json.loads(path.read_text())
        print(f"\n{kind}:")
        for c in CONDITIONS:
            vals = data.get(c)
            if vals is None:
                print(f"  {c:<11} ABSENT from {kind}.json")
                continue
            rows = json.loads((run_path / f"{c}.json").read_text())["rows"]
            empty_src = sum(1 for r in rows
                            if not (r.get("content" if kind == "estimates"
                                          else "reasoning") or "").strip())
            nulls = [i for i, v in enumerate(vals) if v is None]
            print(f"  {c:<11} n={len(vals)} judged_ok={len(vals)-len(nulls)} "
                  f"null={len(nulls)} empty_source={empty_src}")
            if nulls:
                print(f"      null indices: {nulls[:20]}"
                      f"{' ...' if len(nulls) > 20 else ''}")

    if log and Path(log).exists():
        text = Path(log).read_text(errors="ignore")
        errs = re.findall(r"idx (\d+): (\w+Error|\w+Exception)", text)
        print(f"\nAPI exceptions reported in {log}: {len(errs)}")
        for idx, kind in errs[:20]:
            print(f"  idx {idx}: {kind}")
        if not errs:
            print("  none — every null is a parse outcome, not a failed call")


if __name__ == "__main__":
    fire.Fire(main)
