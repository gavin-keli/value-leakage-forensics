"""Judge a run dir with the repo's own judges, and derive the threshold.

Two modes:
  --kind estimates    final visible answer per rollout, for every condition present.
                      Run once after baseline (to set the threshold) and again after the
                      incentive conditions exist — the shipped pipeline never does the
                      second pass, which is why estimates.json in all 10 shipped runs
                      contains only `baseline`.
  --kind trajectories in-CoT estimate sequence per rollout.

Threshold is written only when baseline estimates are present and threshold.json is absent,
so re-running after the incentive conditions cannot silently move the threshold underneath
data that was already sampled against it.
"""

import asyncio
import json
from pathlib import Path

import fire
from dotenv import load_dotenv

from value_leakage.judge import _judge
from value_leakage.run import compute_threshold


def main(run_dir: str, kind: str = "estimates",
         judge_model: str = "claude-sonnet-5", max_concurrent: int = 40):
    load_dotenv(Path.cwd() / ".env")
    run_path = Path(run_dir)

    out = asyncio.run(_judge(kind, run_path, judge_model, max_concurrent))
    (run_path / f"{kind}.json").write_text(json.dumps(out, indent=2))
    print(f"saved {run_path / f'{kind}.json'} — conditions: {list(out)}")

    if kind == "estimates" and "baseline" in out:
        tpath = run_path / "threshold.json"
        if tpath.exists():
            print(f"threshold.json exists, leaving it alone: {tpath.read_text().strip()}")
        else:
            compute_threshold(out["baseline"], run_path)

    for cond, vals in out.items():
        n = sum(1 for v in vals if v is not None)
        print(f"  {cond}: {n}/{len(vals)} parsed")


if __name__ == "__main__":
    fire.Fire(main)
