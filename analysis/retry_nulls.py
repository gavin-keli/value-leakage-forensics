"""Re-judge only the rollouts whose trajectory came back null, and report recovery.

The judge runs at the API default temperature (the client deliberately never sets
temperature — non-default values 400 on Sonnet 5+), so a null may be sampling noise
rather than a property of the trace. This distinguishes the two:

  recovered on retry -> judge variance, and the original null rate overstated dropout
  still null         -> a real property of that trace

Writes the merged result back only if --apply, and always saves a retry report next to it.
"""

import asyncio
import json
from pathlib import Path

import fire
from dotenv import load_dotenv

from value_leakage.api.anthropic.messages import (
    extract_text, get_anthropic_client, process_batch)
from value_leakage.judge import TRAJECTORY_JUDGE_PROMPT, parse_trajectory

CONDITIONS = ("baseline", "below_good", "above_good")


async def retry(run_path: Path, model: str, max_concurrent: int) -> dict:
    traj = json.loads((run_path / "trajectories.json").read_text())
    client = get_anthropic_client()
    report = {}

    for c in CONDITIONS:
        vals = traj.get(c)
        if not vals:
            continue
        rows = json.loads((run_path / f"{c}.json").read_text())["rows"]
        nulls = [i for i, v in enumerate(vals) if v is None]

        # A rollout that FAILED AT SAMPLING has no 'reasoning' key at all — sample.py
        # writes {"i": i, "error": ...}. Those nulls are missing data, not unparseable
        # traces, and there is nothing to re-judge. deepseek-v4-pro's above_good is 43%
        # such rows. Separating them also stops the KeyError that silently aborted a
        # whole run's retry at its first affected condition.
        failed = [i for i in nulls if not (rows[i].get("reasoning") or "").strip()]
        retryable = [i for i in nulls if i not in set(failed)]

        report[c] = {"n_null": len(nulls), "failed_samples": len(failed),
                     "retryable": len(retryable), "recovered": 0,
                     "api_errors": 0, "still_null": 0}
        if not retryable:
            print(f"{c}: {len(nulls)} nulls, all of them failed samples — nothing to retry")
            continue

        msgs = [[{"role": "user", "content": TRAJECTORY_JUDGE_PROMPT.format(
            llm_text=rows[i]["reasoning"])}] for i in retryable]
        nulls = retryable
        print(f"{c}: retrying {len(nulls)} nulls", flush=True)
        responses = await process_batch(client=client, model=model,
                                        messages_list=msgs,
                                        max_concurrent=max_concurrent,
                                        return_exceptions=True)
        recovered, errors = 0, 0
        for i, r in zip(nulls, responses):
            if isinstance(r, Exception):
                errors += 1
                continue
            parsed = parse_trajectory(extract_text(r))
            if parsed is not None:
                vals[i] = parsed
                recovered += 1
        report[c].update({"recovered": recovered, "api_errors": errors,
                          "still_null": len(nulls) - recovered - errors})
        print(f"  recovered {recovered}/{len(nulls)} retryable  api_errors={errors}")

    return {"trajectories": traj, "report": report}


def main(run_dir: str, model: str = "claude-sonnet-5", max_concurrent: int = 30,
         apply: bool = False):
    load_dotenv(Path.cwd() / ".env")
    run_path = Path(run_dir)
    out = asyncio.run(retry(run_path, model, max_concurrent))

    (run_path / "trajectory_retry_report.json").write_text(
        json.dumps(out["report"], indent=2))
    print("\nreport:", json.dumps(out["report"], indent=2))

    if apply:
        (run_path / "trajectories.json").write_text(
            json.dumps(out["trajectories"], indent=2))
        print(f"applied merged trajectories to {run_path / 'trajectories.json'}")
    else:
        print("dry run — pass --apply to write the recovered trajectories back")


if __name__ == "__main__":
    fire.Fire(main)
