"""Step 6: validate the regex answer parser against the Claude estimate judge.

The write-up lists this as an open limitation, and the k=-1 bug showed why parse RATE is
not enough: 97.5% of completions parsed while the parsed VALUES were an order of magnitude
wrong, because a truncated "45 300 000" reads as 45,300 and still passes a >=1000 filter.

Samples visible answers across runs, parses each with the regex used in the screen and
resampling passes, judges the same text with the estimate judge, and reports:
  - agreement within a relative tolerance (order-of-magnitude errors cannot hide)
  - the disagreements themselves, so failure modes are visible rather than counted

  python analysis/validate_parser.py --runs runs/a,runs/b --n 200
"""

import asyncio
import json
import random
from pathlib import Path

import fire
import numpy as np
from dotenv import load_dotenv

from value_leakage.api.anthropic.messages import (
    extract_text, get_anthropic_client, process_batch)
from value_leakage.judge import NUMBER_JUDGE_PROMPT, parse_tagged_estimate

import sys
sys.path.insert(0, "analysis")
from forced_answer import parse_number

CONDITIONS = ("baseline", "below_good", "above_good")


async def judge_all(texts, model, max_concurrent):
    client = get_anthropic_client()
    msgs = [[{"role": "user", "content": NUMBER_JUDGE_PROMPT.format(llm_text=t)}]
            for t in texts]
    responses = await process_batch(client=client, model=model, messages_list=msgs,
                                    max_concurrent=max_concurrent, return_exceptions=True)
    out = []
    for r in responses:
        out.append(None if isinstance(r, Exception)
                   else parse_tagged_estimate(extract_text(r)))
    return out


def main(runs: str, n: int = 200, model: str = "claude-sonnet-5",
         max_concurrent: int = 40, tol: float = 0.02, seed: int = 0):
    load_dotenv(Path.cwd() / ".env")
    rng = random.Random(seed)

    pool = []
    for run_dir in runs.split(","):
        p = Path(run_dir.strip())
        for c in CONDITIONS:
            f = p / f"{c}.json"
            if not f.exists():
                continue
            for row in json.loads(f.read_text())["rows"]:
                txt = (row.get("content") or "").strip()
                if txt:
                    pool.append((p.name, c, row["i"], txt))
    rng.shuffle(pool)
    sample = pool[:n]
    print(f"sampled {len(sample)} visible answers from {len(pool)} available")

    judged = asyncio.run(judge_all([t for *_, t in sample], model, max_concurrent))

    agree = disagree = judge_none = regex_none = both_none = 0
    rows = []
    for (run, cond, idx, txt), j in zip(sample, judged):
        r = parse_number(txt)
        if r is None and j is None:
            both_none += 1
        elif j is None:
            judge_none += 1
        elif r is None:
            regex_none += 1
        elif abs(r - j) <= tol * max(abs(j), 1):
            agree += 1
        else:
            disagree += 1
            rows.append((run, cond, idx, r, j, txt[:110]))

    scored = agree + disagree
    print(f"\nboth parsed        : {scored}")
    print(f"  agree (<={tol:.0%} rel): {agree}  ({agree/max(1,scored):.1%})")
    print(f"  disagree          : {disagree}")
    print(f"judge UNKNOWN, regex got a number : {judge_none}")
    print(f"regex None, judge got a number    : {regex_none}")
    print(f"both declined                     : {both_none}")

    if rows:
        print(f"\ndisagreements (regex vs judge):")
        for run, cond, idx, r, j, txt in rows[:15]:
            ratio = r / j if j else float('inf')
            print(f"  {run[:22]:<22} {cond[:5]} #{idx:<3} regex={r:>15,.0f} "
                  f"judge={j:>15,.0f}  x{ratio:>7.2f}")
            print(f"      {txt}")
        oom = sum(1 for *_, r, j, _ in rows if j and (r / j > 5 or r / j < 0.2))
        print(f"\norder-of-magnitude disagreements (>5x either way): {oom}/{len(rows)}")

    Path("analysis/parser_validation.json").write_text(json.dumps(
        {"n": len(sample), "agree": agree, "disagree": disagree,
         "judge_unknown": judge_none, "regex_none": regex_none,
         "both_none": both_none, "tolerance": tol}, indent=2))
    print(f"\nsaved analysis/parser_validation.json")


if __name__ == "__main__":
    fire.Fire(main)
