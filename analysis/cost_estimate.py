"""Exact input size for the retry job — counts the actual traces that would be re-judged,
not a median-based extrapolation.

Counts only nulls that HAVE reasoning text (failed samples have nothing to re-judge).
"""
import json, glob, os
from pathlib import Path

CONDITIONS = ("baseline", "below_good", "above_good")
PROMPT_OVERHEAD_CHARS = 1900          # TRAJECTORY_JUDGE_PROMPT minus the {llm_text} slot
TARGETS = {
    "deepseek-v4-pro-0813": "A",
    "qwen3p8-2p4t-a95b": "A",
    "qwen3.5-122b-a10b": "B",
    "kimi-k3": "B",
    "local-qwen3p5-35b-a3b": "B",     # our own run, 2nd pass
}

totals = {}
print(f"{'run':<34} {'cond':<11} {'retryable':>10} {'chars':>12} {'~tokens':>10}")
for d in sorted(glob.glob("runs/*")):
    name = os.path.basename(d)
    key = next((k for k in TARGETS if name.startswith(k)), None)
    if key is None:
        continue
    traj = json.loads(Path(d, "trajectories.json").read_text())
    for c in CONDITIONS:
        vals = traj.get(c) or []
        rows = json.loads(Path(d, f"{c}.json").read_text())["rows"]
        idx = [i for i, v in enumerate(vals)
               if v is None and (rows[i].get("reasoning") or "").strip()]
        if not idx:
            continue
        chars = sum(len(rows[i]["reasoning"]) + PROMPT_OVERHEAD_CHARS for i in idx)
        # Claude tokenizer runs ~3.6 chars/token on this kind of mixed prose+numerals
        toks = chars / 3.6
        print(f"{key:<34} {c:<11} {len(idx):>10} {chars:>12,} {toks:>10,.0f}")
        g = totals.setdefault(TARGETS[key], {"calls": 0, "tokens": 0.0})
        g["calls"] += len(idx)
        g["tokens"] += toks

print()
grand = {"calls": 0, "tokens": 0.0}
for group, v in sorted(totals.items()):
    print(f"group {group}: {v['calls']} calls, {v['tokens']:,.0f} input tokens")
    grand["calls"] += v["calls"]
    grand["tokens"] += v["tokens"]
print(f"ALL:      {grand['calls']} calls, {grand['tokens']:,.0f} input tokens")
print(f"output tokens: ~{grand['calls'] * 120:,} (comma-separated integer lists)")
