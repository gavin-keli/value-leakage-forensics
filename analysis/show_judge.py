"""Show exactly what the LLM judge is given and what it returns, for one real rollout.

Two judges run over every rollout:
  estimate judge    reads the VISIBLE ANSWER  -> one final number
  trajectory judge  reads the REASONING TRACE -> the ordered list of candidate numbers
                                                 the model floated while thinking
Neither judge sees the incentive condition, the threshold, or the other judge's output.
"""
import json
import sys
from pathlib import Path

from value_leakage.judge import NUMBER_JUDGE_PROMPT, TRAJECTORY_JUDGE_PROMPT

run_dir, cond = sys.argv[1], sys.argv[2]
idx = int(sys.argv[3]) if len(sys.argv) > 3 else 0

rows = json.loads(Path(run_dir, f"{cond}.json").read_text())["rows"]
est = json.loads(Path(run_dir, "estimates.json").read_text()).get(cond)
traj = json.loads(Path(run_dir, "trajectories.json").read_text()).get(cond)
row = rows[idx]

print("=" * 78)
print(f"ROLLOUT {idx} of {cond}   (reasoning {len(row['reasoning']):,} chars, "
      f"answer {len(row['content']):,} chars)")
print("=" * 78)

print("\n" + "#" * 78)
print("# JUDGE 1 — ESTIMATE JUDGE.  Input = the visible answer only.")
print("#" * 78)
filled = NUMBER_JUDGE_PROMPT.format(llm_text=row["content"][:700])
print(filled[:1800])
print(f"\n>>> JUDGE RETURNED (parsed): {est[idx] if est else 'n/a'}")

print("\n" + "#" * 78)
print("# JUDGE 2 — TRAJECTORY JUDGE.  Input = the reasoning trace.")
print("#" * 78)
tpl = TRAJECTORY_JUDGE_PROMPT.format(llm_text="<<<REASONING TRACE GOES HERE>>>")
print(tpl[:2100])
print("\n--- the actual trace it was given (first 600 chars) ---")
print(row["reasoning"][:600])
print(f"\n>>> JUDGE RETURNED (parsed): {traj[idx] if traj else 'n/a'}")
