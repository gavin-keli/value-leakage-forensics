"""Dump one real rollout as sentences, to ground the resampling explanation.

Also checks which conditions estimates.json actually covers.
"""
import json, re, glob, os
from pathlib import Path

print("=== which conditions are judged in estimates.json vs trajectories.json ===")
for d in sorted(glob.glob("runs/*")):
    e = json.loads(Path(d, "estimates.json").read_text()) if os.path.exists(f"{d}/estimates.json") else {}
    t = json.loads(Path(d, "trajectories.json").read_text()) if os.path.exists(f"{d}/trajectories.json") else {}
    print(f"  {os.path.basename(d):<38} estimates={list(e)} trajectories={list(t)}")

RUN = Path("runs/qwen3.5-122b-a10b_20260815_030702")
COND = "above_good"

thr = json.loads((RUN / "threshold.json").read_text())["threshold"]
rows = json.loads((RUN / f"{COND}.json").read_text())["rows"]
traj = json.loads((RUN / "trajectories.json").read_text())[COND]

print(f"\nthreshold = {thr:,}")

cands = [i for i, t in enumerate(traj)
         if isinstance(t, list) and 3 <= len(t) <= 10
         and rows[i].get("reasoning")]
cands.sort(key=lambda i: len(rows[i]["reasoning"]))
print(f"{len(cands)} candidate rollouts\n")

SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z*#\-\d])')

for i in cands[:1]:
    r = rows[i]
    print("=" * 78)
    print(f"rollout {i} | judged in-CoT estimates: {traj[i]}")
    print("=" * 78)
    sents = [s.strip() for s in SPLIT.split(r["reasoning"]) if s.strip()]
    print(f"{len(sents)} sentences\n")
    for k, s in enumerate(sents):
        low, flag = s.lower(), ""
        if any(w in low for w in ("bet", "donat", "threshold", "good cause", "bad cause")):
            flag += " [INC]"
        if re.search(r'\d', s):
            flag += " [NUM]"
        print(f"  s{k:>2}{flag}: {s[:135]}")
    print(f"\n--- visible answer (first 500 chars) ---\n{(r.get('content') or '')[:500]}")
