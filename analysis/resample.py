"""Step 5 — counterfactual importance (Thought Anchors, arXiv 2506.19143), adapted.

At position k we need two answer distributions:

  base        continue from the prefix INCLUDING the original unit k
  treatment   continue from the prefix ENDING AT unit k-1, so the model writes its own
              unit k; keep only continuations whose first unit MEANS something different

Everything after the resampled unit is regenerated, never spliced: the measured quantity
is unit k's total effect, direct plus indirect.

Cost control: the base set at position k is the same generation set as the treatment
source at position k+1. So we generate R continuations once per UNIQUE cut offset and
reuse them, rather than 2R per position.

Adaptations to this experiment (see analysis/PLAN.md section 4):
  - outcome is binary Y = 1[estimate > threshold] -> Bernoulli KL, comparable to the paper
  - meaning split is numeric-aware: sentence embedders rate "2,200" vs "1,200" at ~0.97
    cosine though it halves the answer, so numeric units split on the committed value
  - the split is a function of the RESAMPLED UNIT ONLY, never of the continuation or its
    answer, which would condition on the dependent variable

  python analysis/resample.py --model <id> --family qwen --run_dir runs/<name> \
      --positions_file analysis/positions_A.json --R 16
"""

import json
import math
import re
from pathlib import Path

import fire
import numpy as np

from local_gen import render, split_output
from forced_answer import parse_number
from segment import split_units
from value_leakage.sample import build_prompt

RESUME = {"qwen": "", "gptoss": "<|channel|>analysis<|message|>"}

NUM_TOKEN = re.compile(r"\d[\d,\.]*")
LOG_RATIO_DIFFERENT = 0.06        # ~15% relative change counts as a different commitment


def committed_value(text: str) -> float | None:
    """The largest number a unit commits to, or None. Used for the numeric meaning split."""
    vals = []
    for m in NUM_TOKEN.finditer(text.replace(" ", "")):
        raw = m.group().rstrip(".").replace(",", "")
        try:
            v = float(raw)
        except ValueError:
            continue
        if v >= 100:              # skip list indices, small counts, years handled loosely
            vals.append(v)
    return max(vals) if vals else None


def meaning_differs(original: str, candidate: str, embedder, cos_threshold: float) -> bool:
    a, b = committed_value(original), committed_value(candidate)
    if a is not None and b is not None and a > 0 and b > 0:
        return abs(math.log10(b / a)) > LOG_RATIO_DIFFERENT
    va, vb = embedder.encode([original, candidate], normalize_embeddings=True)
    return float(np.dot(va, vb)) < cos_threshold


def bernoulli_kl(p: float, q: float, eps: float = 1e-3) -> float:
    p = min(max(p, eps), 1 - eps)
    q = min(max(q, eps), 1 - eps)
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


def main(model: str, family: str, run_dir: str, positions_file: str,
         R: int = 16, max_tokens: int = 24000, max_model_len: int = 32768,
         temperature: float = 1.0, gpu_memory_utilization: float = 0.90,
         cos_threshold: float = 0.8, out_name: str = "counterfactual.json",
         embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
         conditions: str | None = None):
    """conditions: comma-separated subset to run (default: all in the positions file).
    Lets the two mirror conditions run on separate GPUs in parallel — the conditions are
    independent, so splitting them costs nothing analytically."""
    from sentence_transformers import SentenceTransformer
    from vllm import LLM, SamplingParams

    run_path = Path(run_dir)
    threshold = json.loads((run_path / "threshold.json").read_text())["threshold"]
    plan = json.loads(Path(positions_file).read_text())

    llm = LLM(model=model, max_model_len=max_model_len,
              gpu_memory_utilization=gpu_memory_utilization,
              enable_prefix_caching=True)
    tok = llm.get_tokenizer()
    params = SamplingParams(temperature=temperature, max_tokens=max_tokens, n=R)
    embedder = SentenceTransformer(embed_model, device="cpu")

    results = {"threshold": threshold, "model": model, "family": family, "R": R,
               "cos_threshold": cos_threshold, "conditions": {}}

    wanted = ([c.strip() for c in conditions.split(",")] if conditions
              else list(plan["conditions"]))
    for cond, traces in plan["conditions"].items():
        if cond not in wanted:
            continue
        rendered = render(tok, build_prompt(cond, threshold))
        rows = json.loads((run_path / f"{cond}.json").read_text())["rows"]
        segs = {s["i"]: s for s in
                json.loads((run_path / f"segments_{cond}.json").read_text())}

        # ---- unique cut offsets: base(k) and source(k-1) share generation sets --------
        jobs = {}                                  # (trace, offset) -> prompt
        for t in traces:
            ti, units = t["trace"], segs[t["trace"]]["units"]
            reasoning = rows[ti]["reasoning"]
            for k in t["positions"]:
                for kk in (k, k - 1):
                    off = units[kk]["end"] if kk >= 0 else 0
                    jobs[(ti, off)] = rendered + RESUME[family] + reasoning[:off]

        keys = sorted(jobs)
        print(f"\n=== {cond}: {len(keys)} unique prefixes x R={R} "
              f"({len(traces)} traces) ===", flush=True)
        outs = llm.generate([jobs[k] for k in keys], params)

        # ---- decode each continuation once: first unit (for the split) + final answer --
        pool = {}
        for key, o in zip(keys, outs):
            entries = []
            for c in o.outputs:
                reasoning_part, answer = split_output(family, c.text)
                first_units = split_units(reasoning_part)
                est = parse_number(answer) if answer else None
                entries.append({
                    "first_unit": first_units[0]["text"] if first_units else "",
                    "estimate": est,
                    "y": None if est is None else int(est > threshold),
                    "truncated": c.finish_reason == "length",
                })
            pool[key] = entries

        # ---- per position: base vs semantically-different resamples ------------------
        records = []
        for t in traces:
            ti, units = t["trace"], segs[t["trace"]]["units"]
            for k in t["positions"]:
                base = pool[(ti, units[k]["end"])]
                src = pool[(ti, units[k - 1]["end"] if k >= 1 else 0)]
                original = units[k]["text"]

                diff, same = [], []
                for e in src:
                    if not e["first_unit"]:
                        continue
                    (diff if meaning_differs(original, e["first_unit"],
                                             embedder, cos_threshold) else same).append(e)

                def rate(entries):
                    ys = [e["y"] for e in entries if e["y"] is not None]
                    return (float(np.mean(ys)) if ys else None), len(ys)

                p_base, n_base = rate(base)
                p_diff, n_diff = rate(diff)
                p_same, n_same = rate(same)

                rec = {
                    "trace": ti, "k": k, "tags": units[k]["tags"],
                    "text": original[:220],
                    "n_base": n_base, "n_diff": n_diff, "n_same": n_same,
                    "p_base": p_base, "p_diff": p_diff, "p_same": p_same,
                    # per-sample outcomes, so the analysis can measure the sampling-noise
                    # floor empirically (split-half of the base arm) instead of assuming it
                    "y_base": [e["y"] for e in base if e["y"] is not None],
                    "y_diff": [e["y"] for e in diff if e["y"] is not None],
                    "y_same": [e["y"] for e in same if e["y"] is not None],
                    "kl_diff_vs_base": (bernoulli_kl(p_diff, p_base)
                                        if None not in (p_diff, p_base) else None),
                    "kl_same_vs_base": (bernoulli_kl(p_same, p_base)
                                        if None not in (p_same, p_base) else None),
                    "delta": (p_base - p_diff) if None not in (p_diff, p_base) else None,
                    "truncation_rate": float(np.mean([e["truncated"] for e in base])),
                }
                # signed by the incentive direction: positive = the ORIGINAL unit pushed
                # the answer toward the side this condition rewards
                d = 1.0 if cond == "above_good" else -1.0
                rec["signed_effect"] = (d * rec["delta"]) if rec["delta"] is not None else None
                records.append(rec)

        results["conditions"][cond] = records
        usable = [r for r in records if r["n_diff"] >= 5 and r["p_diff"] is not None]
        print(f"  {len(usable)}/{len(records)} positions with >=5 different resamples")
        if usable:
            print(f"  median |delta| = "
                  f"{np.median([abs(r['delta']) for r in usable]):.3f}")

        # Checkpoint after EVERY condition. This run is hours long and a failure at the
        # final write would discard all of it — which has already happened once here.
        ckpt = run_path / out_name
        ckpt.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str))
        print(f"  checkpointed {ckpt} ({len(results['conditions'])} condition(s))",
              flush=True)

    out_path = run_path / out_name
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    fire.Fire(main)
