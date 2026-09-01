"""Step 4 — forced-answer screen across truncation positions.

At each cut point k, freeze the trace through unit k, force an immediate answer, and
record P_k = P(estimate > threshold) over R samples. k = -1 (no reasoning at all) is
included, so this also produces the step-2 pre-reasoning tilt in the same model load.

This is the paper's FORCED-ANSWER importance, not counterfactual importance. Its known
failure mode (Thought Anchors, arXiv 2506.19143): when a necessary sentence appears late,
every earlier position's importance is artificially suppressed. Used here only to map
where the answer becomes determined and to nominate positions for step 5 — never as the
causal verdict.

Prefix construction is per-family and verified against raw decodes with special tokens
preserved (see analysis/STATUS.md section 4):
  qwen    the chat template already emits '<think>\\n', so resuming is just
          prompt + reasoning[:cut]; forcing an answer appends '</think>'.
  gptoss  harmony channels: resume inside the analysis channel, then close it and open
          the final channel. At k = -1 there is no analysis channel to close.

  python analysis/screen.py --model <id> --family qwen --run_dir runs/<name> \
      --n_traces 6 --max_positions 60 --R 24
"""

import json
from pathlib import Path

import fire
import numpy as np

from local_gen import render
from forced_answer import parse_number, wilson
from value_leakage.sample import build_prompt

RESUME = {"qwen": "", "gptoss": "<|channel|>analysis<|message|>"}
FORCE_AFTER = {"qwen": "\n</think>\n\n",
               "gptoss": "<|end|><|start|>assistant<|channel|>final<|message|>"}
FORCE_EMPTY = {"qwen": "\n</think>\n\n", "gptoss": "<|channel|>final<|message|>"}


def pick_positions(units: list, max_positions: int) -> list[int]:
    """A uniform grid plus every tagged unit, deduped and ordered.

    Tagged units are chosen a priori from the hypothesis, so they must not depend on any
    outcome; the grid gives unbiased coverage to compare them against.
    """
    n = len(units)
    tagged = [int(u["i"]) for u in units if u["tags"]]
    grid_n = max(2, max_positions - len(tagged))
    # int() everywhere: np.linspace yields np.int64, which json.dumps cannot serialise —
    # and the failure lands at the very END of the run, after all the GPU work is spent.
    grid = [int(x) for x in np.unique(np.linspace(0, n - 1, min(grid_n, n)).astype(int))]
    picked = sorted(set(grid) | set(tagged))
    if len(picked) > max_positions:            # tags alone can exceed the cap
        keep = {int(x) for x in np.unique(
            np.linspace(0, len(picked) - 1, max_positions).astype(int))}
        picked = [p for j, p in enumerate(picked) if j in keep]
    return [int(p) for p in picked]


def main(model: str, family: str, run_dir: str,
         conditions: str = "below_good,above_good",
         n_traces: int = 6, max_positions: int = 60, R: int = 24,
         max_tokens: int = 60, max_model_len: int = 32768,
         temperature: float = 1.0, gpu_memory_utilization: float = 0.90,
         out_name: str = "screen.json"):
    from vllm import LLM, SamplingParams

    run_path = Path(run_dir)
    threshold = json.loads((run_path / "threshold.json").read_text())["threshold"]
    conds = [c.strip() for c in conditions.split(",")]

    llm = LLM(model=model, max_model_len=max_model_len,
              gpu_memory_utilization=gpu_memory_utilization,
              enable_prefix_caching=True)
    tok = llm.get_tokenizer()
    params = SamplingParams(temperature=temperature, max_tokens=max_tokens, n=R)

    results = {"threshold": threshold, "model": model, "family": family,
               "R": R, "max_positions": max_positions, "conditions": {}}

    for cond in conds:
        rendered = render(tok, build_prompt(cond, threshold))
        rows = json.loads((run_path / f"{cond}.json").read_text())["rows"]
        segs = json.loads((run_path / f"segments_{cond}.json").read_text())

        # k = -1: no reasoning at all. One prompt, R samples.
        prompts = [rendered + FORCE_EMPTY[family]]
        index = [{"trace": None, "k": -1}]

        chosen = segs[:n_traces]
        for s in chosen:
            reasoning = rows[s["i"]]["reasoning"]
            units = s["units"]
            if not units:
                continue
            for k in pick_positions(units, max_positions):
                cut = units[k]["end"]
                prompts.append(rendered + RESUME[family] + reasoning[:cut]
                               + FORCE_AFTER[family])
                index.append({"trace": s["i"], "k": k,
                              "tags": units[k]["tags"],
                              "text": units[k]["text"][:200]})

        print(f"\n=== {cond}: {len(prompts)} prefixes x R={R} "
              f"({len(chosen)} traces) ===", flush=True)
        outs = llm.generate(prompts, params)

        records = []
        for meta, o in zip(index, outs):
            vals = [parse_number(c.text) for c in o.outputs]
            good = [v for v in vals if v is not None]
            k_above = sum(1 for v in good if v > threshold)
            lo, hi = wilson(k_above, len(good))
            records.append({
                **meta,
                "n_parsed": len(good), "n": len(vals),
                "p_above": k_above / len(good) if good else None,
                "ci95": [lo, hi],
                "median": float(np.median(good)) if good else None,
            })
        results["conditions"][cond] = records

        anchor = records[0]
        print(f"  k=-1 (no reasoning): P(>thr)={anchor['p_above']}, "
              f"median={anchor['median']}")
        parsed_rate = np.mean([r["n_parsed"] / r["n"] for r in records])
        print(f"  mean parse rate {parsed_rate:.1%}")

    out_path = run_path / out_name
    # default=str is a belt-and-braces guard: a serialisation failure here would discard
    # the entire run's GPU work at the last line.
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    print(f"\nsaved {out_path}")

    # the quantity the screen exists to produce: where the two conditions separate
    if len(conds) == 2:
        a, b = (results["conditions"][c] for c in conds)
        for label, recs in ((conds[0], a), (conds[1], b)):
            byk = {}
            for r in recs:
                if r["k"] is None or r["p_above"] is None or r["trace"] is None:
                    continue
                byk.setdefault(r["k"] // 10 * 10, []).append(r["p_above"])
            line = "  ".join(f"{k}:{np.mean(v):.2f}" for k, v in sorted(byk.items())[:12])
            print(f"{label:<11} P(>thr) by position decile: {line}")


if __name__ == "__main__":
    fire.Fire(main)
