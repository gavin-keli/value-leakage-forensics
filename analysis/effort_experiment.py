"""What changes behaviourally when gpt-oss's reasoning effort changes?

Same prompt, same condition, same sampling — only the `Reasoning:` line differs. Measures
trace length, unit counts, tag composition, and the answer distribution, so "high effort
thinks more" becomes a set of numbers rather than an impression.

Relevant to two open questions in the write-up:
  - does the trace-length gap between gpt-oss and Qwen3.5 shrink at high effort?
  - does gpt-oss start producing DISAVOWAL units when it reasons longer, or is their
    absence a property of the model rather than of how much it writes?
"""
import json
import sys
from pathlib import Path

import fire
import numpy as np

sys.path.insert(0, "analysis")
from segment import segment_trace
from local_gen import split_output
from forced_answer import parse_number
from value_leakage.sample import build_prompt

TAGS = ("incentive-acknowledgment", "disavowal", "parameter-selection",
        "threshold-comparison", "directional-sensitivity-check")


def main(model: str = "openai/gpt-oss-20b", run_dir: str = "runs/local-gpt-oss-20b_20260830",
         condition: str = "above_good", count: int = 40, levels: str = "low,medium,high",
         max_tokens: int = 12000, max_model_len: int = 16384, temperature: float = 1.0):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    threshold = json.loads(Path(run_dir, "threshold.json").read_text())["threshold"]
    user_msg = build_prompt(condition, threshold)
    hf_tok = AutoTokenizer.from_pretrained(model)

    llm = LLM(model=model, max_model_len=max_model_len, gpu_memory_utilization=0.90,
              enable_prefix_caching=True)
    params = SamplingParams(temperature=temperature, max_tokens=max_tokens, n=1)

    out = {"model": model, "condition": condition, "count": count,
           "threshold": threshold, "levels": {}}

    for lvl in levels.split(","):
        prompt = hf_tok.apply_chat_template(
            [{"role": "user", "content": user_msg}], tokenize=False,
            add_generation_prompt=True, reasoning_effort=lvl)
        gens = llm.generate([prompt] * count, params)

        toks, units, tagcount, ests, truncated = [], [], {t: 0 for t in TAGS}, [], 0
        for g in gens:
            o = g.outputs[0]
            toks.append(len(o.token_ids))
            truncated += o.finish_reason == "length"
            reasoning, answer = split_output("gptoss", o.text)
            us = segment_trace(reasoning)
            units.append(len(us))
            for u in us:
                for t in u["tags"]:
                    if t in tagcount:
                        tagcount[t] += 1
            v = parse_number(answer) if answer else None
            if v is not None:
                ests.append(v)

        rec = {
            "median_tokens": float(np.median(toks)),
            "median_units": float(np.median(units)),
            "truncated": truncated,
            "tags_per_trace": {t: round(c / count, 2) for t, c in tagcount.items()},
            "n_parsed": len(ests),
            "median_estimate": float(np.median(ests)) if ests else None,
            "p_above_threshold": (float(np.mean([e > threshold for e in ests]))
                                  if ests else None),
        }
        out["levels"][lvl] = rec
        print(f"\n=== Reasoning: {lvl} ===")
        print(f"  median tokens {rec['median_tokens']:,.0f} | median units "
              f"{rec['median_units']:,.0f} | truncated {truncated}/{count}")
        print(f"  tags per trace: " + "  ".join(
            f"{t.split('-')[0]}={v}" for t, v in rec["tags_per_trace"].items()))
        print(f"  parsed {rec['n_parsed']}/{count} | median est "
              f"{rec['median_estimate']:,.0f} | P(>thr) {rec['p_above_threshold']:.3f}")

    Path(run_dir, "effort_experiment.json").write_text(json.dumps(out, indent=2))
    print(f"\nsaved {Path(run_dir, 'effort_experiment.json')}")


if __name__ == "__main__":
    fire.Fire(main)
