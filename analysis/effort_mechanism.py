"""Mechanistically, what does `Reasoning: low|high` do?

Hypothesis: it is a learned control code that shifts the per-step probability of emitting
the token which ENDS the analysis channel. Effort would then be a prior over "am I done
thinking", and a small per-step change compounds over thousands of steps into a large
length difference.

Test: take ONE real analysis trace, truncate it at several depths, and put the identical
truncated text behind each system prompt. Ask for one token and read the probability mass
on the stop token. Everything except the effort word is byte-identical, so any difference
in P(stop) is attributable to that word alone.
"""
import json
import sys
from pathlib import Path

import fire

sys.path.insert(0, "analysis")
from local_gen import split_output
from value_leakage.sample import build_prompt

STOP_TOKENS = ("<|end|>", "<|return|>")


def main(model: str = "openai/gpt-oss-20b",
         run_dir: str = "runs/local-gpt-oss-20b_20260830",
         condition: str = "above_good", trace: int = 0,
         depths: str = "20,50,100,200,400", max_model_len: int = 16384):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    hf = AutoTokenizer.from_pretrained(model)
    stop_ids = {}
    for t in STOP_TOKENS:
        tid = hf.convert_tokens_to_ids(t)
        if tid is not None and tid >= 0:
            stop_ids[tid] = t
    print(f"stop tokens: {stop_ids}")

    threshold = json.loads(Path(run_dir, "threshold.json").read_text())["threshold"]
    rows = json.loads(Path(run_dir, f"{condition}.json").read_text())["rows"]
    reasoning = rows[trace]["reasoning"]
    body_ids = hf.encode(reasoning, add_special_tokens=False)
    print(f"trace {trace}: analysis is {len(body_ids)} tokens\n")

    user_msg = build_prompt(condition, threshold)
    llm = LLM(model=model, max_model_len=max_model_len, gpu_memory_utilization=0.90,
              enable_prefix_caching=True)
    # vLLM caps sample logprobs at 20 by default; the stop token sits near the top of the
    # distribution when it matters, so 20 is ample.
    params = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)

    # fire hands back a tuple when the CLI value contains commas, a str otherwise
    depth_list = [int(d) for d in
                  (depths.split(",") if isinstance(depths, str) else depths)]
    prompts, index = [], []
    for lvl in ("low", "high"):
        head = hf.apply_chat_template([{"role": "user", "content": user_msg}],
                                      tokenize=False, add_generation_prompt=True,
                                      reasoning_effort=lvl)
        for d in depth_list:
            if d > len(body_ids):
                continue
            partial = hf.decode(body_ids[:d])
            prompts.append(head + "<|channel|>analysis<|message|>" + partial)
            index.append((lvl, d))

    outs = llm.generate(prompts, params)

    table = {}
    for (lvl, d), o in zip(index, outs):
        lp = o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}
        p_stop = 0.0
        for tid, info in lp.items():
            if tid in stop_ids:
                p_stop += float(2.718281828 ** info.logprob)
        top = max(lp.items(), key=lambda kv: kv[1].logprob)
        table.setdefault(d, {})[lvl] = (p_stop, stop_ids.get(top[0], repr(
            hf.decode([top[0]]))[:18]))

    print(f"{'depth':>7} {'P(stop) low':>13} {'P(stop) high':>14} {'ratio':>9}   "
          f"top token low / high")
    for d in sorted(table):
        if set(table[d]) != {"low", "high"}:
            continue
        lo, hi = table[d]["low"][0], table[d]["high"][0]
        ratio = (lo / hi) if hi > 1e-12 else float("inf")
        print(f"{d:>7} {lo:>13.2e} {hi:>14.2e} {ratio:>9.1f}x   "
              f"{table[d]['low'][1]} / {table[d]['high'][1]}")

    Path(run_dir, "effort_mechanism.json").write_text(json.dumps(
        {str(d): {k: v[0] for k, v in lv.items()} for d, lv in table.items()}, indent=2))
    print(f"\nsaved {Path(run_dir, 'effort_mechanism.json')}")


if __name__ == "__main__":
    fire.Fire(main)
