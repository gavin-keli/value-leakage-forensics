"""Step 1b: sample both incentive conditions in ONE model load.

Reads threshold.json (written when the baseline was judged) so the two conditions are
sampled against exactly the threshold the baseline defined.

  python analysis/run_conditions.py --model <id> --family qwen --run_dir runs/<name>
"""

import json
import time
from pathlib import Path

import fire

from value_leakage.sample import build_prompt

from local_gen import render, split_output

CONDITIONS = ("below_good", "above_good")


def main(model: str, family: str, run_dir: str, count: int = 100,
         max_tokens: int = 24000, max_model_len: int = 32768,
         temperature: float = 1.0, top_p: float = 1.0,
         gpu_memory_utilization: float = 0.90):
    from vllm import LLM, SamplingParams

    run_path = Path(run_dir)
    threshold = json.loads((run_path / "threshold.json").read_text())["threshold"]
    print(f"threshold = {threshold:,}", flush=True)

    llm = LLM(model=model, max_model_len=max_model_len,
              gpu_memory_utilization=gpu_memory_utilization,
              enable_prefix_caching=True)
    tok = llm.get_tokenizer()
    params = SamplingParams(temperature=temperature, top_p=top_p,
                            max_tokens=max_tokens, n=1)

    for condition in CONDITIONS:
        user_msg = build_prompt(condition, threshold)
        prompt = render(tok, user_msg)
        print(f"\n=== {condition} | n={count} ===", flush=True)

        t0 = time.time()
        outs = llm.generate([prompt] * count, params)
        dt = time.time() - t0

        rows, truncated, unclosed = [], 0, 0
        for i, o in enumerate(outs):
            gen = o.outputs[0]
            reasoning, answer = split_output(family, gen.text)
            truncated += gen.finish_reason == "length"
            unclosed += not answer
            rows.append({
                "i": i, "reasoning": reasoning, "content": answer,
                "finish_reason": gen.finish_reason,
                "usage": {"prompt_tokens": len(o.prompt_token_ids),
                          "completion_tokens": len(gen.token_ids)},
            })

        ntok = sum(r["usage"]["completion_tokens"] for r in rows)
        (run_path / f"{condition}.json").write_text(json.dumps({
            "model": model.split("/")[-1], "backend": "local-vllm", "provider": None,
            "family": family, "condition": condition, "threshold": threshold,
            "prompt": user_msg, "rendered_prompt": prompt,
            "max_tokens": max_tokens, "temperature": temperature, "top_p": top_p,
            "reasoning_effort": None, "rows": rows,
        }, indent=2, ensure_ascii=False))

        print(f"{count} rollouts in {dt:.0f}s | {ntok} tokens = {ntok/dt:.0f} tok/s")
        print(f"truncated: {truncated}/{count} | no closing marker: {unclosed}/{count}")
        print(f"median reasoning chars: "
              f"{sorted(len(r['reasoning']) for r in rows)[len(rows)//2]}")
        print(f"saved {run_path / f'{condition}.json'}", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
