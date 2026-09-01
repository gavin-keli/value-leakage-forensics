"""Local vLLM sampling for the value-leakage prompts, writing the repo's run-dir schema.

Uses the OFFLINE generate() API on fully-rendered prompt strings rather than chat(),
because the resampling work needs exact prefix control — chat() would re-apply the
template over a partial chain of thought. One model load serves both jobs.

Reasoning/answer split is done here rather than by a vLLM reasoning parser:

  qwen   the chat template ends with '<|im_start|>assistant\\n<think>\\n', so the
         opening tag is in the PROMPT and the model emits only '</think>'.
  gptoss harmony channels: the model emits an analysis channel then a final channel;
         in decoded text these appear as bare 'analysis' / 'assistantfinal' markers.

Output matches value_leakage's schema exactly, so judge.py / plot.py / panel.py all
work on these run dirs unchanged.

  python analysis/local_gen.py --model <hf-id> --family qwen --condition baseline \
      --out runs/<name>/baseline.json --count 100
"""

import json
import time
from pathlib import Path

import fire

from value_leakage.sample import build_prompt

FAMILIES = ("qwen", "gptoss")


def split_output(family: str, text: str) -> tuple[str, str]:
    """(reasoning, visible answer). Empty answer => the trace never closed."""
    if family == "qwen":
        marker = "</think>"
        if marker in text:
            head, _, tail = text.partition(marker)
            return head.strip(), tail.strip()
        return text.strip(), ""
    # gptoss
    for marker in ("assistantfinal", "<|channel|>final<|message|>"):
        if marker in text:
            head, _, tail = text.partition(marker)
            head = head.split("analysis", 1)[-1] if head.startswith("analysis") else head
            return head.strip(), tail.strip()
    return text.strip(), ""


def render(tokenizer, user_msg: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_msg}],
        tokenize=False, add_generation_prompt=True)


def main(
    model: str,
    family: str,
    condition: str,
    out: str,
    count: int = 100,
    threshold: int | None = None,
    max_tokens: int = 12000,
    max_model_len: int = 32768,
    temperature: float = 1.0,
    top_p: float = 1.0,
    gpu_memory_utilization: float = 0.90,
):
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}, got {family!r}")

    from vllm import LLM, SamplingParams

    user_msg = build_prompt(condition, threshold)
    llm = LLM(model=model, max_model_len=max_model_len,
              gpu_memory_utilization=gpu_memory_utilization,
              enable_prefix_caching=True)
    tokenizer = llm.get_tokenizer()
    prompt = render(tokenizer, user_msg)

    print(f"{model} | {condition} | n={count} | temp={temperature} | max_tokens={max_tokens}",
          flush=True)
    t0 = time.time()
    outs = llm.generate(
        [prompt] * count,
        SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens, n=1),
    )
    dt = time.time() - t0

    rows, truncated, unclosed = [], 0, 0
    for i, o in enumerate(outs):
        gen = o.outputs[0]
        reasoning, answer = split_output(family, gen.text)
        if gen.finish_reason == "length":
            truncated += 1
        if not answer:
            unclosed += 1
        rows.append({
            "i": i,
            "reasoning": reasoning,
            "content": answer,
            "finish_reason": gen.finish_reason,
            "usage": {"prompt_tokens": len(o.prompt_token_ids),
                      "completion_tokens": len(gen.token_ids)},
        })

    ntok = sum(r["usage"]["completion_tokens"] for r in rows)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "model": model.split("/")[-1], "backend": "local-vllm", "provider": None,
        "family": family, "condition": condition, "threshold": threshold,
        "prompt": user_msg, "rendered_prompt": prompt,
        "max_tokens": max_tokens, "temperature": temperature, "top_p": top_p,
        "reasoning_effort": None, "rows": rows,
    }, indent=2, ensure_ascii=False))

    print(f"{count} rollouts in {dt:.0f}s | {ntok} completion tokens = {ntok/dt:.0f} tok/s")
    print(f"truncated (hit max_tokens): {truncated}/{count} | "
          f"no closing marker: {unclosed}/{count}")
    print(f"median reasoning chars: "
          f"{sorted(len(r['reasoning']) for r in rows)[len(rows)//2]}")
    print(f"saved {out_path}")


if __name__ == "__main__":
    fire.Fire(main)
