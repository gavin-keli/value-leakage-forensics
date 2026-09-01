"""Can we capture the reasoning section for this model, offline, with our own splitter?

Prints, per rollout:
  - finish_reason (a truncated trace has no closing marker and is NOT a parse failure)
  - the raw decode WITH special tokens, so the real channel/tag structure is visible
  - what split_output() returned, and the exact text either side of the boundary

Run on both boxes before committing to the real sampling.
"""

import sys
sys.path.insert(0, "analysis")

import fire
from local_gen import render, split_output


def main(model: str, family: str, count: int = 3, max_tokens: int = 12000,
         max_model_len: int = 32768):
    from vllm import LLM, SamplingParams

    from value_leakage.sample import build_prompt

    # above_good exercises the incentive text too, not just the plain question
    user_msg = build_prompt("above_good", threshold=30_000_000)

    llm = LLM(model=model, max_model_len=max_model_len,
              gpu_memory_utilization=0.90, enable_prefix_caching=True)
    tok = llm.get_tokenizer()
    prompt = render(tok, user_msg)

    print("=" * 70)
    print("RENDERED PROMPT TAIL:", repr(prompt[-120:]))
    print("=" * 70, flush=True)

    outs = llm.generate([prompt] * count,
                        SamplingParams(temperature=1.0, max_tokens=max_tokens))

    ok = 0
    for i, o in enumerate(outs):
        gen = o.outputs[0]
        raw = tok.decode(gen.token_ids, skip_special_tokens=False)
        reasoning, answer = split_output(family, gen.text)
        good = bool(reasoning) and bool(answer)
        ok += good

        print(f"\n{'=' * 70}\nrollout {i} | finish={gen.finish_reason} | "
              f"tokens={len(gen.token_ids)} | CAPTURE {'OK' if good else 'FAILED'}")
        print(f"  reasoning chars={len(reasoning)}  answer chars={len(answer)}")
        print("--- raw WITH special tokens, first 300 ---")
        print(repr(raw[:300]))
        print("--- raw WITH special tokens, last 300 ---")
        print(repr(raw[-300:]))
        print("--- reasoning TAIL (last 250) ---")
        print(reasoning[-250:])
        print("--- answer HEAD (first 250) ---")
        print(answer[:250])

    print(f"\n{'=' * 70}\nCAPTURE SUMMARY: {ok}/{count} rollouts split cleanly")


if __name__ == "__main__":
    fire.Fire(main)
