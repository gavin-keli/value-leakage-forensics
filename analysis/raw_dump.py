"""Print one generation's FULL raw decode with special tokens, plus the exact
channel/tag transition. Resampling prefixes must reproduce this byte-for-byte;
guessing the harmony separator would silently corrupt every continuation.
"""
import re
import fire

from value_leakage.sample import build_prompt
from local_gen import render


def main(model: str, family: str, max_tokens: int = 2500, max_model_len: int = 16384):
    from vllm import LLM, SamplingParams

    llm = LLM(model=model, max_model_len=max_model_len,
              gpu_memory_utilization=0.90, enable_prefix_caching=True)
    tok = llm.get_tokenizer()
    prompt = render(tok, build_prompt("above_good", 47_500_000))

    out = llm.generate([prompt], SamplingParams(temperature=1.0, max_tokens=max_tokens))
    raw = tok.decode(out[0].outputs[0].token_ids, skip_special_tokens=False)

    print("=== PROMPT TAIL (repr) ===")
    print(repr(prompt[-160:]))

    print("\n=== ALL SPECIAL-TOKEN MARKERS IN OUTPUT, IN ORDER ===")
    for m in re.finditer(r"<\|[^|]*\|>|</think>|<think>", raw):
        print(f"  @{m.start():>6}  {m.group()!r}")

    print("\n=== TRANSITION REGION (200 chars either side of the first channel switch) ===")
    markers = [m for m in re.finditer(r"<\|end\|>|<\|start\|>|</think>", raw)]
    if markers:
        p = markers[0].start()
        print(repr(raw[max(0, p - 200):p + 260]))
    else:
        print("(no transition marker found)")

    print(f"\n=== RAW LENGTH {len(raw)} ===")
    print("=== FIRST 200 ===")
    print(repr(raw[:200]))
    print("=== LAST 200 ===")
    print(repr(raw[-200:]))


if __name__ == "__main__":
    fire.Fire(main)
