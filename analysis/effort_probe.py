"""What does changing reasoning effort actually change in the prompt?

Renders the same message at each effort level and diffs the result, so the mechanism is
visible rather than assumed. Works for any model whose chat template accepts the kwarg.
"""
import sys
from transformers import AutoTokenizer

MODEL = sys.argv[1]
LEVELS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["low", "medium", "high"]

tok = AutoTokenizer.from_pretrained(MODEL)
msgs = [{"role": "user", "content": "How many spots?"}]

base = None
for lvl in LEVELS:
    try:
        out = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                      reasoning_effort=lvl)
    except Exception as e:
        print(f"{lvl}: template rejected reasoning_effort ({type(e).__name__}: {e})")
        continue
    if base is None:
        base = out
        print(f"=== {lvl} (reference), {len(out)} chars ===")
        print(repr(out[:340]))
    else:
        same = out == base
        print(f"\n=== {lvl} === identical to reference: {same}")
        if not same:
            # show the first differing region
            i = next((j for j in range(min(len(out), len(base))) if out[j] != base[j]), 0)
            print(f"  first difference at char {i}:")
            print(f"    reference: {base[max(0,i-40):i+60]!r}")
            print(f"    {lvl:<9}: {out[max(0,i-40):i+60]!r}")

print("\n--- does the template mention effort at all? ---")
tpl = tok.chat_template or ""
for kw in ("reasoning_effort", "Reasoning:", "enable_thinking", "thinking"):
    print(f"  {kw!r}: {kw in tpl}")
