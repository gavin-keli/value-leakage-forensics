"""Print the COMPLETE rendered prompt for each setting, plus an exact character diff.

No truncation: the point is to see precisely which bytes the setting controls.
"""
import difflib
import sys

from transformers import AutoTokenizer

MODEL = sys.argv[1]
MODE = sys.argv[2] if len(sys.argv) > 2 else "effort"   # 'effort' | 'thinking'

tok = AutoTokenizer.from_pretrained(MODEL)
MSG = [{"role": "user", "content": "How many black spots are there on all giraffes?"}]

if MODE == "effort":
    settings = [("reasoning_effort=low", {"reasoning_effort": "low"}),
                ("reasoning_effort=medium", {"reasoning_effort": "medium"}),
                ("reasoning_effort=high", {"reasoning_effort": "high"})]
else:
    settings = [("(no kwargs)", {}),
                ("enable_thinking=True", {"enable_thinking": True}),
                ("enable_thinking=False", {"enable_thinking": False}),
                ("reasoning_effort=high", {"reasoning_effort": "high"})]

rendered = []
for label, kw in settings:
    try:
        out = tok.apply_chat_template(MSG, tokenize=False, add_generation_prompt=True, **kw)
    except Exception as e:
        out = f"<<REJECTED: {type(e).__name__}: {e}>>"
    rendered.append((label, out))
    print(f"\n{'='*76}\n{label}   ({len(out)} chars)\n{'='*76}")
    print(out)

print(f"\n{'='*76}\nCHARACTER DIFF vs the first setting\n{'='*76}")
base_label, base = rendered[0]
for label, out in rendered[1:]:
    if out == base:
        print(f"\n{label}: IDENTICAL to {base_label} — the setting had no effect")
        continue
    print(f"\n{label}: differs from {base_label}")
    for line in difflib.unified_diff(base.splitlines(), out.splitlines(),
                                     fromfile=base_label, tofile=label, lineterm="", n=1):
        print("   " + line)
