"""What does Qwen3.5 expose instead of graded reasoning effort?"""
import sys
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(sys.argv[1])
msgs = [{"role": "user", "content": "How many spots?"}]

for kwargs in ({}, {"enable_thinking": True}, {"enable_thinking": False}):
    out = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                  **kwargs)
    print(f"{str(kwargs) or '(default)':<28} -> {out[-60:]!r}")
