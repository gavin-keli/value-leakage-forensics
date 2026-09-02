"""Step 2: forced-answer at k = -1 — the tilt before any reasoning exists.

Cut the trace at zero sentences and make the model answer immediately, then compare the
answer distribution between the mirror conditions. Anything here was set when the model
encoded the prompt, not by the chain of thought.

Per-family forcing, using each template's own structure:
  qwen    the template already emits '<think>\\n'; append '</think>' to close the block
          with nothing in it, so the visible answer starts immediately.
  gptoss  the template ends at '<|start|>assistant'; append '<|channel|>final<|message|>'
          to open the final channel directly, skipping the analysis channel.

  python analysis/forced_answer.py --model <id> --family qwen --run_dir runs/<name>
"""

import json
import re
from pathlib import Path

import fire
import numpy as np

from value_leakage.sample import build_prompt

from local_gen import render

CONDITIONS = ("baseline", "below_good", "above_good")

FORCE = {
    "qwen": "</think>\n\n",
    "gptoss": "<|channel|>final<|message|>",
}

SCALES = {"trillion": 1e12, "billion": 1e9, "million": 1e6, "thousand": 1e3,
          "t": 1e12, "b": 1e9, "m": 1e6, "k": 1e3}
# These models group thousands with narrow/thin/no-break spaces ("80\u202f000\u202f000"),
# so those are normalised to ASCII before grouping is undone.
_SEP_SPACES = "\u00a0\u2007\u2008\u2009\u202f\u2060"

# The number may not end on a comma or period, and the word boundary now applies only to
# the scale word. Requiring one after the number made "23,500,000 black" (spaces stripped)
# backtrack onto "23,500," and parse 1000x low.
NUM = re.compile(
    r"([0-9](?:[0-9,\.]*[0-9])?)\s*(?:(trillion|billion|million|thousand|[tbmk])\b)?",
    re.IGNORECASE)


def _normalise(text: str) -> str:
    """Undo digit grouping without letting digits collide with the following word.

    Removing every space merged "23,500,000 black spots" into "23,500,000black spots";
    with no boundary after the digits the pattern backtracked onto the comma. Only spaces
    sitting *between digit groups* are removed, which is what grouping actually produces.
    """
    for c in _SEP_SPACES:
        text = text.replace(c, " ")
    return re.sub(r"(?<=\d)[ ](?=\d{3}(?:\D|$))", "", text)


def parse_number(text: str) -> float | None:
    """First number in the text, scale words applied. None if nothing parses."""
    for m in NUM.finditer(_normalise(text)):
        raw, scale = m.group(1), m.group(2)
        raw = raw.rstrip(".").replace(",", "")
        if not raw or raw == ".":
            continue
        try:
            val = float(raw)
        except ValueError:
            continue
        if scale:
            val *= SCALES[scale.lower()]
        if val >= 1000:            # a bare "9" is a subspecies count, not an estimate
            return val
    return None


def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    if n == 0:
        return (float("nan"), float("nan"))
    p, d = k / n, 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return float(centre - half), float(centre + half)


def main(model: str, family: str, run_dir: str, count: int = 200,
         max_tokens: int = 400, max_model_len: int = 16384,
         temperature: float = 1.0, gpu_memory_utilization: float = 0.90,
         out_name: str = "forced_answer_k-1.json"):
    """max_tokens must be generous. At 60 the completion is cut mid-number, and because
    this model writes space-separated thousands ("45 300 000"), a truncated "45 300"
    parses as 45,300 — an order of magnitude low, and it still passes the >=1000 filter.
    Truncated completions are now dropped rather than parsed."""
    from vllm import LLM, SamplingParams

    run_path = Path(run_dir)
    threshold = json.loads((run_path / "threshold.json").read_text())["threshold"]
    print(f"threshold = {threshold:,}", flush=True)

    llm = LLM(model=model, max_model_len=max_model_len,
              gpu_memory_utilization=gpu_memory_utilization,
              enable_prefix_caching=True)
    tok = llm.get_tokenizer()
    params = SamplingParams(temperature=temperature, max_tokens=max_tokens, n=1)

    out = {"threshold": threshold, "count": count, "conditions": {}}
    for condition in CONDITIONS:
        prompt = render(tok, build_prompt(condition, threshold)) + FORCE[family]
        gens = llm.generate([prompt] * count, params)
        # A completion cut at max_tokens may be cut MID-NUMBER, and this model writes
        # space-separated thousands ("45 300 000"), so a cut "45 300" parses as 45,300 —
        # an order of magnitude low, and it passes the >=1000 filter. Rather than discard
        # every truncated completion (which selects for short answers), strip only a
        # trailing run of digits/separators from truncated text: a number followed by more
        # text is provably complete, one at the very end may not be.
        TRAILING_NUM = re.compile(r"[\d,.\s]+$")
        texts, n_trunc = [], 0
        for g in gens:
            t = g.outputs[0].text
            if g.outputs[0].finish_reason == "length":
                n_trunc += 1
                t = TRAILING_NUM.sub("", t)
            texts.append(t)
        kept = gens
        vals = [parse_number(t) for t in texts]
        good = [v for v in vals if v is not None]
        k = sum(1 for v in good if v > threshold)
        lo, hi = wilson(k, len(good))
        rec = {
            "n": count, "n_truncated_dropped": n_trunc, "n_parsed": len(good),
            "p_above_threshold": k / len(good) if good else None,
            "ci95": [lo, hi],
            "median": float(np.median(good)) if good else None,
            "median_rel_deviation": float(np.median(good) / threshold - 1) if good else None,
            "sample_texts": [t.strip()[:160] for t in texts[:3]],
        }
        out["conditions"][condition] = rec
        print(f"{condition:<11} parsed {len(good)}/{count} "
              f"({n_trunc} truncated, trailing digits stripped)  "
              f"P(>thr)={rec['p_above_threshold']:.3f} [{lo:.3f},{hi:.3f}]  "
              f"median={rec['median']:,.0f}  rel_dev={rec['median_rel_deviation']:+.3f}",
              flush=True)

    a = out["conditions"]["above_good"]["p_above_threshold"]
    b = out["conditions"]["below_good"]["p_above_threshold"]
    if a is not None and b is not None:
        out["pre_reasoning_gap"] = a - b
        print(f"\nPRE-REASONING GAP (above - below) = {a - b:+.3f}")
        print("Any gap here predates the chain of thought entirely.")

    (run_path / out_name).write_text(json.dumps(out, indent=2))
    print(f"saved {run_path / out_name}")


if __name__ == "__main__":
    fire.Fire(main)
