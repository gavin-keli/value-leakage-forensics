# Where motivated reasoning enters the trace

A replication and extension of the **value leakage** experiment: a model is asked to estimate a
quantity while a donation outcome depends on whether its answer clears a threshold. The
incentive is irrelevant to the true value, so any systematic difference between the two mirror
conditions is motivated reasoning.

This repository re-runs the paradigm on two open-weight models whose weights and raw chain of
thought are available locally, then applies sentence-level causal interventions to ask *which
sentences actually carry the bias*.

> **Headline: the metric the original analysis ranks models by misses the largest incentive
> effect measured here.** Qwen3.5-35B-A3B scores MRF +0.012 (permutation p = 0.32,
> indistinguishable from zero) while the probability its final answer clears the threshold moves
> from 0.38 to 0.86 between conditions — a 48-point swing. MRF measures drift *inside* the
> trace; the effect lives in *where the estimate lands*.

Full report: **[`REPORT.md`](REPORT.md)**.

---

## The four findings

**1. The bias rides on the sentences that pick numbers, not the ones that discuss the bet.**
Replace a sentence with an alternative the model itself would have written, let it redo
everything downstream, and see whether the answer changes. Only units choosing a Fermi input
move it — signed effect **+0.039 [+0.004, +0.065]** at the paper's R=100, against ~0 for units
that name the bet or compare against the threshold, measured on the same traces at the same
sample size. ([§4.3](REPORT.md))

**2. Stated commitments to honesty are causally inert.** Qwen3.5 asserts honesty ~5 times per
trace under incentive. Across 72 such units against 64 matched neighbouring controls they move
the answer no more than the ordinary sentences beside them (0.040 vs 0.047, both at or below a
0.054 noise floor). The sentences are unambiguous — *"I will not bias the biological data to
influence the charity donation type."* — and changing them changes nothing.

A second, independent line agrees. `gpt-oss-safeguard-20b` is a safety fine-tune of gpt-oss:
same base weights, same format, and — measured here — the same estimation behaviour, down to an
identical median baseline estimate across 200 rollouts. It produces roughly **half** the
honesty-talk, and its bias gap is statistically indistinguishable. Halving the narration changed
nothing measurable, reached by comparing checkpoints rather than intervening on sentences.
([§5](REPORT.md))

**3. No pre-reasoning tilt, but large direction-blind anchoring.** Forced to answer with no
reasoning, both models give the same answers under both incentives. But merely stating a
threshold moves the estimate toward it from whichever side the model starts on — gpt-oss rises
3.3M → 24–28M (threshold 47.5M), Qwen3.5 falls from nonsense magnitudes to ~65M (threshold
25.2M). Anchoring is not "estimates get bigger"; it is "estimates move toward the number in the
context", which is what makes it a confound for a threshold-crossing design. ([§4.1](REPORT.md))

**4. The 10-model MRF leaderboard is mostly unsupported.** Only two of ten runs have an effect
clearly distinguishable from zero; two more are borderline; the remaining six have intervals
that all contain zero and all overlap each other. More rollouts would not fix the ordering —
each score is measured against its own model's spread, and those spreads differ by 36×, so the
numbers are not on a common scale. ([§3.3](REPORT.md))

### Is this unfaithful chain of thought?

Yes, but not the hidden-cue kind. Both models name the bet in nearly every trace. What the trace
misreports is the *mechanism*: the sentences doing the causal work read as ordinary estimation,
and the sentences that visibly engage the incentive do nothing. A monitor looking for a model
*talking about* an incentive would flag the wrong sentences, and would clear a trace whose
visible reasoning is impeccable while its parameter choices carry the bias.

---

## Method findings worth reusing

These are about the instrument rather than the models, and they are the most transferable part
of the work ([§6](REPORT.md)).

- **Counterfactual importance has a horizon limit, and the variable is *remaining* reasoning —
  not trace length.** Measured on 1,328 already-resampled positions: the same-meaning placebo
  separation runs **+0.005** when under 0.5k characters remain after the cut, to **−0.058** when
  over 20k do. The confound-free test is *within* one configuration, where model, effort and
  sample size are fixed and only position varies — and both long-trace runs degrade inside
  themselves (gpt-oss @ high **+0.061 [+0.001, +0.104]** between its own late and early halves;
  Qwen3.5 **+0.043 [+0.001, +0.080]**).

  **The practical consequence is specific: place interventions late.** Late cuts in a long trace
  recover most of the method's resolving power at no extra cost. This is the main caution for
  anyone applying the technique to long agentic trajectories, which sit squarely inside the
  limit.

- **The regime is bounded at both ends.** Very short traces fail too, for the opposite reason:
  when a trace has three or four units, replacing one rewrites most of the reasoning, so even a
  same-meaning resample produces a different derivation. The method needs traces long enough
  that one unit is a small part of the whole, and cuts late enough that little remains to
  re-randomise the outcome.

- **Which arm moves tells you whether you are measuring signal or noise.** As horizons lengthen
  it is the *same-meaning* placebo that grows (0.121 → 0.176), not the different-meaning arm
  (0.126 → 0.118). If early sentences genuinely mattered more, the treatment arm should have
  pulled ahead. It does not — so what grows is downstream sampling re-randomising the outcome.

- **The signed metric survives that failure where an unsigned one would not.** Symmetric
  perturbation noise cancels under signing, and misclassification can only dilute a treatment
  arm toward the base, never manufacture an effect.

- **Sampling noise dominates below R≈32.** An early pass at R=16 produced a treatment effect of
  0.149 against a same-meaning placebo of 0.144 — noise measuring itself. The noise floor is
  measured by split-half of the base arm rather than assumed.

- **Reasoning effort is a one-word system-prompt control with 25× consequences.** For gpt-oss,
  `Reasoning: low|medium|high` changes exactly one line and moves median trace length 260 →
  1,050 → 6,560 tokens. Measured mechanism: at a trace's natural stopping point P(end-of-reasoning
  token) is 0.998 under `low` and 0.675 under `high`. Qwen3.5 has no graded equivalent —
  `reasoning_effort` is silently ignored. **The silent no-op is a trap**: a run can be labelled
  effort-controlled while nothing changed, so the sampling code raises rather than accepting the
  argument for that family.

- **Validate parsers against an independent reader, not against a parse rate.** Checking the
  regex against the estimate judge on 200 answers found **two order-of-magnitude bugs** a 97.5%
  parse *rate* had concealed, both reading low, both now fixed (agreement 96.5% → 98.0%). One was
  introduced by the fix for an earlier parser bug.

- **100 rollouts per condition is not enough to compare models here.** Across four
  gpt-oss-family runs every measured bias gap had a confidence interval spanning zero. The design
  resolved a 48-point swing and nothing near a 10-point one — the same conclusion the shipped
  10-model leaderboard forces, from the other direction.

---

## Provenance

- Task, prompts and judge design come from
  **[adsingh-64/value-leakage](https://github.com/adsingh-64/value-leakage)**, a minimal
  reproduction of the original study, which also ships raw rollouts for 10 models. All analysis
  of those shipped runs here is re-analysis of that data.
- That reproduction derives from **TruthfulAI-research/value_leakage** (Owain Evans' group).
  Judge prompts are used byte-for-byte, typos included.
- The sentence-level causal method follows **Thought Anchors** (Bogdan, Macar, Nanda & Conmy,
  [arXiv:2506.19143](https://arxiv.org/abs/2506.19143)), with three adaptations described below.

Two defects found in the reproduction while working with it, reported in full in
[`REPORT.md`](REPORT.md) §3.2. They are noted here because they affect anyone reusing that
pipeline, not as criticism of a small reproduction:

- `estimates.json` contains only `baseline` in all 10 shipped runs. The estimate judge runs
  immediately after baseline sampling, before the incentive conditions exist, and the judge
  helper silently skips conditions whose file is missing. Every treatment-condition number in
  that repo therefore comes from the trajectory judge reading the CoT.
- `plot.drift()` passes `outlier_factor=None` while `curve()` applies the `[thr/10, thr*10]`
  filter, so MRF and the plotted curves use different subsets, and `n_kept` printed beside MRF
  comes from the filtered path.

Whether the upstream study shares either is untested — only the reproduction was used.

---

## What was run

| | Track A | Track B |
|---|---|---|
| model | `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4` | `openai/gpt-oss-20b` |
| role | mainstream open-weight reference; same family and MoE architecture as a shipped run | independent family and post-training lineage |
| rollouts | 3 conditions × 100, 0 truncated, 0 parse failures | 3 × 100, 0 truncated, 0 parse failures |
| threshold (median baseline estimate) | 25,178,000 | 47,500,000 |
| median reasoning per rollout | 25,151–32,347 chars | 2,324–3,036 chars |

Three further configurations separate trace length from model identity: **gpt-oss at low and
high reasoning effort**, and **gpt-oss-safeguard-20b** at medium and high. These carry the
method findings above.

Both models were run locally under vLLM, so **the chain of thought is raw**, not a provider-side
summary — a prerequisite for resampling, and the reason the shipped `claude-opus-4-7` run cannot
be used for it.

## Method

- **Forced-answer screen** — truncate at a unit boundary, force an immediate answer, read
  P(estimate > threshold). Cheap; used to map where the answer becomes determined and to nominate
  positions. Never the verdict: Thought Anchors documents that a late necessary sentence
  suppresses every earlier position's score.
- **Counterfactual importance** — resample the unit from the model itself, split resamples by
  meaning, regenerate the whole remainder, compare answer distributions. Not deletion, which
  leaves an off-distribution hole and confounds content with incoherence.

Three adaptations the task forces:

1. **Binary outcome** `Y = 1[estimate > threshold]`, so effects are in probability units.
2. **Signed importance** `A_i = d(c)·Δp`, `d = +1` for `above_good` and `−1` for `below_good`.
   Thought Anchors' metric is unsigned — it asks which sentences matter. Here the question is
   which sentences matter *toward the incentive*, which only the mirror-condition design makes
   answerable.
3. **Numeric-aware meaning split.** Sentence embedders rate *"spots per individual: 2,200"*
   against *"…: 1,200"* at ~0.97 cosine, though it halves the answer. Numeric units split on the
   committed value; others on MiniLM cosine. The split is a function of the resampled unit alone,
   never of the outcome.

---

## Layout

```
analysis/
  local_gen.py            sampling with per-family CoT splitting (raw, not summarised)
  segment.py              units + char offsets + tags; aborts if offsets do not round-trip
  screen.py               forced-answer screen across truncation positions
  resample.py             counterfactual importance; regenerates everything downstream
  disavowal_test.py       disavowal units vs matched neighbouring controls
  forensics.py            MRF with bootstrap CIs and label-permutation tests, all runs
  fingerprints.py         early-commitment vs sustained-drift signature per run
  horizon_curve.py        separation as a function of reasoning remaining after the cut
  validate_parser.py      regex answer parser vs the independent estimate judge
  retry_nulls.py          judge-null repair, separating failed samples from parse failures
  effort_experiment.py    reasoning effort sweep
  effort_mechanism.py     P(stop) under low vs high at matched context
  positions_*.json        frozen position plans; a-priori and exploratory ledgers separate
  pre_retry/              as-shipped metrics before the judge repair
runs/<model>_<stamp>/     rollouts, judge outputs, screens, counterfactual results
REPORT.md                 full report: findings, method validity, limitations, corrections
```

## Reproducing

Needs a GPU (48GB for the Qwen Int4 model, 24GB for gpt-oss) and an `ANTHROPIC_API_KEY` for the
judges.

```bash
conda create -y -n vl python=3.12 && conda activate vl
pip install "numpy<2.4" vllm sentence-transformers fire matplotlib scipy pandas \
            anthropic python-dotenv
pip install -e .

# sample, judge, screen, resample
python analysis/local_gen.py --model <hf-id> --family qwen --condition baseline \
    --out runs/<name>/baseline.json --count 100
python analysis/judge_and_threshold.py --run_dir runs/<name> --kind estimates
python analysis/screen.py    --model <hf-id> --family qwen --run_dir runs/<name>
python analysis/resample.py  --model <hf-id> --family qwen --run_dir runs/<name> \
    --positions_file analysis/positions_A.json --R 32
```

Two environment notes that cost hours to find: set `VLLM_USE_FLASHINFER_SAMPLER=0` (its sampling
kernel JIT-compiles against the system CUDA and fails on both machines used here), and keep every
vLLM entry point behind `if __name__ == "__main__":` — the v1 engine spawns a subprocess that
re-imports the module.

## Limitations

Stated in full in [`REPORT.md`](REPORT.md) §8, with every correction and retraction collected in
its Appendix A. The main ones: statistical power below the paper's density on Track A; Track A's
resampled traces selected as cheapest-to-continue, biasing toward terser traces; Bernoulli KL
unreliable at these sample sizes (|Δp| carries the argument); Track A quantised to 4-bit; and the
resampling runs carry a ~3% parse-error rate that can be removed only by re-running them, since
they stored thresholded outcomes rather than completion text.
