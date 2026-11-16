# PixelPerfect

Gated fast/slow perception for a simulated drone: fast perception (Depth Anything V2-Small metric + rule controller; SAM 2 is an ablation only, D7) flies every frame, and a small learned gate asks a local VLM (Qwen3-VL) a yes/no question only when it predicts the fast perception is wrong.

Headline result: crash rate vs. VLM calls per episode for never / always / random / hand-tuned threshold / learned gate / oracle. Simulation only.

- Start here, one page: [`PLAN.md`](PLAN.md) (the idea, where we are, the milestone map, who owns what)
- What to do next: [`TASKS.md`](TASKS.md) (milestones with gates, ordered by dependency)
- Source of truth for decisions, metrics, blockers and fallbacks: [`PLAN_DETAILED.md`](PLAN_DETAILED.md)
- Working rules for the AI assistant: [`CLAUDE.md`](CLAUDE.md)
- Agent Log (AI actions and mistakes, per member): [`AGENT_LOG.md`](AGENT_LOG.md); progress log: `logs/PROGRESS.md`

Status: see [`PLAN.md`](PLAN.md) §3. Project AirSim runs headless on the A100 partition and a CityEnviron frame bank exists (411 episodes, 19,267 frames); the closed loop does not exist yet, so there are no crash rates.

## Setup (laptop)

```bash
uv venv --python 3.10 .venv        # or any Python >= 3.10 environment
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest         # interface tests
```

## Cloud Bursting smoke test (task S-06)

Run once per person, then paste the output into the team chat:

```bash
sbatch hpc/job_template.sbatch     # runs hpc/smoke_test.sh; output in slurm-pp-<jobid>.out
```

Shared types between workstreams live in [`interfaces.py`](interfaces.py) (PLAN_DETAILED.md §6).

## VLM judge (workstream C, `vlm/`)

- `vlm/judge.py`: Qwen3-VL behind `interfaces.VLMJudge`. One forward pass per question; the answer is `logit(yes) - logit(no)`.
- `vlm/overlay.py`: the highlighted region (the corridor window the labels use) drawn on the frame, plus a full-resolution crop of it.
- `vlm/surrogate.py`: the surrogate judge (noisy oracle with measured error rates). Every result from it is labelled "surrogate VLM".
- `vlm/bench_latency.py`: latency benchmark (chunk M2.1).
- `vlm/eval_bank.py`: standalone accuracy on labelled frame-bank frames and the M2 gate verdict (chunk M2.4). Writes `<out>.jsonl` (one row per model, condition and frame), `<out>.summary.json`, `<out>.report.md`, and eyeball sheets. `--summarize-only` re-scores saved rows without a GPU.

Setup and submission are in the headers of [`hpc/vlm_latency.sbatch`](hpc/vlm_latency.sbatch) and [`hpc/vlm_eval_bank.sbatch`](hpc/vlm_eval_bank.sbatch) (`pip install -e ".[dev,vlm]"`, weights under `HF_HOME` on scratch).

### VLM accuracy (M2.4), first look

Job 33462, 2B, overlay + crop, D9 question, 500 val frames (50 per obstacle type): recall 0.40 (95% interval over flights 0.27 to 0.54), precision 0.90, false-positive rate 0.05, AUROC 0.70. On the visible types the gate counts, recall 0.28 (0.12 to 0.48) over 15 flights, below the 0.60 gate. Not the final table: glass had no ground truth in this run, and val alone gives only 2 to 5 flights per type. The full run (train + val flights, 2B and 8B, six conditions) replaces it.

### VLM latency (M2.1)

Measured by `hpc/vlm_latency.sbatch`, Cloud Bursting job 33341, 2026-11-15, one A100 40 GB, commit `f29104c`. One question = one forward pass over two images (225 x 400 frame with the corridor overlay, plus the 356 x 320 crop), single-token yes/no read from the logits, no `generate()`. Frames were random noise (`--frames` not given); the number to beat on real frames is the same size because the token count does not depend on content. Full per-call data: `results/vlm_latency_33341.json` on the HPC.

| Model | dtype | n | median (s) | p95 (s) | p99 (s) | load (s) | peak GPU (GB) | outcome |
|---|---|---|---|---|---|---|---|---|
| Qwen/Qwen3-VL-2B-Instruct | bfloat16 | 50 | 0.073 | 0.076 | 0.086 | 25.5 | 4.37 | success |
| Qwen/Qwen3-VL-8B-Instruct | bfloat16 | 50 | 0.089 | 0.091 | 0.092 | 97.1 | 17.67 | success |

Same job on 100 real CityEnviron bank frames (job 33400, seeded random sample across all conditions, no labels): latency unchanged (2B 0.076 s median, 8B 0.091 s). Yes-fraction 0.22 for the 2B (confidence median -0.6) and 0.05 for the 8B (median -4.5), against 0.00 for both on noise. So the judge reacts to content and the 8B is far more conservative; which one is right is what the M2.4 table measures.

Reading: on a data-centre GPU with a one-token answer, the real latency is under 0.1 s for both sizes, and the 8B costs only 16 ms more than the 2B, so the vision encoder and fixed overhead dominate, not the language model. The seconds-long delays the project is designed around (`PLAN_DETAILED.md` §1) correspond to onboard or shared hardware, not to this setup, which is why the delay `d` is a controlled, swept variable (D10) and not this measurement. Both models fit the 40 GB A100 at once (about 22 GB together).
