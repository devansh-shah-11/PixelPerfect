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

Setup and submission are in the header of [`hpc/vlm_latency.sbatch`](hpc/vlm_latency.sbatch) (`pip install -e ".[dev,vlm]"`, weights under `HF_HOME` on scratch).

### VLM latency (M2.1)

Not measured yet. The table below is filled from `results/vlm_latency_<jobid>.json` once `hpc/vlm_latency.sbatch` has run on the A100; the script prints it ready to paste.

| Model | dtype | n | median (s) | p95 (s) | p99 (s) | load (s) | peak GPU (GB) | outcome |
|---|---|---|---|---|---|---|---|---|
| Qwen/Qwen3-VL-2B-Instruct | bfloat16 | | pending | pending | | | | not run |
| Qwen/Qwen3-VL-8B-Instruct | bfloat16 | | pending | pending | | | | not run |
