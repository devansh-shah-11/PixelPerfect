# PixelPerfect

Gated fast/slow perception for a simulated drone: fast perception (Depth Anything V2 + SAM 2 + rule controller) flies every frame, and a small learned gate asks a local VLM (Qwen3-VL) a yes/no question only when it predicts the fast perception is wrong.

Headline result: crash rate vs. VLM calls per episode for never / always / random / hand-tuned threshold / learned gate / oracle. Simulation only.

- Plan, schedule, metrics, blockers and fallbacks: [`PLAN.md`](PLAN.md)
- Working rules for the AI assistant: [`CLAUDE.md`](CLAUDE.md)
- Agent Log (AI actions and mistakes, per member): [`AGENT_LOG.md`](AGENT_LOG.md); progress log: `logs/PROGRESS.md`

Status: Sprint 1 (Thu Oct 29 to Wed Nov 11, PLAN.md §13). Repo skeleton and shared interfaces in place; no simulator yet.

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

Shared types between workstreams live in [`interfaces.py`](interfaces.py) (PLAN.md §6).
