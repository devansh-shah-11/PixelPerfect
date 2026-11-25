# CLAUDE.md — Gated fast/slow perception for a simulated drone

Read `PLAN.md` for orientation (one page: the idea, where we are, the milestone map), then `PLAN_DETAILED.md`, which is the source of truth for scope. If this file and `PLAN_DETAILED.md` disagree, `PLAN_DETAILED.md` wins: follow it, tell the user, and log the mismatch in `AGENT_LOG.md`. `PLAN.md` holds no decisions of its own; it only points at `PLAN_DETAILED.md`. Task status lives in `TASKS.md` (milestones and gates); `TASKS_DETAILED.md` is the old dated board, kept only for its task ids, handoffs and progress log.

## Project in one paragraph
A simulated drone flies on a fast stack (Depth Anything V2-Small, metric checkpoint, + rule controller; SAM 2 is an ablation only, `PLAN_DETAILED.md` D7). A small learned **gate** predicts when that perception is wrong (labels from simulator ground truth). Only when the gate fires do we ask **Qwen3-VL-2B** (8B as a size reference; `PLAN_DETAILED.md` D8) one yes/no question about the drawn flight path. The slow system is swappable (`PLAN_DETAILED.md` D18); the gate is the contribution. Deliverable: crash rate vs. VLM calls/flight for never / always / random / hand-tuned threshold / learned gate / oracle (plus the true-depth reference), crash-cause analysis, Gradio replay and a report. Simulation only.

## Environments (do not mix them up)
- **NYU HPC Cloud Bursting (Slurm), the only cluster we use** (`PLAN_DETAILED.md` D17, §8). **Never use the Torch cluster** (course rule). Slurm account `csci_ga_2271-2026fa`; Open OnDemand at `https://ood.burst.hpc.nyu.edu/` (NYU VPN off-campus). 300 GPU-hours per student. No sudo, no display. Work in scratch (`/scratch/<NetID>`), not home. Cache models via `HF_HOME` on scratch. Partitions: `c12m85-a100-1` (1 × A100, the one that works for us), `c24m170-a100-2` (2 × A100), `g2-standard-12` (1 × L4), `g2-standard-24` (2 × L4), `n2c48m24` (CPU), `interactive`; see `hpc/instructions.md`. Batch jobs carry `#SBATCH --requeue`.
  - Simulator (offscreen) + fast stack + VLM run **in one job**. `PLAN_DETAILED.md` D17 names the L4 for this, but L4 nodes failed to boot for two members (jobs 15186, 15418), so we use the **A100 (40 GB)** until an L4 job runs. The simulator is Project AirSim v1.0.1, run natively (deviation from `PLAN_DETAILED.md` D3, needs team sign-off).
  - **20-minute idle shutdown**: nodes with low GPU use for 20 minutes are terminated. Never leave a GPU session idle, never run the VLM as a separate mostly-idle server, and checkpoint every long job.
- **GCP GPU VM**: fallback only, if Cloud Bursting can't render the simulator and credits exist (Q2). **Plan B simulator** (`PLAN_DETAILED.md` §11) if neither works.
- **Laptop (macOS)**: editing, reading and debugging, CPU-only work (thin-structure filters, gradient boosting), Gradio demo, Plan B simulator, plotting. No GPU assumptions.
- Before running anything, state which environment the command is for. Never launch HPC jobs, start/stop cloud VMs, or spend credits without asking.

## Working rules
0. **Never modify `PLAN_DETAILED.md` unless the user explicitly asks.** It is read-only for the assistant. If the plan seems wrong or outdated, say so in chat or record it in `AGENT_LOG.md`; do not edit the file. Deviations from the plan go in the log, not in the plan.
1. **Plan first for anything non-trivial.** Small, reversible steps. Prefer the simplest thing that meets the milestone in `PLAN_DETAILED.md`.
2. **No scope creep.** Items marked Parked or Backlog in `PLAN_DETAILED.md` Appendix A stay out unless the user says otherwise; pulling one in needs a Decision-log row (`PLAN_DETAILED.md` §0).
3. **Determinism.** Every episode is keyed by `(split, seed)`. Seed lists live in `configs/seeds.yaml` and are frozen once task A-06 creates them (`PLAN_DETAILED.md` D13). Never edit them afterwards. Never tune anything on test seeds.
4. **Split by seed, never by frame.** Any code that splits data must be checked for leakage.
5. **No ground truth at runtime.** Controller, gate and policies must not read GT depth/segmentation/collision. GT is for labels and metrics only. Flag any code path that violates this.
6. **Do not fabricate results.** Never invent numbers, plots, or citations. If a run failed or was skipped, say so with the output. Results in the report must trace to logged runs.
7. **Surrogate VLM results must be labeled** "surrogate VLM" everywhere (tables, plots, file names).
8. **Reuse before writing.** Check the `PLAN_DETAILED.md` §7 layout and existing modules first. All policies go behind the single interface in `policies/`.
9. **Match surrounding code style.** Python 3.10+, type hints on public functions, config via YAML in `configs/`, no hard-coded paths.
10. **Long jobs must be resumable**: append per-episode results to JSONL/CSV and skip finished `(policy, seed)` pairs on restart. This is required, not optional: Cloud Bursting kills idle nodes after 20 minutes.

## Git workflow
- Repo: `github.com/devansh-shah-11/PixelPerfect`, default branch `main`.
- **Commit or push only when the user asks.** Never force-push, never rewrite history, never commit data, weights, or secrets (`data/`, `*.pt`, `*.ckpt`, `.env`, credentials are gitignored).
- Branch per chunk: `<workstream>/<chunk-id>` (e.g. `C/M2.1`, per `TASKS.md`), merged into `main` at least weekly; `main` should always run.
- Commit messages: imperative, one line summary, a body saying *why*. Prefix with the area: `sim:`, `perception:`, `gate:`, `vlm:`, `eval:`, `demo:`, `docs:`.

### After every commit (do all of these, then report what you did)
1. **Append to `logs/PROGRESS.md`**: date/time, commit hash, one-line summary, which `TASKS.md` chunk it advances, and what is now unblocked or still blocked.
2. **Check `AGENT_LOG.md` and update it** (see below): read the latest entries first. If there is already an entry from today for the same task, update that entry with the key points from this commit. If the commit is a different task (for example VLM work after simulator work), add a new entry. Do not rely on the user remembering to ask, and do not write one entry per commit.
3. **Run a quick sanity check relevant to the change** (unit tests if they exist, or the smoke command for that area) and record pass/fail. Do not hide failures.
4. **Check the working tree is clean** and nothing large or secret was staged (`git status`, `git diff --cached --stat` before the commit; `git show --stat` after).
5. **Backup reminder**: if the commit touched `configs/`, `gate/` labels, or scenario generation, remind the user to `rsync` data/logs to the second location (VM ↔ HPC scratch ↔ cloud storage) per `PLAN_DETAILED.md` §11 "Insurance to create now". Run the sync only if asked.
6. **Milestone tags**: if this commit completes the MVP milestone (all MVP policies run end to end and the first curve exists; `TASKS_DETAILED.md` S-08), suggest the tag `mvp-v1` and ask before tagging. Later milestone tags as the team agrees.
7. **Push only if the user asked** to push. Otherwise say it has not been pushed.

## Monitoring and logging AI behaviour
Goal: a transparent, auditable record of what the assistant did, assumed, got wrong, and why, so the report can describe the methodology honestly and mistakes are caught early.

Maintain **`AGENT_LOG.md`** in the repo root (append-only, newest at the bottom). It feeds the Agent Log appendix of the report (`PLAN_DETAILED.md` §12), so every entry names the member whose session it was. **One entry per task per day** (not per commit, not per chat message). Commits and chats on the same task and the same day go into one entry, updated in place; a different task (different area or chunk) gets its own entry, even in the same session. Summarize the key points: what was asked, what was done, decisions and deviations, failures and corrections, what is still open. Use this format, with `Action` and `Decisions` holding several short bullets when an entry covers several things:

```
## YYYY-MM-DD HH:MM  |  <area>  |  <short title>
- Member: initials of the teammate whose session this is
- Request: what the user asked (quote or paraphrase)
- Action: what was done (files touched, commands run, environment: VM/HPC/laptop)
- Assumptions: anything guessed rather than known; mark each as [UNVERIFIED] until confirmed
- Decisions & alternatives: what was chosen, what was rejected, why
- Evidence: command output, metric, seed, run id, or file path backing the claim
- Outcome: success / partial / failed (with the error text if short)
- Follow-ups: open questions or checks needed
```

Rules for the log:
- **Log**: design decisions (label definition, features, thresholds, latency model), any deviation from `PLAN_DETAILED.md`, every failure or retry, any fallback from `PLAN_DETAILED.md` §11 that was triggered (name the blocker rank, e.g. "rank 4: thin wires"), any result that surprised us, and any time the assistant was corrected by the user.
- **Mark uncertainty explicitly.** Do not present guesses, recalled library APIs, or memory of paper numbers as verified. Mark them `[UNVERIFIED]` and verify (run it, read the source, or look it up) before relying on them.
- **Flag risky behaviour in the entry** (and tell the user immediately): deleting or overwriting files, long or costly jobs, changes to frozen configs, anything that could cause test-set leakage, anything touching credentials.
- **Cite sources**: research claims carry arXiv ids (as in `PLAN_DETAILED.md`); dataset/model versions carry exact names and checkpoints.
- **Reproducibility fields** on every experiment entry: git commit hash, config file(s), seed list, model checkpoint names, environment.
- **Weekly self-review (at each Sunday walkthrough, `PLAN_DETAILED.md` §12)**: summarize from `AGENT_LOG.md` the assumptions that turned out wrong, repeated failures, and anything unverified that still influences results. Put the summary at the top of the next report draft.

Also keep **`logs/runs.jsonl`**: one line per experiment run with `{time, commit, env, command, config, seeds, n_episodes, wall_seconds, outcome}`. The eval scripts should append to it automatically; if they don't yet, add that first.

## Escalation: stop and ask the user when
- A kill switch from `PLAN_DETAILED.md` §11 is hit — recommend the fallback, don't silently switch.
- A task would exceed ~1 hour of compute, use paid resources, or need the user's credentials.
- Results look too good (e.g. gate AUROC > 0.99, crash rate 0%) — suspect label leakage or GT use at runtime before celebrating.
- Requirements are ambiguous in a way that changes the experiment (label definition, latency model, metric definitions).
- Open questions from `PLAN_DETAILED.md` §16 become blocking.

## Reporting style
Be concise and factual. State what was run, where, and the result. Say plainly when something failed, was skipped, or is unverified. Give a recommendation rather than a menu of options. Reference code as `path:line`.

## Quick commands (fill in as they come to exist)
- Run one episode: `python eval/run_episode.py --policy never --seed 0`
- Sweep: `python eval/sweep.py --config configs/sweep.yaml`
- Metrics/plots: `python eval/plots.py`
- Demo: `python demo/app.py` (reads logs only; no simulator needed)
- Tests: `.venv/bin/python -m pytest` (laptop)
- Frame-bank split (once; the file is frozen): `python -m perception.bank_split --bank <bank dir> --out configs/bank_split.yaml`
- Fast stack over the frame bank (resumable): `python -m perception.run_bank --bank <bank dir> --out <file>.jsonl`
- First look at a bank run (train and val only): `python -m perception.analyze_bank --run <file>.jsonl [--bank <bank dir> --sheet <file>.png]`
- Clean ground-truth corridor distances (CPU only): `python -m gate.bank_gt --bank <bank dir> --out <file>.jsonl`, then pass it to `analyze_bank --gt`
- Benchmark table, depth / VLM / depth + VLM (CPU only, reads result files): `python -m eval.benchmark --vlm results/vlm_eval.jsonl --run <file>.jsonl --latency results/vlm_latency_*.json [--flights <sweep dirs>]`
- Sweep with overrides (one config, many variants): `python -m eval.sweep --config <file>.yaml --set judge_kw.model_id=Qwen/Qwen3-VL-8B-Instruct@4bit --set out=<dir>`
- Seed lists: `configs/seeds.yaml` is frozen (1000 / 500 / 500, D24). Scenario settings freeze once, after M4.7: `python -m sim.scenario --freeze-scenario <sweep>.yaml --to configs/scenario.yaml`
- HPC job template: `sbatch hpc/job_template.sbatch` (default: smoke test; other commands via `--export=ALL,CMD="..."`)
