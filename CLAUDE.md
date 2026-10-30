# CLAUDE.md — Gated fast/slow perception for a simulated drone

Read `PLAN.md` first. It is the source of truth for scope. If this file and `PLAN.md` disagree, `PLAN.md` wins: follow it, tell the user, and log the mismatch in `AGENT_LOG.md`. Task status lives in `TASKS.md`.

## Project in one paragraph
A simulated drone flies on a fast stack (Depth Anything V2-Small, metric checkpoint, + rule controller; SAM 2 is an ablation only, `PLAN.md` D7). A small learned **gate** predicts when that perception is wrong (labels from simulator ground truth). Only when the gate fires do we ask **Qwen3-VL-2B** (8B as a size reference; `PLAN.md` D8) one yes/no question about the drawn flight path. The slow system is swappable (`PLAN.md` D18); the gate is the contribution. Deliverable: crash rate vs. VLM calls/flight for never / always / random / hand-tuned threshold / learned gate / oracle (plus the true-depth reference), crash-cause analysis, Gradio replay and a report. Simulation only.

## Environments (do not mix them up)
- **NYU HPC Cloud Bursting (Slurm), the only cluster we use** (`PLAN.md` D17, §8). **Never use the Torch cluster** (course rule). Slurm account `csci_ga_2271-2026fa`; Open OnDemand at `https://ood.burst.hpc.nyu.edu/` (NYU VPN off-campus). 300 GPU-hours per student. No sudo, no display. Work in scratch, not home. Cache models via `HF_HOME` on scratch.
  - **L4**: simulator (Singularity, offscreen) + fast stack + VLM **in one job**. **A100 (40 GB)**: batch VLM evaluation and the 8B model.
  - **20-minute idle shutdown**: nodes with low GPU use for 20 minutes are terminated. Never leave a GPU session idle, never run the VLM as a separate mostly-idle server, and checkpoint every long job.
- **GCP GPU VM**: fallback only, if Cloud Bursting can't render the simulator and credits exist (Q2). **Plan B simulator** (`PLAN.md` §11) if neither works.
- **Laptop (macOS)**: editing, reading and debugging, CPU-only work (thin-structure filters, gradient boosting), Gradio demo, Plan B simulator, plotting. No GPU assumptions.
- Before running anything, state which environment the command is for. Never launch HPC jobs, start/stop cloud VMs, or spend credits without asking.

## Working rules
0. **Never modify `PLAN.md` unless the user explicitly asks.** It is read-only for the assistant. If the plan seems wrong or outdated, say so in chat or record it in `AGENT_LOG.md`; do not edit the file. Deviations from the plan go in the log, not in the plan.
1. **Plan first for anything non-trivial.** Small, reversible steps. Prefer the simplest thing that meets the milestone in `PLAN.md`.
2. **No scope creep.** Items marked Parked or Backlog in `PLAN.md` Appendix A stay out unless the user says otherwise; pulling one in needs a Decision-log row (`PLAN.md` §0).
3. **Determinism.** Every episode is keyed by `(split, seed)`. Seed lists live in `configs/seeds.yaml` and are frozen once task A-06 creates them (`PLAN.md` D13). Never edit them afterwards. Never tune anything on test seeds.
4. **Split by seed, never by frame.** Any code that splits data must be checked for leakage.
5. **No ground truth at runtime.** Controller, gate and policies must not read GT depth/segmentation/collision. GT is for labels and metrics only. Flag any code path that violates this.
6. **Do not fabricate results.** Never invent numbers, plots, or citations. If a run failed or was skipped, say so with the output. Results in the report must trace to logged runs.
7. **Surrogate VLM results must be labeled** "surrogate VLM" everywhere (tables, plots, file names).
8. **Reuse before writing.** Check the `PLAN.md` §7 layout and existing modules first. All policies go behind the single interface in `policies/`.
9. **Match surrounding code style.** Python 3.10+, type hints on public functions, config via YAML in `configs/`, no hard-coded paths.
10. **Long jobs must be resumable**: append per-episode results to JSONL/CSV and skip finished `(policy, seed)` pairs on restart. This is required, not optional: Cloud Bursting kills idle nodes after 20 minutes.

## Git workflow
- Repo: `github.com/devansh-shah-11/PixelPerfect`, default branch `main`.
- **Commit or push only when the user asks.** Never force-push, never rewrite history, never commit data, weights, or secrets (`data/`, `*.pt`, `*.ckpt`, `.env`, credentials are gitignored).
- Branch per task: `<workstream>/<task-id>` (e.g. `A/A-03`, per `TASKS.md`), merged into `main` at least weekly; `main` should always run.
- Commit messages: imperative, one line summary, a body saying *why*. Prefix with the area: `sim:`, `perception:`, `gate:`, `vlm:`, `eval:`, `demo:`, `docs:`.

### After every commit (do all of these, then report what you did)
1. **Append to `logs/PROGRESS.md`**: date/time, commit hash, one-line summary, which `PLAN.md` §13 step or `TASKS.md` task it advances, and what is now unblocked or still blocked.
2. **Append to `AGENT_LOG.md`** (see below) any decisions, assumptions, or mistakes from the work in this commit.
3. **Run a quick sanity check relevant to the change** (unit tests if they exist, or the smoke command for that area) and record pass/fail. Do not hide failures.
4. **Check the working tree is clean** and nothing large or secret was staged (`git status`, `git diff --cached --stat` before the commit; `git show --stat` after).
5. **Backup reminder**: if the commit touched `configs/`, `gate/` labels, or scenario generation, remind the user to `rsync` data/logs to the second location (VM ↔ HPC scratch ↔ cloud storage) per `PLAN.md` §11 "Insurance to create now". Run the sync only if asked.
6. **Milestone tags**: if this commit completes the MVP milestone (all MVP policies run end to end and the first curve exists; `TASKS.md` S-08), suggest the tag `mvp-v1` and ask before tagging. Later milestone tags as the team agrees.
7. **Push only if the user asked** to push. Otherwise say it has not been pushed.

## Monitoring and logging AI behaviour
Goal: a transparent, auditable record of what the assistant did, assumed, got wrong, and why, so the report can describe the methodology honestly and mistakes are caught early.

Maintain **`AGENT_LOG.md`** in the repo root (append-only, newest at the bottom). It feeds the Agent Log appendix of the report (`PLAN.md` §12), so every entry names the member whose session it was. One entry per meaningful action or decision:

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
- **Log**: design decisions (label definition, features, thresholds, latency model), any deviation from `PLAN.md`, every failure or retry, any fallback from `PLAN.md` §11 that was triggered (name the blocker rank, e.g. "rank 4: thin wires"), any result that surprised us, and any time the assistant was corrected by the user.
- **Mark uncertainty explicitly.** Do not present guesses, recalled library APIs, or memory of paper numbers as verified. Mark them `[UNVERIFIED]` and verify (run it, read the source, or look it up) before relying on them.
- **Flag risky behaviour in the entry** (and tell the user immediately): deleting or overwriting files, long or costly jobs, changes to frozen configs, anything that could cause test-set leakage, anything touching credentials.
- **Cite sources**: research claims carry arXiv ids (as in `PLAN.md`); dataset/model versions carry exact names and checkpoints.
- **Reproducibility fields** on every experiment entry: git commit hash, config file(s), seed list, model checkpoint names, environment.
- **Weekly self-review (at each Sunday walkthrough, `PLAN.md` §12)**: summarize from `AGENT_LOG.md` the assumptions that turned out wrong, repeated failures, and anything unverified that still influences results. Put the summary at the top of the next report draft.

Also keep **`logs/runs.jsonl`**: one line per experiment run with `{time, commit, env, command, config, seeds, n_episodes, wall_seconds, outcome}`. The eval scripts should append to it automatically; if they don't yet, add that first.

## Escalation: stop and ask the user when
- A kill switch from `PLAN.md` §11 is hit — recommend the fallback, don't silently switch.
- A task would exceed ~1 hour of compute, use paid resources, or need the user's credentials.
- Results look too good (e.g. gate AUROC > 0.99, crash rate 0%) — suspect label leakage or GT use at runtime before celebrating.
- Requirements are ambiguous in a way that changes the experiment (label definition, latency model, metric definitions).
- Open questions from `PLAN.md` §16 become blocking.

## Reporting style
Be concise and factual. State what was run, where, and the result. Say plainly when something failed, was skipped, or is unverified. Give a recommendation rather than a menu of options. Reference code as `path:line`.

## Quick commands (fill in as they come to exist)
- Run one episode: `python eval/run_episode.py --policy never --seed 0`
- Sweep: `python eval/sweep.py --config configs/sweep.yaml`
- Metrics/plots: `python eval/plots.py`
- Demo: `python demo/app.py` (reads logs only; no simulator needed)
- HPC job template: `hpc/job_template.sbatch`
