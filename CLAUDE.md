# CLAUDE.md — Gated fast/slow perception for a simulated drone

Read `PLAN.md` first. It is the source of truth for scope. If this file and `PLAN.md` disagree, stop and ask.

## Project in one paragraph
A simulated drone flies on fast perception (Depth Anything V2 + SAM 2 + rule controller). A small learned **gate** predicts when that perception is wrong (labels from simulator ground truth). Only when the gate fires do we ask **Qwen3-VL** a yes/no question. Deliverable: crash rate vs. VLM calls/episode for never / always / random / hand-tuned threshold / learned gate / oracle, plus failure analysis, Gradio replay and a report. Simulation only.

## Environments (do not mix them up)
- **GCP GPU VM**: AirSim, closed-loop episodes, anything needing a display or sudo.
- **NYU HPC (SLURM)**: batch perception, gate training, standalone VLM eval. No sudo, no display. Work in scratch, not home. Cache models via `HF_HOME` on scratch.
- **Laptop (macOS)**: editing, Gradio demo, Plan B simulator, plotting. No GPU assumptions.
- Before running anything, state which environment the command is for. Never launch HPC jobs, start/stop cloud VMs, or spend credits without asking.

## Working rules
1. **Plan first for anything non-trivial.** Small, reversible steps. Prefer the simplest thing that meets the milestone in `PLAN.md`.
2. **No scope creep.** Items under "Cut first if late" in `PLAN.md` §3 stay cut unless the user says otherwise.
3. **Determinism.** Every episode is keyed by `(split, seed)`. Seed lists live in `configs/seeds.yaml` and are frozen after Day 3. Never edit them afterwards. Never tune anything on test seeds.
4. **Split by seed, never by frame.** Any code that splits data must be checked for leakage.
5. **No ground truth at runtime.** Controller, gate and policies must not read GT depth/segmentation/collision. GT is for labels and metrics only. Flag any code path that violates this.
6. **Do not fabricate results.** Never invent numbers, plots, or citations. If a run failed or was skipped, say so with the output. Results in the report must trace to logged runs.
7. **Surrogate VLM results must be labeled** "surrogate VLM" everywhere (tables, plots, file names).
8. **Reuse before writing.** Check `PLAN.md` §5 layout and existing modules first. All policies go behind the single interface in `policies/`.
9. **Match surrounding code style.** Python 3.10+, type hints on public functions, config via YAML in `configs/`, no hard-coded paths.
10. **Long jobs must be resumable**: append per-episode results to JSONL/CSV and skip finished `(policy, seed)` pairs on restart.

## Git workflow
- **Commit or push only when the user asks.** Never force-push, never rewrite history, never commit data, weights, or secrets (`data/`, `*.pt`, `*.ckpt`, `.env`, credentials are gitignored).
- Branch per milestone area when helpful (`sim`, `gate`, `vlm`, `eval`); `main` should always run.
- Commit messages: imperative, one line summary, a body saying *why*. Prefix with the area: `sim:`, `perception:`, `gate:`, `vlm:`, `eval:`, `demo:`, `docs:`.
- The project directory is **not yet a git repo**. First commit request: ask for the remote URL, then `git init`, add `.gitignore`, and commit.

### After every commit (do all of these, then report what you did)
1. **Append to `logs/PROGRESS.md`**: date/time, commit hash, one-line summary, which `PLAN.md` day/phase it advances, and what is now unblocked or still blocked.
2. **Append to `logs/AI_LOG.md`** (see below) any decisions, assumptions, or mistakes from the work in this commit.
3. **Run a quick sanity check relevant to the change** (unit tests if they exist, or the smoke command for that area) and record pass/fail. Do not hide failures.
4. **Check the working tree is clean** and nothing large or secret was staged (`git status`, `git diff --cached --stat` before the commit; `git show --stat` after).
5. **Backup reminder**: if the commit touched `configs/`, `gate/` labels, or scenario generation, remind the user to `rsync` data/logs to the second location (VM ↔ HPC scratch ↔ cloud storage) per `PLAN.md` §12 "Backups to create now". Run the sync only if asked.
6. **Milestone tags**: if this commit completes a checkpoint (Day 5 closed loop, Day 8 gate, Day 11 full pipeline), suggest a tag (`day5-loop`, `day8-gate`, `day11-pipeline`) and ask before tagging.
7. **Push only if the user asked** to push. Otherwise say it has not been pushed.

## Monitoring and logging AI behaviour
Goal: a transparent, auditable record of what the assistant did, assumed, got wrong, and why, so the report can describe the methodology honestly and mistakes are caught early.

Maintain **`logs/AI_LOG.md`** (append-only, newest at the bottom). Create it on first use. One entry per meaningful action or decision:

```
## YYYY-MM-DD HH:MM  |  <area>  |  <short title>
- Request: what the user asked (quote or paraphrase)
- Action: what was done (files touched, commands run, environment: VM/HPC/laptop)
- Assumptions: anything guessed rather than known; mark each as [UNVERIFIED] until confirmed
- Decisions & alternatives: what was chosen, what was rejected, why
- Evidence: command output, metric, seed, run id, or file path backing the claim
- Outcome: success / partial / failed (with the error text if short)
- Follow-ups: open questions or checks needed
```

Rules for the log:
- **Log**: design decisions (label definition, features, thresholds, latency model), any deviation from `PLAN.md`, every failure or retry, any fallback from §12 that was triggered (name the blocker ID, e.g. "B4-A"), any result that surprised us, and any time the assistant was corrected by the user.
- **Mark uncertainty explicitly.** Do not present guesses, recalled library APIs, or memory of paper numbers as verified. Mark them `[UNVERIFIED]` and verify (run it, read the source, or look it up) before relying on them.
- **Flag risky behaviour in the entry** (and tell the user immediately): deleting or overwriting files, long or costly jobs, changes to frozen configs, anything that could cause test-set leakage, anything touching credentials.
- **Cite sources**: research claims carry arXiv ids (as in `PLAN.md`); dataset/model versions carry exact names and checkpoints.
- **Reproducibility fields** on every experiment entry: git commit hash, config file(s), seed list, model checkpoint names, environment.
- **Weekly self-review (Day 7 and Day 13)**: summarize from `AI_LOG.md` the assumptions that turned out wrong, repeated failures, and anything unverified that still influences results. Put the summary at the top of the next report draft.

Also keep **`logs/runs.jsonl`**: one line per experiment run with `{time, commit, env, command, config, seeds, n_episodes, wall_seconds, outcome}`. The eval scripts should append to it automatically; if they don't yet, add that first.

## Escalation: stop and ask the user when
- A kill switch from `PLAN.md` §12 is hit (Day 2, 3, 5, 8, 9, 11, 12 checks) — recommend the fallback, don't silently switch.
- A task would exceed ~1 hour of compute, use paid resources, or need the user's credentials.
- Results look too good (e.g. gate AUROC > 0.99, crash rate 0%) — suspect label leakage or GT use at runtime before celebrating.
- Requirements are ambiguous in a way that changes the experiment (label definition, latency model, metric definitions).
- Open questions from `PLAN.md` §11 become blocking.

## Reporting style
Be concise and factual. State what was run, where, and the result. Say plainly when something failed, was skipped, or is unverified. Give a recommendation rather than a menu of options. Reference code as `path:line`.

## Quick commands (fill in as they come to exist)
- Run one episode: `python eval/run_episode.py --policy never --seed 0`
- Sweep: `python eval/sweep.py --config configs/sweep.yaml`
- Metrics/plots: `python eval/plots.py`
- Demo: `python demo/app.py` (reads logs only; no simulator needed)
- HPC job template: `hpc/job_template.sbatch`
