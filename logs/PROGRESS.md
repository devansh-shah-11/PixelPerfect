# Progress log (append-only)

## 2026-10-20 23:10  |  c2df14b  |  docs: plan, CLAUDE.md, README, gitignore
- Advances: PLAN.md Day 1 (repo skeleton / repo exists).
- Repo initialized and pushed to github.com/devansh-shah-11/PixelPerfect (main, fast-forward on top of bce9faf).
- Unblocked: Day 1 tasks (GCP VM, HPC env/SLURM template, weights download, directory skeleton).
- Still open (PLAN.md §11): team size, GCP credits, sim-only acceptance, OpenFly requirement, HPC specifics.
- Sanity check: ignore rules verified with `git check-ignore` (data/, *.pt, .DS_Store, __pycache__). No tests exist yet.

## 2026-10-29 19:26  |  70f3252  |  docs: align CLAUDE.md with PLAN.md; AI log moved to AGENT_LOG.md
- Advances: housekeeping only (S-01 repo setup). No Sprint 1 implementation task yet.
- Not pushed.
- Unblocked: assistant rules now match PLAN.md v1.0, so work can follow PLAN.md and TASKS.md directly.
- Still blocked: every Sprint 1 task is TODO with no owner; all Sprint 1 dates through Oct 28 have passed (see AGENT_LOG.md). Simulator (A-01/A-02) is the critical path.
- Sanity check: cross-reference script (every §, D-id, Q-id, task id in CLAUDE.md exists): pass. No leftover `AI_LOG` references: pass.

## 2026-10-29 20:00  |  f27ee87  |  docs: PLAN.md v1.1 after meeting with Prof. Xie, HPC facts, related work
- Advances: S-03 (related work started), Q5 and Q7 closed; D17 decided pending A-01.
- Not pushed. Needs one teammate's review (PLAN.md §0).
- Unblocked: Sprint 1 (re-dated, Thu Oct 29 to Wed Nov 11) can start: owners (Q1), Cloud Bursting test jobs, A-01 on an L4.
- Still blocked: TASKS.md dates are stale; workstream owners unassigned.
- Sanity check: cross-reference and table check over CLAUDE.md, PLAN.md, report/*.md: pass.

## 2026-10-29 20:09  |  078e7ac, f0ebbeb  |  skeleton + interfaces + smoke test; CoDrone read
- Advances: S-04 (draft interfaces), S-06 (smoke test ready), S-03 (CoDrone done via AI summary), TASKS.md re-dated to PLAN.md v1.1.
- Not pushed (4 commits ahead of origin/main).
- Unblocked: B and C can code against interfaces.py with DDOS stand-ins; each member can run the smoke test today.
- Still blocked: workstream owners (Q1); partition names; simulator (A-01 to A-03).
- Sanity check: pytest 7 passed; HPC scripts pass bash -n.

## 2026-10-30 12:50  |  PR  |  logs committed; branch S/plan-v1.1-skeleton pushed
- Advances: the 4 local commits are now on GitHub for review (PLAN.md §0).
- Unblocked: teammates can review PLAN.md v1.1, interfaces.py (S-04) and the smoke test (S-06).
- Still blocked: workstream owners (Q1); partition names; simulator (A-01 to A-03, frame due Sat Oct 31).
- Sanity check: pytest 7 passed.

## 2026-10-30 13:30  |  docs  |  logs/next_steps_PP.md: five starting action items
- Advances: S-06 / A-01 (instructions for the first Cloud Bursting session), S-03 (human read of CoDrone and AESOP), Q1.
- Pushed to S/plan-v1.1-skeleton, so it joins PR #1.
- Unblocked: PP can start the L4 smoke test today.
- Still blocked: workstream owners (Q1); partition names; simulator (A-01 to A-03, frame due Sat Oct 31).
- Sanity check: docs only; no code changed.

## 2026-11-13 21:50  |  merge  |  main merged into S/plan-v1.1-skeleton; simulator smoke test reproduced by PP
- Advances: A-01 (simulator renders on HPC for a second member, job 31620, A100), S-06 (HPC facts and partitions written down), PR #1 mergeable again.
- Pushed to S/plan-v1.1-skeleton (PR #1). Needs one teammate's review (PLAN.md §0).
- Unblocked: PR #1 can be merged; PP can start perception and gate work (W-S2, W-B1 code, W-B3) on the laptop.
- Still blocked: reading the CityEnviron frame bank on /scratch/dns5508 (permission denied); owners not written in TASKS.md; week-plan dates (Nov 2 to 8) have passed; closed loop not built.
- Sanity check: pytest 7 passed; bash -n on hpc/*.sh and *.sbatch: ok.

## 2026-11-13 22:10  |  perception  |  depth clipping, corridor stats, L1 labels, Depth Anything V2 wrapper
- Advances: W-S2 (done, unit-tested), W-B3 (label function and tests done; eyeball check pending), W-B1 (wrapper done; bank run pending). Also B-01 to B-04 in the Sprint 1 list.
- Pushed to B/W-B1-perception; PR opened on top of PR #1.
- Unblocked: the bank runner and the split script can be written against the frames.jsonl format.
- Still blocked: any real result needs the frame bank on /scratch/dns5508 (permission denied) or our own bank; `d_safe` not chosen.
- Sanity check: pytest 29 passed; real model ran on the Mac (22 ms per frame).

## 2026-11-13 22:45  |  perception  |  frame-bank reader, episode split script, resumable bank runner
- Advances: W-S1 (split script done; the split file itself needs the bank's episodes.jsonl), W-B1 (runner done; HPC run pending).
- Pushed to B/W-B1-perception (PR #2).
- Unblocked: one HPC job produces per-frame GT and estimated corridor distances for all 19,267 frames.
- Still blocked: read access to /scratch/dns5508/frame_bank; `d_safe` not chosen; PR #1 not merged.
- Sanity check: pytest 39 passed; both CLIs ran end to end on a fake bank with the real model.

## 2026-11-14 00:20  |  perception  |  reader verified on the real bank; runner saves the depth map before the row
- Advances: W-S1 (split written on HPC: 202/40/39, 130 excluded), W-B1 (5-frame CPU run matches the bank's corridor minimum).
- Pushed to B/W-B1-perception (PR #2).
- Unblocked: the full bank job.
- Still blocked: configs/bank_split.yaml exists only on HPC; possible stale frames in the bank (GT corridor minimum constant over 5 steps) to be checked.
- Sanity check: pytest 39 passed.

## 2026-11-14 00:45  |  perception  |  configs/bank_split.yaml frozen (202/40/39, 130 excluded)
- Advances: W-S1 (split file committed; gallery note still open), W-B1 (full bank job 32117 submitted by PP).
- Pushed to B/W-B1-perception (PR #2).
- Unblocked: any analysis can now name its split.
- Still blocked: job 32117 output; `d_safe` choice (must stay well under about 18 m, see AGENT_LOG.md).
- Sanity check: `load_split` checksum ok; pytest 39 passed.
- Backup reminder: a frozen config was added. The bank itself lives only on /scratch/dns5508; a second copy is worth making (PLAN.md §11). Not run.

## 2026-11-14 12:55  |  perception  |  full bank run done (19,267 frames); analysis script added
- Advances: W-B1 (bank run complete, job 32117), W-B2 and W-B3 (script for scale check, miss rates and eyeball sheet; not yet run on real data).
- Pushed to B/W-B1-perception (PR #2).
- Unblocked: first real numbers (miss rate per obstacle type, scale), then the threshold baseline and first gate (W-B4).
- Still blocked: `d_safe` choice; PR #1 and PR #2 not reviewed.
- Sanity check: pytest 44 passed.

## 2026-11-14 15:10  |  gate  |  first results; clean GT labels from the segmentation image
- Advances: W-B2 (scale check run: no correction frozen), W-B3 (eyeball check done on 20+20; rain label noise found and a fix written), W-B1 (results: 76% of dangerous train frames missed at d_safe 8 m).
- Pushed to B/W-B1-perception (PR #2).
- Unblocked: clean labels for all frames (CPU only), then the threshold baseline and first gate (W-B4).
- Still blocked: `d_safe` choice; simulator questions for DS (floating obstacles, one start point, 400x225 resolution); PRs not reviewed.
- Sanity check: pytest 45 passed.

## 2026-11-15 14:55  |  (pending commit)  |  docs: plan split into a one-page PLAN.md and a milestone TASKS.md
- Advances: a trackable board (new `TASKS.md`, M0 to M7 with gates, no dates); both open branches merged into one line of work; Q1 filled in pending confirmation.
- Branch `S/plan-v2-milestones`, built on `main` + `sim/capability-test` + `S/plan-v1.1-skeleton`. Not pushed.
- Unblocked: SB can start workstream C at chunk M2.1 (serve Qwen3-VL, measure latency), which is the current longest pole for the first curve.
- Still blocked: C has no finished HPC job yet (M0.3); the frame-bank work is still on PP's branch (M1.5); wires unverified (M4.1); pause/step untested (M4.2); Project AirSim sign-off (M0.5).
- Sanity check: cross-reference checker over the two new files: all ids and paths resolve except `frames.jsonl` (on HPC scratch, gitignored). `interfaces.py` parses. pytest not run (no venv on this laptop).

## 2026-11-15 15:35  |  7c59ece  |  vlm: Qwen3-VL judge, corridor overlay and crop, surrogate judge, latency benchmark
- Advances: M2.1 (benchmark and sbatch written, job not run), M2.2 (overlay and crop, checked on a synthetic wire scene), M2.3 (`ask()` behind the interface, fake-model tests), M2.5 (surrogate judge, done pending measured rates). Branch `C/M2.1`, not pushed.
- Unblocked: DS can build the paused-time delay loop (M4.6) against `vlm.surrogate.SurrogateJudge`; the latency job can be submitted as soon as SB has the HPC env (M0.3).
- Still blocked: M2.1 numbers (needs the A100 job); M2.4 standalone evaluation (needs the bank reader and split from PP's branch, M1.5); eyeballing the overlay on real bank frames (PNGs are on HPC scratch).
- Sanity check: pytest 19 passed, 1 skipped (laptop, `.venv`, Python 3.11). transformers 5.23.0 source checked for the class export, the processor call path and the `dtype=` kwarg.

## 2026-11-18 16:45  |  ea37902  |  vlm results for benchmarking (jobs 35101 latency, 35102 bank evaluation)
- Advances: M2.1 (latency and memory of all five systems, measured), M2.4 (evaluation v3 on 8 conditions), M2.7 (inputs ready; table not built).
- Unblocked: judge pick on memory (`@4bit`, 6.9 GB); jobs 3 and 4 (quantized accuracy) now known to load and run.
- Still blocked: `results/benchmark.md` and the D23 recall on depth's misses need `da2_small.jsonl` (job 32117, teammate's scratch) or a rerun (`hpc/run_bank.sbatch`, new).
- Sanity check: results read only on the laptop; no tests run for this commit (data only). Latency rows measured on random-noise frames. Not pushed by the assistant.

## 2026-11-20 15:30  |  (pending commit)  |  da2_small analysis, clean labels, benchmark, first learned gate
- Advances: M1.8, M1.9 Part A, M2.4 verdict input, M3.3 to M3.5 (best gate val AUROC 0.829 vs baseline 0.425), M4 start-pose smoke (10/10). Not pushed.
- Unblocked: gate in the closed loop (M5). Still blocked: headroom sweep (M4.7), quantized VLM rows, D23 approval, M1.5 merge.
- Sanity check: all three bank files have 19,267 rows; the analysis rerun reproduced the earlier numbers.

- 2026-11-20 18:50 | 41b96c7 | gate: load the learned gate from sweep configs and compute image features in flight | TASKS.md M5.1 / M5.3 (plumbing) | Unblocked: a gate policy in any sweep config (model_path + tau); trained pickle in place on HPC (val AUROC 0.829). Still blocked: M5 gate sweep waits for the M4.7 headroom run; 0.4 vs 0.5 m step mismatch in temporal features needs a team call. Sanity: pytest 146 passed, 1 skipped. Pushed to origin/C/M2.1.
