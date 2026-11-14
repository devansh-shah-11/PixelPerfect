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
