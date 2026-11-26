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

## 2026-11-22 13:21  |  gate  |  temporal gate features per metre flown (0.4 vs 0.5 m step mismatch fixed)
- Advances: M5.1 / M5.3 (gate in the flight loop now sees the same feature scale as in training), M3.4 (retrained, val AUROC unchanged at 0.829).
- Pushed to C/M2.1 (PR #3).
- Unblocked: the M5 gate sweep no longer depends on the cruise speed; the scenario freeze can pick any speed.
- Still blocked: every gate pickle trained before this commit must be retrained (keys renamed d_ -> dpm_); M4.7 headroom run; PR #3 review.
- Sanity check: pytest 147 passed, 1 failed (test_quantized_judge_refuses_cpu, pre-existing laptop ImportError); raycast plumbing sweep ran (10 val seeds, not a result).

## 2026-11-22 13:39  |  eval, gate  |  flight forking and replay labels; L2/L3 label functions; CPU sbatch
- Advances: M6.1 (forking, code and tests), M6.3 groundwork (L2, L3, replay runner), gate labels L2/L3 (§4.3).
- Pushed to C/M2.1 (PR #3).
- Unblocked: replay labels on the raycast backend now (CPU job); on Project AirSim after M4.7.
- Still blocked: real replay labels need the da2 stack on AirSim and cached VLM answers (M6.2); L3 definition (save-only positives) needs one teammate's agreement.
- Sanity check: pytest 160 passed, 1 failed (pre-existing, laptop ImportError); bash -n hpc/cpu_job.sbatch ok.
## 2026-11-22 23:55  |  eval  |  close the simulator after replay and sweep runs (job 39930 hung to its time limit)
- Advances: M6.3 (first AirSim replay labels: val, 20 flights, 827 frames, surrogate VLM).
- Unblocked: train replay resubmission. Still blocked: gate.train_replay needs the train replay labels.
- Sanity check: pytest 156 passed, 1 failed (pre-existing laptop ImportError).
## 2026-11-23 00:20  |  gate  |  paired flight-level bootstrap for gate-L3 minus gate-L1 (Headline 1)
- Advances: M6.3 (evaluation of gate-L3 vs gate-L1). First AirSim val replay labels in: 20 flights, 59 L3 positive frames (surrogate VLM).
- Running: 40411 (train replay, 20 flights), 40412 (val replay extended to 60 flights).
- Sanity check: pytest 157 passed, 1 failed (pre-existing laptop ImportError).
## 2026-11-23 04:10  |  hpc  |  job scripts exit with the command's exit code
- Advances: M6.3. Train replay labels in (20 flights, 183 L3 positive frames, surrogate VLM); val at 41 of 60 flights.
- Sanity check: bash -n on the three scripts ok; rc capture checked locally.
## 2026-11-23  |  gate  |  per-family flights-saved table and saved val scores in the replay gate report
- Advances: M6.3. Replay labels complete for this config: train 100 flights, val 60 (surrogate VLM).
- Sanity check: pytest 157 passed, 1 failed (pre-existing laptop ImportError).
## 2026-11-23  |  gate  |  first Headline 1 result (surrogate VLM); tau per query rate in the replay report
- Advances: M6.3. gate-L3 val AUROC against L3 0.857 vs gate-L1 0.717; L3 recall higher at 5/10/20% (paired CIs exclude 0); flights saved not significantly different offline.
- Next: closed-loop sweep on val with both gates (M5), needs the taus from a rerun of gate.train_replay.
- Sanity check: pytest 157 passed, 1 failed (pre-existing laptop ImportError).
## 2026-11-23  |  eval  |  closed-loop sweep config: gate-L3 vs gate-L1 in flight (surrogate VLM)
- Advances: M5 / M6.3 (Headline 1 in closed loop). Per-family replay table logged.
- Unblocked: the AirSim sweep (configs/sweeps/airsim_replay_gates.yaml). Still blocked: final table needs frozen scenario (M4.7) and real VLM answers (M6.2).
- Sanity check: config parses, non-gate policies construct.
## 2026-11-23  |  sim, eval  |  retry dropped AirSim captures; per-flight judge seed in sweeps
- Advances: M5 / M6.3. Closed-loop sweep at 432 of 600 flights (job 41022), resumable.
- Still blocked: a fair val-tuned threshold baseline; frozen scenario (M4.7); real VLM answers (M6.2).
- Sanity check: pytest 159 passed, 1 failed (pre-existing laptop ImportError).
## 2026-11-23  |  eval  |  paired policy comparison and never-ask repeatability check
- Advances: M5 / M6.3. Partial closed-loop result logged (surrogate VLM): no gate-L3 gain over gate-L1 in flight so far.
- Blocked: trusting replay labels fully until the never-ask mismatch (29 vs 28 crashes) is explained.
- Sanity check: see the commit; the pre-existing laptop ImportError remains.
## 2026-11-23  |  gate, eval  |  paired closed-loop comparison logged; threshold baseline cuts; family counts corrected
- Advances: M5 / M6.3. No gate-L3 vs gate-L1 difference shown in flight on 60 val seeds (surrogate VLM).
- Next: crash causes from the flight logs, then random at matched calls and the tuned threshold.
- Sanity check: full pytest at this commit, pre-existing laptop ImportError only.
## 2026-11-23  |  eval  |  config for matched random and tuned threshold baselines; crash causes logged
- Advances: M5 / M6.3 (surrogate VLM development run). Waiting: main sweep remainder (job 41336).
- Sanity check: config parses, settings equal to the main sweep, policies construct.
## 2026-11-23  |  eval  |  closed-loop gate sweep complete (840 flights, surrogate VLM)
- Advances: M5 / M6.3. Gates crash less than random and the tuned threshold at similar calls; no gate-L3 vs gate-L1 difference shown.
- Blocked: frozen scenario (M4.7) and real VLM answers (M6.2) for a reportable table; `safety retained` needs a team decision, since always-ask completes no mission.
- Sanity check: log-only commit; flight counts 600 and 240, both jobs exit 0.
## 2026-11-23  |  eval  |  real-VLM closed-loop config, with a smoke run first
- Advances: M5 / M6.3 toward real VLM answers (complements M6.2, which caches them for replay). Not run yet.
- Sanity check: config parses, judge constructs, other settings equal to the surrogate sweep.
## 2026-11-23  |  eval  |  tool to separate simulator noise from ask-caused crashes
- Advances: M5.4 (crash analysis). Blocked: real-VLM smoke run (A100 nodes not booting).
- Sanity check: full pytest 164 passed, 1 failed (pre-existing laptop ImportError).
## 2026-11-23  |  eval, sim  |  scenery crashes traced to start poses; guard now records its trigger
- Advances: M5.4, and input to M4.7 (start pose 2 suspected). Next: per-pose crash table on the HPC; real-VLM smoke run.
- Sanity check: full pytest 164 passed, 1 failed (pre-existing laptop ImportError).
## 2026-11-23  |  eval, sim  |  obstacle hits no longer labelled scenery; bare-map scan for start poses
- Advances: M5.4 and input to M4.7. Next: run the scan on the HPC.
- Sanity check: full pytest 166 passed, 1 failed (pre-existing laptop ImportError).
## 2026-11-23  |  eval, sim  |  unplaced-object crashes quantified (13 of 188, start poses 0 and 2)
- Advances: input to M4.7. Next: bare-map scan result (job 41846), then choose replacement start poses.
- Sanity check: log-only commit; counts from eval.ask_harm on the HPC.
## 2026-11-23  |  sim, vlm  |  start poses scanned; real VLM runs in flight but its cutoff needs retuning
- Advances: M4.7 (poses 0, 2, 3 blocked; 1, 4 clear), M5.2 (Qwen answers in the flight job). Blocked: M4.7 freeze, then the M6.3 rerun.
- Report: first draft of the gate section (report/sections/gate.tex).
- Sanity check: docs-only change; scan 3,280 captures and smoke runs exit 0.

## 2026-11-24  |  docs  |  D21 approved (Project AirSim), Q4 closed, AGENT_LOG rule changed
- Advances: M0 (M0.5 done). `CLAUDE.md` now says: after a commit, update the AGENT_LOG entry for the same task and day, or add one for a new task.
- Unblocked: the simulator deviation from D3 is now a recorded decision. D22 to D24 still need teammate approval.
- Sanity check: docs-only commit, no code run.

## 2026-11-24  |  eval  |  seeds.skip for disjoint tune/score blocks; safety retained against the oracle
- Advances: M5 / M6.3 scoring protocol (closed-loop problems 3 and 4 from the Nov 23 sweep). D24 text updated for `skip`.
- Unblocked: Parva can tune cuts on val 0-59 and score on `{skip: 60, n: 100}` after the scenario freeze. Still blocked: the freeze itself (M4.7: new start poses, difficulty retune, wall decision).
- Sanity check: tests/test_policies_metrics.py and tests/test_sim_loop.py, 39 passed.

## 2026-11-25  |  sim  |  DESCEND action and cone courses (65a4a3b)
- Advances: M4.7 (plan v7 step 2, chunk 1 of 5: action set and course generator). Line layout unchanged and still the default.
- Still blocked: controller, judge direction, flight loop outcomes (chunks 2 to 5), then the raycast smoke, the A100 pose scan and the headroom run, then the freeze.
- Sanity check: full pytest 170 passed, 1 skipped (laptop). Note: the commit also carried an earlier v2 draft of PLAN_SCENARIO_FREEZE.md that was already staged; v7 follows in the next commit.

## 2026-11-25  |  control  |  situation-based avoidance and landing (2e96721)
- Advances: M4.7 (plan v7 step 2, chunk 2 of 5: controller). Also commits the AGENT_LOG analysis and plan v7.
- Next: chunk 3 (judge direction), chunk 4 (flight loop outcomes and logging), chunk 5 (vertical pole, configs).
- Sanity check: full pytest 175 passed, 1 skipped (laptop).

## 2026-11-25  |  control, vlm  |  geofence and VLM direction (chunk 3)
- Advances: M4.7 (plan v7 step 2: soft geofence in the controller; chunk 3, the judge's direction answer). Hard fence and outcomes come with chunk 4.
- Sanity check: full pytest 180 passed, 1 skipped (laptop).

## 2026-11-25  |  eval, sim  |  Job A world checks; flight loop for cone courses; raycast smoke (pending commit)
- Advances: M4.7 (plan v7 chunks 4 and 5, steps 3 and 4 code). Job A ready to submit; Job B configs wait for Job A's poses.
- Sanity check: full pytest 189 passed, 1 skipped; raycast cone smoke: oracle 20/20 landed (surrogate VLM, stand-in depth; not a result).

## 2026-11-25  |  eval  |  Job B poses from the world checks (889af3e)
- Advances: M4.7 (plan v7 step 4 done: 5 poses chosen). Next: Job B1 cone smoke, then B2 headroom grid.
- Sanity check: config-only change; both configs parse (checked before the edit).
## 2026-11-25  |  control, eval  |  Edge-loop and side-flip fixes, crash records, Job B2 at DS's settings (78e7629)
- Advances: M4.7 (plan v7 step 5). Controller: the fence cuts only a sideways hold; one left/right switch per avoidance, then over; hold margin 2 m. Every crash records family, phase, action and pose (`extra["crash"]`). Job B2 config: 3 to 8 obstacles 5 m apart, p_in_path 0.35, pose D back; city pose scan config added.
- Sanity check: 193 passed, 1 skipped (laptop); raycast smoke (surrogate VLM) never 12/20 crashed, oracle 20/20 landed.
- Unblocked: Job B2 (DA2, repeat, true depth) and the city scan, submitted by DS. Still blocked: the freeze (final pose list, then one more B2), policy sweeps on city poses (surrogate VLM and oracle truth for map objects).
## 2026-11-25  |  eval, demo, sim  |  Avoidance-band logging, third-person video, Job B3 config (7e5d3c9, 54f8037, 0db6232)
- Advances: M4.7 (scenario freeze). `7e5d3c9`: `extra["avoid_bands"]` (left, right, over, side, action per avoidance step) and `--set` for `eval.flight_video`, for the rerun of B2 true-depth seeds 101150 and 101084 (pose 2, never climbed). `54f8037`: third-person chase view in the flight video (viewer only; dots = surfaces the true depth saw). `0db6232`: `configs/sweeps/airsim_cone_headroom_city.yaml` (B3, 8 poses), not run.
- Sanity check: 196 passed, 1 skipped (laptop); raycast flight rendered and the third-person tile checked by eye (oracle, seed 101062).
- Unblocked: the HPC rerun of two seeds with frames. Still blocked: B3 and the freeze, until the pose 2 / true-depth non-landing question is settled. Not pushed.
