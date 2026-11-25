# Plan v2: scenario freeze (M4.7)

Draft for review (DS, 2026-11-24, revised after reading Parva's PR #4 files). Not a decision record. `PLAN_DETAILED.md` stays unchanged; any deviation goes in `AGENT_LOG.md` plus a Decision-log row.

Task: new start poses, difficulty retune to never-ask crashes of 15 to 30%, then freeze the scenario (`TASKS.md` M4.7). Also Problem 2: the never-ask noise floor.

## What Parva's files settled
(Read locally on `sim/scenario-freeze`, after merging `B/M6.3-results`.)
- **Map-object crashes are not in never-ask.** `results/sweeps/ask_harm.txt`: 55 "scenery" crashes = 42 on the placed obstacle (guard fired first) + 13 elsewhere. Never-ask has 6 of the 42 and none of the 13.
- **The 13 map-object crashes are all at pose 0 (x = 13 m, altitude 3.9 to 6 m) and pose 2 (x = 18.5 to 21.5 m, altitude 4 to 9 m; one at x = 1.5 m).** Eight end while climbing, five while continuing (after climbing to 4 to 9 m). Only policies that ask, and so climb, hit them. Pose 0's objects are posts, not overhead (Parva's correction).
- **Pose scan** (job 42050, 3,280 captures, `results/pose_clearance/report.md`): poses 1 and 4 are clear at 2 to 12 m; 0, 2 and 3 are obstructed. Parva says the "bad capture" column is unreliable, so judge poses by "map hits" and "depth under 2 m". The scan probes only straight along each pose's heading.
- **Never-ask at 48% (29/60) comes from placed thin and transparent obstacles:** pole 11/12, glass 8/8, wire_3cm 4/5, wire_1cm 3/6, boxes 3/19, wall 0/9. Never-ask crashes by pose are 6/3/6/8/6 of 12, too few to read as pose difficulty.
- **The crash-cause table shows "controller" 0 everywhere,** but that run had no true-depth reference, so the controller was never checked.
- **Job commits were not recorded** (Parva's log), so old numbers can't be tied to a commit. One clean rerun on one code version after the freeze replaces them.
- **Owners** (Parva's log): DS poses and freeze; SB retunes the real-VLM cut and replay answers (M6.2); PP reruns both sweeps once after the freeze.

## New risk: scene richness against the real VLM
Parva's real-Qwen runs (12 and 30 flights, jobs 41843, 41890): with cut -4.02, 12 of 13 asks with nothing within 30 m got "yes" (bank val rate 10%), and 6 of 7 asks under 15 m. The cut does not transfer to flight scenes (his guess: new scenes, prompt without distance) [UNVERIFIED].
- Rich scenes (trees, buildings beside the path) make the real VLM say "yes" to far scenery, and a "yes" makes the controller climb. The surrogate (flat fpr 0.14) hides this.
- Pick poses by corridor clearance only; do not tune scenery to flatter the VLM. Record richness as a covariate.
- The 30-flight real-Qwen check (never 1/5, gates 0/5, random 1/5, oracle 0/5, wide CIs) shows the pipeline runs and nothing more.
- Retuning the VLM cut is SB's job (M6.2); ask that it use frames from the frozen poses.

## Steps (laptop unless stated)
1. **Log first.** `AGENT_LOG.md` entry (DS) with the corrections and this risk. Done on 2026-11-24 (see the entry "Problem 1: map-object crashes, pose scan, surrogate caveats").
2. **Poses.**
   - Keep poses 1 and 4. Need 3 to 5 in total (D24), so at least one more.
   - Shortlist from `capability_CityEnviron_16300/results.json` (128 candidates; local, gitignored): open at 2 m, nothing in the corridor, buildings and trees beside the path allowed. Earlier shortlist: A (50, -50, 270), B (-50, 50, 270), C (-50, -50, 90); spares D (150, 150, 270), E (50, 50, 180). The seg-colour richness proxy is [UNVERIFIED].
   - Verify each candidate with `eval.pose_clearance` on the A100 (altitudes 2 to 12 m, x 0 to 40 m, 2 passes). Accept only with no map hits and no "depth under 2 m" points. Ask before submitting; a few minutes of A100.
   - Put the chosen `starts` in a new sweep config copied from `configs/sweeps/airsim_replay_gates.yaml`. Do not edit Parva's configs.
3. **Difficulty retune from never-ask only** (does not depend on the VLM).
   - Regenerate val scenarios from seeds with `sim/scenario.py`, add an in-path flag, and tabulate never-ask crashes by family and in-path from `results/sweeps/replay_gates_surrogate/results.jsonl`.
   - Predict the rate for candidate `p_in_path` (now 0.8) and family weights. Target about 25%. Keep every family; the retune comes from obstacle placement, not from emptying the map.
   - Tune on val seeds 0 to 59 and check on a later block via `seeds.skip`. No test seeds.
   - Wall: 0/9 crashes. Keep a small share as the controller sanity check (D6), or drop it: team call.
   - **Result (laptop, 2026-11-24, old poses, surrogate-independent):** regenerating the 60 val scenarios with the sweep's settings reproduces the logged family on 60/60 flights. An obstacle counts as in-path when its lateral offset is under 1 m. In-path flights: 53, never-ask crashed 29 (0.55, Wilson 0.41 to 0.67). Off-path flights: 7, crashed 0. Thin or transparent obstacles in the path crash every time (pole 11/11, glass 8/8, wire_3cm 4/4, wire_1cm 3/3); boxes 3/18; wall 0/9; ball has no in-path flight.
   - **So the difficulty knob is `p_in_path`.** Predicted never-ask rate = p_in_path x 0.55: 0.8 gives 0.44 (observed 0.48), 0.5 gives 0.27, 0.45 gives 0.25 (range 0.19 to 0.30), 0.4 gives 0.22. Recommended start: `p_in_path` 0.45, all families kept. About half the flights then carry an off-path obstacle in view that the gate and VLM should ignore, which fits the goal of a scene the model has to read rather than an empty map. The prediction assumes the old poses; the headroom run (step 4) is the real test.
4. **Headroom run on the A100 (ask first).**
   - Never-ask only, at least 100 val seeds (M4.7), about 22 s per flight, so about 40 minutes per pass. Run the same seeds twice for the noise floor (earlier repeat: 58/60 agree).
   - Resumable JSONL and one `logs/runs.jsonl` line per run.
   - A100 nodes have failed to boot before (jobs 41680 and 41846, 17 attempts), so use `--requeue` and expect delays. About 1.5 A100-hours of the 300.
5. **Freeze** only if never-ask is 15 to 30%: `python -m sim.scenario --freeze-scenario <sweep>.yaml --to configs/scenario.yaml`. A Decision-log row is needed. `PLAN_DETAILED.md` is not edited unless asked.
6. **Handoff.**
   - Parva: hold old-pose reruns until the freeze, then rerun both sweeps on one commit with the new poses and `seeds.skip` (tune val 0 to 59, score a later block).
   - SB: the real-VLM cut needs retuning on frozen-pose frames.

## Open decisions
- Number of poses: 3 (fastest) or 5 (more variety, more verification).
- Wall share in the catalog: keep small or drop.

## Verification
- Poses: the `eval.pose_clearance` report shows zero map hits and zero "depth under 2 m" points at every altitude from 2 to 12 m on each chosen pose.
- Difficulty: never-ask crash rate with Wilson 95% CI over at least 100 val seeds is in 15 to 30%; report the two-pass agreement.
- Tests: `.venv/bin/python -m pytest` on the laptop after any code change (known failure: `test_quantized_judge_refuses_cpu`).
- No HPC job, push or commit without approval.
