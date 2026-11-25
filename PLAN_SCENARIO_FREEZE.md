# Plan v7: a realistic course before the scenario freeze (M4.7)


Draft for review (DS). Not a decision record: `PLAN_DETAILED.md` stays unchanged; deviations go in `AGENT_LOG.md` plus Decision-log rows.
Goal point with landing, a random goal inside a scanned cone, 1 to 5 obstacles, situation-based avoidance with the VLM saying where to go, return to cruise altitude, and map-scenery crashes logged properly. Then retune difficulty and freeze. Repo copy `PLAN_SCENARIO_FREEZE.md` (still v2) is refreshed to this in step 1.

## Context
Until now every flight was a straight 40 m line with exactly one placed obstacle (`n_obstacles: [1, 1]` in every AirSim config), success meant "x >= 40 m", and the controller avoided only by climbing in place and never came back down. Parva's sweep and DS's jobs 42952/42953 showed never-ask crashes come from depth-blind obstacles in the path, while map objects hit only drones that had climbed. DS wants a course closer to a real mission before freezing: reach and land at a destination, several obstacles, go around them, come back down, and know the cause of every crash.

## Decisions (DS, 2026-11-24/25)
- Never-ask band 35 to 40% (deviation from `PLAN_DETAILED.md` 15 to 30%, inside the 10 to 80% kill switch). Wall share small.
- No spawned trees; map trees and buildings stay. L1 labels already see them (ground-truth depth, `gate/labels.py:107`), L3 replay counts map crashes. Map-scenery crashes are logged and reported as their own cause.
- **Mission:** reach the goal (x, y) and land: descend to 1 m above the ground at the goal. Landing ends the flight and is not a crash. The landing spot must be free of obstacles.
- **Course:** goal random inside a cone from each verified start pose: 50 to 60 m ahead, up to 10 m to either side. 1 to 5 obstacles at random positions between start and goal, in the path or beside it.
- **Avoidance:** decided by the situation, never hard-coded; the VLM, when asked, says where to go (left, right or over). Return to cruise altitude afterwards (new DESCEND action).
- **Time limits:** max flight 120 s (was 60 s); "lost" when the distance to the goal has not shrunk for 30 s.
- Every course is random but reproducible from `(split, seed)` (`sim.scenario.generate`); tune on val only, test seeds untouched.

## Design
- **Action set:** add `Action.DESCEND` in `interfaces.py` (down at `climb_rate`, moving at `v_descend`, floor at the landing height). Changes D4's action set: Decision-log row. Modules that switch on actions: `sim/drone.py` (`step`), `control/rules.py`, tests `tests/test_sim_loop.py` and `tests/test_interfaces.py`.
- **Scenario (`sim/scenario.py`):** `Scenario` gains `goal_xy` (path frame) and `land_alt_m` (1.0). New placement mode in `ScenarioConfig` (old mode stays default so old configs keep their meaning): goal distance U(50, 60) m and lateral U(-10, 10) m; n ~ U{1..5}; obstacle positions uniform along the start-to-goal line in [12 m, goal distance - 8 m] with a minimum gap of 8 m (draws that cannot fit are dropped, the realised count is logged); in-path with `p_in_path` (lateral N(0, 0.3) from the line), else 3 to 8 m beside it; no obstacle within 6 m of the goal (landing spot clear). `pick_start` checks goal distance + 5 m against `clear_m`.
- **How the drone knows the target:** the goal (x, y) is part of the mission, given at takeoff like a GPS waypoint; the drone knows its own position and heading each step (`DroneState`, like GPS plus IMU). Both are legitimate run-time inputs (CLAUDE.md rule 5 bans ground truth about obstacles, not the mission or own state).
- **Controller (`control/rules.py`), run-time inputs only** (estimated depth, own pose, mission goal, VLM answer):
  - Steer toward the bearing from the drone's position to the goal every step (replaces the fixed `goal_bearing = 0`); turn when off by more than 10 degrees.
  - Depth says blocked (no ask): score left, right and over from the fast stack's depth estimate (free distance in each band, a bonus for the side toward the goal), pick the highest. Kept while it stays free, re-scored when the chosen side becomes blocked.
  - VLM asked and answers "left", "right" or "over": the controller follows that direction for the avoidance episode (`vlm_avoid_s`, re-checked as above). This is what lets a depth-blind obstacle (wire, glass) be avoided on the correct side. "No": fly on.
  - Return to cruise: after avoiding, hold altitude until the drone has flown `d_safe` (8 m) past the avoidance start with the estimate clear, then DESCEND toward cruise altitude while moving forward.
  - Landing: within 2 m of the goal horizontally, STOP forward motion and DESCEND to `land_alt_m`.
- **VLM answer with a direction (`vlm/`):**
  - Real judge (Qwen3-VL): one question about the drawn path, answer one of no / left / right / over. Changes the prompt (D9 allows one prompt round, chosen on val) and the parsing; the confidence cut for "yes vs no" is kept and retuned by SB on new frames.
  - Surrogate judge (`vlm/surrogate.py`): keeps the measured yes/no rates; given a "yes", returns the correct side with probability `p_dir` and another side otherwise. The correct side is computed by the evaluation loop from the obstacle geometry (as `set_truth` does today). `p_dir` is not measured yet: start at 0.8 [UNVERIFIED], then replace it with the rate measured from the real VLM on val frames. Everything labelled "surrogate VLM".
  - Policies and the gate do not change: the gate still decides when to ask; the slow system stays swappable (D18). Decision-log row: the judge's answer changes from yes/no to no/left/right/over.
- **Flight loop (`eval/run_episode.py`):** `t_max` 120 s. Success = within 2 m of the goal horizontally and at or below `land_alt_m` + 0.1 m. Ground contact stays a crash. Four outcomes, reported separately: landed, crashed, lost (distance to goal not reduced for 30 s, or the drone leaves the scanned cone), timeout (120 s). New result fields: `outcome`, `landed`, `goal_dist_m`, the VLM direction per ask; crashes tagged with the phase (cruise, avoid, return, landing).
- **Scenery crash logging:** when the depth guard fires with no placed obstacle within 1 m, record `scenery` with x, y, altitude, heading, phase, pose index and the segmentation colour of the nearest window pixel (extends `guard_hit`, `eval/run_episode.py:128`). `eval/crash_causes.py` and `eval/ask_harm.py` report map scenery as its own cause, per pose and per phase.
- **Poles:** use the vertical `pole_10cm_axis_y` instead of the horizontal `pole_10cm_axis_z` bar (`sim/scenario.py:43,47-48`; `gate/labels.py:27-29` confirms the old one renders as a bar); keep the horizontal wires (like power lines). Add its `HALF_WIDTH_M` entry. Pole VLM accuracy in `results/benchmark.md` was measured on bars: note it.

## Steps
1. **Log (laptop).** `AGENT_LOG.md` (DS, 2026-11-25): pose scan and precheck results, the decisions above, the corrected answer on map-object labels, the horizontal-pole finding, that all replay labels and gates become void, and draft Decision-log rows (band, mission with landing, course, DESCEND action, avoidance rule, VLM direction answer, 120 s limit, pole orientation) as text for a teammate. Refresh `PLAN_SCENARIO_FREEZE.md`. `PLAN_DETAILED.md` is not edited.
2. **Code + unit tests (laptop).** Scenario, action, controller, judge (direction), loop and logging changes above, each with tests: deterministic course per seed, obstacle count and spacing, landing spot clear, goal-bearing steering, side scoring and re-scoring, VLM direction followed, climb fallback, return to cruise, landing success not a crash, lost after 30 s, timeout at 120 s, scenery logging. `.venv/bin/python -m pytest`.
3. **Laptop smoke on the Plan B raycast backend** (`sim/raycast.py`, no GPU): never-ask, oracle and always (surrogate VLM with direction) on 20 val seeds; check tracks from logged poses, landings, going around vertical posts, over horizontal wires, no flip-flop.
4. **Wider pose scan (A100, ask first).** Poses: old 1, old 4, B, D, and the two untested 65 m candidates (-150, -150, 90) and (-150, 150, 0). `eval/pose_clearance.py` gains a lateral offset: x 0 to 66 m in 2 m steps, y -12, -6, 0, 6, 12 m, altitudes 1, 2, 4, 6, 10 m, 2 passes: about 10,000 captures, about 40 minutes. Keep 5 poses with zero map hits in the cone and at the landing area (x 50 to 60, y -10 to 10, 1 to 10 m).
5. **Retune difficulty.** Predict never-ask per `p_in_path` with P(no crash) = product over obstacles, using the per-family in-path rates (precheck: depth-blind families about 1.0, black box about 0.5, grey box about 0.1, wall 0) and the 1-to-5 count distribution; then a never-ask headroom run on the A100 (ask first): 5 poses, at least 100 val seeds (skip 60), twice for the noise floor. Never-ask flights end at a crash or landing, about 15 to 25 s each, so about 1 hour per pass. One adjustment of `p_in_path` and one re-run if outside 35 to 40%.
6. **Freeze:** `python -m sim.scenario --freeze-scenario <sweep>.yaml --to configs/scenario.yaml` (it must carry the new fields).
7. **Pose-shortcut check (laptop, CPU)** on the new replay labels: leave-one-pose-out gate check and per-pose tables.
8. **Handoffs.** Parva: old labels and gates void; one rerun on one commit after the freeze. SB: direction prompt and the real-VLM cut on new frames, and the measured `p_dir` for the surrogate. Commits from the laptop; results copied with `scp`; `logs/runs.jsonl` not committed from the HPC.

## Risks
- Scope: a controller, judge and scenario rework, several days of work plus about 3 to 4 A100-hours for scan and headroom, and it blocks Parva's and SB's reruns until the freeze.
- Compute after the freeze: 120 s flights make slow policies (always-ask, stuck flights) up to twice as long as before; the sweeps will cost more A100 time. Budget it before Parva's rerun.
- Surrogate direction accuracy `p_dir` is a guess until measured; results using it are development-only.
- Horizontal wires span 6 m: going sideways may not clear them; "over" must be available and chosen when both sides read blocked.
- Descending blind: the camera looks forward only, so return-to-cruise and landing rely on the course design; crashes in those phases are tagged by phase, not hidden.
- More map exposure when going around: more scenery crashes, counted and reported.
- A100 nodes have failed to boot before; `--requeue` and extra wall time.

## Verification
- Laptop: full pytest passes (known failure `test_quantized_judge_refuses_cpu`); raycast smoke shows landings and sensible tracks for 20 seeds.
- HPC: pose scan shows zero map hits in the cone and landing area for the 5 chosen poses; never-ask crash rate with Wilson 95% CI over at least 100 val seeds in 35 to 40%; two-pass agreement, per-family, per-cause, per-phase and per-outcome tables saved.
- No HPC job, push or commit without DS approval.
