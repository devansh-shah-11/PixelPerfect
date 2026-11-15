# Pixel Perfect: Task Board

Companion to `PLAN.md` (orientation) and `PLAN_DETAILED.md` (the source of truth for every design decision). This file is only **what to do next, and in what order**.

**No dates.** Work is ordered by what it unblocks, not by a calendar, because every date on the old board went stale and stopped being read. The only real deadlines are: final test runs launch at the start of the last week of November, and the report and presentation are due in the last week of the semester.

**Status:** `TODO` · `DOING (XX)` · `DONE` · `BLOCKED` · `CUT` (with a reason)

**How to use it**
1. Find your workstream's next unchecked chunk in the lowest open milestone. Set the chunk status to `DOING (your initials)`.
2. Work on a branch named `<workstream>/<chunk-id>`, for example `C/M2.1`. Small commits.
3. A chunk is finished when its **done when** line is true, not when the code is written.
4. A milestone is finished when its **gate** passes. If the gate fails, take the fallback written under it and record it in `AGENT_LOG.md`. Do not quietly carry on.
5. Changing a design decision happens in `PLAN_DETAILED.md` through a PR with a Decision-log row, never here.
6. Old task ids (`A-xx`, `B-xx`, `C-xx`, `W-xx`) are kept in brackets so the old board, the handoffs H1 to H9 and the commit history still resolve. Look them up in `TASKS_DETAILED.md`.

---

## Right now, one thing each

| Who | Next chunk | Why this one |
|---|---|---|
| **Devansh (A)** | M4.1 verify wires through segmentation and ground-truth depth | The whole thin-wire story is unverified, and it decides whether wires are our main failure case or we switch to poles and glass |
| **Parva (B)** | M1.9 explain the depth quality, then M1.5 merge into `main` | M1's gate has passed, but a val AbsRel of 2.42 and a 76% miss rate need a cause before any gate is trained on them |
| **Sarvesh (C)** | M2.1 serve Qwen3-VL and measure latency | Nothing in workstream C has started, and C is what blocks the first curve (M5) |

---

## M0 · Foundations

**Gate:** every member can push to a branch, run an sbatch job on Cloud Bursting, and import `interfaces.py`.
**Status: nearly done.** Only Sarvesh's smoke test is outstanding.

| | Chunk | Owner | Done when | Status |
|---|---|---|---|---|
| M0.1 | Repo, README, `AGENT_LOG.md`, branch rules [S-01] | B | Everyone can push to their branch | DONE |
| M0.2 | Shared types and flight-log schema in `interfaces.py` + tests [S-04, H1] | B | `pytest` passes; all three have read it and agreed | DONE (needs A and C to confirm they agree) |
| M0.3 | HPC access: account, partitions, scratch, cached weights [S-06] | each | Each member has one finished sbatch job id | DOING: A done, B done, **C outstanding** |
| M0.4 | Owners written down, Q1 closed [S-02] | all | `PLAN.md` §5 table confirmed at a sync | DOING (filled in, needs confirming) |
| M0.5 | Project AirSim sign-off: Decision-log row for D3 and Q4 [S-05] | all | Row added to `PLAN_DETAILED.md` §3 | TODO |

---

## M1 · The frame bank is usable

The CityEnviron bank (411 episodes, 19,267 frames) is the data everything runs on until the closed loop exists.

**Gate:** a frozen episode split plus estimated depth and L1 labels for every kept frame, with no episode in two splits. **Passed**, with the two caveats in M1.8 and M1.9 carried into M3.
**Fallback if the gate fails:** DDOS images for perception and VLM work (`PLAN_DETAILED.md` D12), and say plainly in any result that the bank was not used.

| | Chunk | Owner | Done when | Status |
|---|---|---|---|---|
| M1.1 | Episode split `configs/bank_split.yaml`, stratified by obstacle and condition, with a checksum; keep clear and dusk, rain as clear, exclude fog, exclude or mask snow [W-S1, D13] | B | Script check shows no episode in two splits | DONE: train 202 / val 40 / test 39, 130 excluded (fog 53 + snow 77), checksum `6e3af228c9f6` |
| M1.2 | Depth clipping and sky mask as one function in `perception/` [W-S2] | B | Unit test on 3 saturated frames (65504 m sky) | DONE |
| M1.3 | Depth Anything V2-Small (metric) over the kept frames, resumable JSONL, corridor minimum per frame [W-B1, D7] | B | JSONL for every kept frame | DONE: job 32117, 19,267 rows in `da2_small.jsonl` |
| M1.4 | Scale check against ground-truth distance; frozen global scale and shift fit on **train episodes only** if needed [W-B2] | B | Plot plus a frozen parameter file | DONE (median scale 0.790). Whether to freeze a scale correction is still a team decision; scaling moved val false stops 10 to 24 |
| M1.5 | **Merge the frame-bank work into `main`** | B | PR merged; `main` runs | TODO, and it blocks M3 |
| M1.6 | L1 label function in `gate/labels.py`, false stops logged separately, unit tests, eyeball 20 positive and 20 negative [W-B3, §4.3] | B | Tests pass plus an eyeball note | DONE, but see M1.8 |
| M1.7 | Ground-truth corridor distance from the segmentation mask | B | Agrees with `gt_along_m` on a sample | DONE (code and track check); not yet computed on the real bank, see M1.8 |
| M1.8 | **Clean the L1 labels:** run `gate.bank_gt` and the analysis with `--gt` on the real bank, so labels come from the segmentation mask rather than raw corridor depth | B | Label rates recomputed; rain no longer noisy | TODO, **blocks gate training in M3.4** (flagged by B: labels must be fixed before any gate is trained) |
| M1.9 | **Explain the depth quality before trusting any miss rate.** Val AbsRel is 2.42 raw and 1.88 scaled, with 76 to 78% of in-corridor obstacles missed: wires, the 10 cm pole, the 20 cm ball and the glass pane 100%, 1 m boxes about 90%, but the white wall only 2.5%. An error larger than the distance itself is a bug signal, not a result | B | A written cause: perspective-to-planar conversion, the metric checkpoint on CityEnviron, or the corridor window catching ground and sky (the corridor sees ground at about 12 x altitude) | TODO. Treat as a bug until proven otherwise (`CLAUDE.md` escalation). If the depth really is this bad, the gate will mostly learn "depth is broken" and M3's comparison becomes meaningless |

---

## M2 · The VLM can see the obstacles

Nothing here has started, and it is the longest pole for the first curve. Owner: **C (Sarvesh)**.

**Gate:** a recall and precision table by obstacle type and distance, with crop on and off and path drawing on and off, for the 2B model.
**Kill switch:** if recall is below roughly 60% on the types that are definitely visible (poles, boxes, white walls, glass), take the backup chain in `PLAN_DETAILED.md` §11 blocker 6, in order: bigger crop and zoom, prompt variants, the 8B model, a different open VLM, then the surrogate judge. Wires are reported separately and flagged unverified until M4.1 says whether they render at all.

| | Chunk | Owner | Done when | Status |
|---|---|---|---|---|
| M2.1 | Serve Qwen3-VL-2B and the larger reference model on an A100; one-token yes/no output; median and p95 latency over at least 50 calls [C-01, W-C2, D8] | C | Latency numbers in `README.md` | DOING (SB): `vlm/bench_latency.py` + `hpc/vlm_latency.sbatch` written and unit-tested; the A100 job has not run, so no numbers yet |
| M2.2 | Path overlay and full-resolution corridor crop renderer in `vlm/` [C-04, W-C1, D9]. MVP region is the fixed corridor window the labels use; the projected `T_h` flight tube waits for camera intrinsics | C | Looks right on 10 frames per obstacle type | DOING (SB): `vlm/overlay.py` done and checked on a synthetic wire scene; not yet eyeballed on real bank frames (PNGs are on HPC scratch) |
| M2.3 | `vlm.ask()` behind the `interfaces.py` signature, reading `logit(yes) - logit(no)` so it can be thresholded [H5] | C | Returns answer, confidence and real latency | DOING (SB): `vlm/judge.py` passes the interface tests on a fake model; the real-model code path is unverified until the M2.1 job runs |
| M2.4 | Standalone evaluation on about 200 labelled bank frames per obstacle type; table by type and distance; crop and path ablations [C-03, C-08, W-C3] | C | Table committed; kill switch above decided | TODO, needs M1.5 (bank reader and split on `main`) |
| M2.5 | `vlm/surrogate.py`: fake judge with the true and false positive rates measured in M2.4, same interface [C-05, W-C4] | C | Same interface as `ask()`; every output labelled "surrogate VLM" | DONE (code, tested); rates are placeholders until M2.4 measures them. **A can start M4.6 against it now** |

---

## M3 · Saved-frame results exist (the fallback paper)

If the closed loop never works, this milestone alone is still a defensible project. Do not skip it to chase M4.

**Gate:** a val-only results table showing the gate against the depth-variance baseline, per obstacle family, plus the VLM table from M2.4.
**Fallback if the gate fails:** `PLAN_DETAILED.md` §11 blocker 9, in order: fix the labels first, then the features (temporal, thin-structure, embedding), then L2 labels, then calibration. Reporting that the learned gate does not beat a tuned threshold is an acceptable result if the crash causes explain it.

| | Chunk | Owner | Done when | Status |
|---|---|---|---|---|
| M3.1 | `eval/metrics.py`: crash rate and crashes per km, success, mission time, calls per flight, latency percentiles, safety retained, Wilson and bootstrap CIs [C-07, W-C4, §5] | C | Runs on bank-derived data and produces the results table | TODO |
| M3.2 | `policies/` behind one `should_ask` interface: never, always, random(p), cooldown, plus unit tests [C-06, W-C5, §4.4] | C | Tests pass | TODO |
| M3.3 | Depth-variance threshold baseline, then gradient boosting on corridor statistics; AUROC, AUPRC, ECE on val, per obstacle family [W-B4, B-xx] | B | Val table. AUROC above 0.99 means suspect leakage, not success | TODO |
| M3.4 | Gate-L1 model (small MLP, then embedding variant), under 1 ms per frame [§4.2] | B | Beats M3.3's baseline on val, or a written explanation of why not | TODO |
| M3.5 | Held-out obstacle family check [E6] | B | Numbers for the held-out family reported separately | TODO |
| M3.6 | Report skeleton in Overleaf: sections, owners, metric definitions [S-09] | C | Skeleton compiles | TODO |

---

## M4 · The drone flies closed-loop

**Gate:** the same seed reproduces the same flight exactly, and true-depth flights crash near zero. Until both hold, no crash-rate number from this loop means anything.
**Fallback if the gate fails:** the Plan B simulator (`PLAN_DETAILED.md` §11), a lightweight raycast or Habitat-sim scene behind the same interface. A complete simpler study beats a stalled realistic one.

| | Chunk | Owner | Done when | Status |
|---|---|---|---|---|
| M4.1 | **Verify wires:** spawn 3 cm and 1 cm wire, check the segmentation id and ground-truth depth along the camera ray at 3, 6 and 10 m, not by eye [W-A1, §11 blocker 4] | A | A written yes or no per thickness and distance | TODO, decides the main failure case |
| M4.2 | Test pause/step and measure seconds per step; this is what D10 depends on | A | Numbers in `hpc/instructions.md` | TODO |
| M4.3 | Kinematic drone (fixed altitude band, speed v, 5 actions, hysteresis), scenario generator, **freeze `configs/seeds.yaml`** [A-06, D4, D13] | A | 100 seeded scenarios regenerate identically, seeds checksummed | TODO |
| M4.4 | Own collision check against known obstacle geometry, engine flag only as a cross-check [A-05, D5] | A | Unit test: drone through a wire counts as a collision | TODO |
| M4.5 | `eval/run_episode.py` closed loop, ground-truth depth as a fake fast stack first, then B's modules [A-08, H4] | A | True-depth flights crash near zero | TODO |
| M4.6 | Paused-time delay loop: pause, ask, advance by `d` with brake and hold, `d` never zero [A-11, D10, D11] | A | Same seed and same `d` gives an identical flight | TODO, uses M2.5 surrogate until the real VLM is wired in |
| M4.7 | Never-ask headroom check on at least 100 val episodes, and seconds per flight written into `README.md` [A-09] | A | Never-ask crashes in 15 to 30% of flights, or difficulty retuned | TODO |
| M4.8 | Golden set of about 20 saved episodes [A-10, H3] | A | Two copies, one on HPC scratch | TODO |

---

## M5 · First curve

**Gate:** every MVP policy (never, always, random, tuned threshold, gate-L1, oracle, plus the true-depth reference) runs end to end; crash rate against calls per flight on val seeds with confidence intervals; about 30 crashes each assigned a cause. Then tag `mvp-v1`.
**Fallback if the gate fails:** cut the grid and freeze scope (`PLAN_DETAILED.md` §13 cut order).

| | Chunk | Owner | Done when | Status |
|---|---|---|---|---|
| M5.1 | Threshold(τ) and oracle policies, tuned as carefully as the gate [C-09, §4.4] | C | Both run in the loop | TODO |
| M5.2 | Real `vlm.ask()` reachable from the flight loop, in the same job as the simulator [H5, D17] | C | One gated flight completes | TODO |
| M5.3 | Smoke test every policy on 10 val flights, then a first sweep at 2 or 3 matched query rates [A-13] | A | `results/mvp_sweep.csv` | TODO |
| M5.4 | Crash-cause tool: gate did not ask / VLM wrong / delay / controller [C-10, §5] | C | Breakdown plus examples for about 30 crashes | TODO |
| M5.5 | Milestone memo and `mvp-v1` tag [S-08] | all | Memo in `report/`, tag pushed | TODO |

---

## M6 · The research results

Both headlines live here (`PLAN_DETAILED.md` D16). Test seeds are opened **once**, at the end of this milestone.

**Gate:** headline 1 and headline 2 each have a figure generated by a script from logged runs, and rerunning the script gives identical numbers.

| | Chunk | Owner | Done when | Status |
|---|---|---|---|---|
| M6.1 | Flight forking: replay from frame t [A-14, H8] | A | Two forks from one frame diverge only by the injected answer | TODO |
| M6.2 | Cached VLM answers for fork frames [C-13, H9] | C | Cache file on scratch | TODO |
| M6.3 | **Headline 1:** L3 replay labels, gate-L3 against gate-L1 at equal query rates [E4] | B | Figure plus table | TODO |
| M6.4 | **Headline 2:** delay sweep, top 3 policies by `d` in {0.25, 1, 3} s [E5] | A + C | Figure showing when asking stops being worth it | TODO |
| M6.5 | Extra asking policies: fixed interval, scene change, anomaly in the AESOP style [C-12] | C | Each passes a smoke test | TODO |
| M6.6 | Slow-system comparison: VLM judge against the VLM choosing the action directly [C-19, E7, D18] | C | Table plus figure | TODO |
| M6.7 | Pick the final 2 to 3 ablations [S-10] | all | Decision-log row | TODO |
| M6.8 | **Final test runs**, split across all three accounts | all | Test seeds used once, no tuning afterwards | TODO, the one hard deadline |

---

## M7 · Report, demo, walkthrough

**Gate:** the repo reproduces every figure with one command, and each member can explain every module.

| | Chunk | Owner | Done when | Status |
|---|---|---|---|---|
| M7.1 | Headline figure, crash-cause figure, CI tables, all from scripts [C-14] | C | `results/figures/` regenerates from one command | TODO |
| M7.2 | Gradio replay, GIF fallback [C-16] | C | Loads a saved flight with no simulator running | TODO |
| M7.3 | Crash-versus-catch side-by-side video | A | Opening slide clip | TODO |
| M7.4 | Report: 4 to 9 pages, track restated, individual contributions | all | Compiles, read by all three | TODO |
| M7.5 | Agent Logs merged into the report appendix, with what the tools got wrong [S-11] | all | Appendix drafted | TODO |
| M7.6 | Slides and rehearsal; each member explains the other two's code [S-12] | all | Dry run done | TODO |
| M7.7 | Verification checklist in `PLAN_DETAILED.md` §14 walked through line by line | all | Every box ticked or explained | TODO |

---

## Parked until a gate says otherwise

- **Thin wires as the headline failure case** until M4.1 proves they render and collide. Poles, glass, white walls and boxes are the hard cases in the meantime.
- **Any weather result**, because the bank's weather is unusable (`PLAN.md` §3).
- **Hardware demo** (D20), decided at M5.
- **Everything in `PLAN_DETAILED.md` Appendix A.** Pull an item in with a Decision-log row, and give it a chunk id here before starting.

---

## Progress log

Per-commit entries are in `logs/PROGRESS.md`, and assistant behaviour in `AGENT_LOG.md`. The old dated board, its handoffs H1 to H9 and its original progress log are in `TASKS_DETAILED.md`.
