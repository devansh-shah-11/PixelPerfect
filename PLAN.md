# Pixel Perfect: Plan Overview

**Working title:** *Ask Only When It Matters: Latency-Aware Deferral to Vision-Language Models for Safe Drone Flight*
**Course:** CSCI-GA 2271 Computer Vision, NYU, Fall 2026 · **Track:** Agent · **Format:** Research project
**Team:** Pixel Perfect, 3 members · **This file:** v2 overview

> **What this file is.** One page of orientation: the idea, the result we are going for, where we actually are, and who owns what. Read it first, then go to `TASKS_DETAILED.md`.
>
> **What this file is not.** It is not the design document and it holds no decisions of its own. Every design decision, metric definition, risk, kill switch, related-work claim and backlog item lives in **`PLAN_DETAILED.md`**, which is still the source of truth. Nothing in this file is stated twice there, so the two cannot drift. If you need to know *why* something is the way it is, the answer is in `PLAN_DETAILED.md` and it wins.

---

## 1. Map of the documents

| File | Holds | Changes when |
|---|---|---|
| `PLAN.md` (this file) | Orientation, current position, milestone map, ownership | A milestone is reached, or the position changes |
| **`PLAN_DETAILED.md`** | **Source of truth:** decisions (D1 to D20), architecture, labels, policies, metrics and statistics, interfaces, compute plan, related work, risks and kill switches, experiments, backlog (Appendix A) | Through a PR with one teammate's approval, plus a Decision-log row (`PLAN_DETAILED.md` §0) |
| `TASKS.md` | What to do next: milestones, their gates, ordered chunks | Continuously, by whoever is working |
| `TASKS_DETAILED.md` | The original dated task board, kept for the task ids (`A-xx`, `B-xx`, `C-xx`, `W-xx`), handoffs H1 to H9 and the progress log | Only to look things up; the dates on it have passed |
| `CLAUDE.md` | Rules for AI assistants working in this repo | When the way we work changes |
| `AGENT_LOG.md` | Append-only record of AI use, mistakes and fixes. Feeds the report appendix, and items 2 and 3 of it carry the most weight in Agent Track grading | Every meaningful assistant action |
| `logs/PROGRESS.md` | Append-only per-commit progress | Every commit |
| `hpc/instructions.md` | How to get on Cloud Bursting, the partitions that work, and §3.5 what the capability runs actually proved | Every HPC finding |
| `report/related_work.md` | Full related-work notes and reading assignments | As papers are read |

---

## 2. The idea in plain language

A self-flying drone decides many times a second whether to keep going, climb, turn or stop.

- The **fast stack** (quick thinker) is a fast depth model plus simple steering rules. It keeps up with flight, but it misses thin wires, glass and fog.
- The **VLM judge** (careful thinker) understands scenes far better, but it is much too slow to ask about every frame.
- The **gate** (referee) is a tiny learned model that looks at every frame and asks: is the fast stack probably wrong here, and is it worth the delay to find out?
- Only when the gate fires does the drone ask the VLM **one yes/no question** about its flight path. **The VLM never flies the drone.** Its answer only picks between continuing and avoiding.

```
frame -> fast stack (depth) -> controller proposes an action
             |
             +-> gate: ask?  -- no --> do the proposed action
                      |
                     yes -> pause sim -> VLM yes/no -> charge delay d -> avoid or continue
```

**What makes this a flight problem and not a text-cascade problem:** asking is not free. A drone at 5 m/s that waits 2 s has travelled 10 m. The cost of asking is distance, not money.

**The one result we want:** crash rate against VLM calls per flight, every policy compared at equal query rates, with 95% confidence intervals, plus a breakdown of what caused each crash. The sentence we want to be able to write is "the gate keeps X% of always-ask safety with Y% of the queries". A modest X is still a real result if the crash-cause analysis explains it. Negative results are explicitly valued in this course.

Full architecture, labels (L1, L2, L3), the policy list and the metric definitions: `PLAN_DETAILED.md` §4 to §6.

---

## 3. Where we actually are

**Verified**

- Project AirSim v1.0.1 runs headless on the Cloud Bursting **A100** partition (`c12m85-a100-1`), natively, no container. L4 nodes failed to boot for two members.
- A **CityEnviron frame bank exists**: 411 episodes, 19,267 frames, with RGB, segmentation, float16 depth, pose, condition, obstacle type and ground-truth distance per frame, on HPC scratch. This is the data the perception and VLM work runs on right now.
- Repo skeleton, shared types in `interfaces.py` and the HPC smoke test are in place.
- **The VLM judge runs, and it is fast on the A100:** Qwen3-VL-2B answers one yes/no question in 0.073 s median (p95 0.076 s), the 8B in 0.089 s (p95 0.091 s), one forward pass over the overlay and the crop (job 33341, `README.md`). This is well under the seconds of delay the project's motivation talks about. It does not weaken the question, because the delay `d` is a swept, controlled variable by design (D10) and stands for onboard or shared compute, but the report must say so plainly rather than imply the A100 number is the cost of asking.
- **First VLM accuracy is weak but informative** (job 33462, 2B, 500 val frames): precision 0.90, recall 0.40. It sees the red ball and, thanks to the crop, the thin wires; it misses low-contrast poles, plain walls and, systematically, a building face that fills the view (every one of those frames got a more confident "no" than an empty frame). The cases the VLM misses are not the cases the depth model misses, which is the project's premise. The final M2 table needs the full run; see `TASKS.md` M2.4.
- **Simulator facts found in the VLM sheets:** glass is see-through to both the depth and the segmentation cameras, so image-based ground truth for glass does not exist in the bank; and the "axis_z" poles and wires render as horizontal bars at camera height, not vertical posts.
- **M1 has passed:** the bank split is frozen (202 train / 40 val / 39 test, 130 excluded, checksum `6e3af228c9f6`), Depth Anything V2-Small has been run over all 19,267 frames, and L1 labels plus a scale check exist. Still on a branch, not on `main`.

**Known broken or unknown** (all from `hpc/instructions.md` §3.5, which is the honest version; the first job log over-reported passes)

- **The closed loop exists, but only on the Plan B raycast simulator so far.** Kinematic drone, rule controller, own swept collision check, paused-time delay loop, resumable sweep and metrics all run and are tested; 60 of 60 true-depth flights reached the goal with no crash. The Project AirSim backend is written but untested until M4.1 and M4.2 run. There are still no crash rates from the real simulator or the real fast stack.
- **Thin wires (M4.1, job 33776):** horizontal 1 cm and 3 cm wires are in the segmentation and the ground-truth depth at 3, 6 and 10 m, so the wire labels are real. A *vertical* 1 cm wire was invisible to both, probably sub-pixel sampling; scenario design must keep that in mind.
- **Segmentation does mark spawned objects** (ID 100, colour 185,243,231) since the v2 capability script; confirmed again in M4.1. Glass is the exception: it has no segmentation pixels.
- **Pause/step works exactly (M4.2):** paused time does not move, each 100 ms step is exactly 100 ms, 0.22 s of wall time per step with images (0.45x real time). D10 stands.
- **Weather is unusable from this bank:** fog, dust and rain+fog render near black, rain 0.8 is identical to clear, effects contaminate each other, and only roughly 06:00 to 19:00 is bright enough. Keep clear and dusk, treat rain as clear, exclude fog, exclude or mask snow for depth labels, and report no weather results from this bank.
- **The fast stack's depth is suspect, not just imperfect.** Val AbsRel 2.42 raw and 1.88 scaled, with 76 to 78% of in-corridor obstacles missed and 100% of wires, poles, balls and glass. An error bigger than the distance itself points at a bug (perspective-to-planar conversion, the metric checkpoint in this domain, or the corridor catching ground and sky) rather than at a genuine perception limit. Until it is explained, every miss rate and anything trained on these labels is provisional. See `TASKS.md` M1.9.
- Bank episodes are straight runs at a single obstacle, keyed by seed 10000 + episode. They are **not** `configs/seeds.yaml` and must be split by episode. Good for saved-frame studies, not a substitute for closed-loop crash rates.

**The honest summary:** workstream B has real work in flight on the frame bank, workstream A has a simulator but no flights, and **workstream C has not started**, which is what currently blocks the first curve.

---

## 4. Milestone map

Order, not dates. Each milestone has one gate, which is the test that says it is finished, and a fallback if the gate fails. The gates and the chunks are in `TASKS.md`.

| | Milestone | Gate (how we know it is done) | Owner |
|---|---|---|---|
| **M0** | Foundations | Everyone can push, run a job, and import `interfaces.py` | all |
| **M1** | Frame bank is usable | Frozen episode split, depth clipping, estimated depth and L1 labels for every kept frame | B |
| **M2** | The VLM can see the obstacles | Recall by obstacle type and distance measured, above the floor in `PLAN_DETAILED.md` §11 blocker 6 | **C** |
| **M3** | Saved-frame results exist | Gate beats the depth-variance baseline on val, or we can say why not. This is the fallback paper | B + C |
| **M4** | The drone flies closed-loop | Same seed reproduces the same flight exactly, and true-depth flights crash near zero | A |
| **M5** | First curve | All MVP policies run end to end; crash rate against calls on val seeds; about 30 crashes assigned a cause. Tag `mvp-v1` | all |
| **M6** | The research results | Replay labels (headline 1) and the delay sweep (headline 2), then test seeds run **once** | all |
| **M7** | Report, demo, walkthrough | 4 to 9 page report, repo that reproduces the figures, each member can explain every module | all |

The only real dates left: **final test runs launch at the start of the last week of November**, the report and presentation are due in the last week of the semester. Everything else is ordered by dependency. If we are late, `PLAN_DETAILED.md` §13 says what to cut and in what order.

---

## 5. Who owns what

| | A: Simulator and flights | B: Fast stack and gate | C: VLM and evaluation |
|---|---|---|---|
| Owner | Devansh | Parva | Sarvesh |
| Owns | scenarios, obstacles, kinematic drone, collision check, paused-time delay loop, replays and forking, sweeps, HPC jobs | depth model and scale, controller, features, labels L1/L2/L3, gate models, uncertainty baselines | VLM serving, prompts, path overlay and crop, latency, VLM accuracy, scene-change and anomaly baselines, surrogate judge, metrics and statistics, figures, demo |
| Paper section | Simulator and setup | Method | Experiments and analysis |

Ownership is inferred from who has done the work so far, and closes `PLAN_DETAILED.md` Q1. Confirm it at the next sync and correct this table if it is wrong.

**Branches:** one per chunk, named `<workstream>/<chunk-id>`, for example `C/M2.1`. Merge into `main` at least weekly. `main` should always run. Small, frequent commits: one bulk commit at the deadline is not accepted by the course.

---

## 6. The rules that bite every day

Short list only. The full set is `CLAUDE.md` for assistants and `PLAN_DETAILED.md` §3 for decisions.

1. **Split by episode or seed, never by frame.** Check every split for leakage.
2. **Tune on val only. Test seeds are opened once, at the end.**
3. **No ground truth at runtime.** The controller, gate and policies never read ground-truth depth, segmentation or collisions. Ground truth is for labels and metrics only.
4. **Never invent a number.** If a run failed or was skipped, say so with the output. Every number in the report traces to a logged run.
5. **Surrogate VLM results are labelled "surrogate VLM"** everywhere, and a real-VLM run validates them.
6. **The delay `d` is never zero**, and it is charged in simulated time.
7. **Results that look too good are a bug until proven otherwise.** Gate AUROC above 0.99 or a 0% crash rate means suspect leakage first.
8. **Long jobs resume.** Append per-episode results and skip finished work on restart, because idle nodes are killed after 20 minutes.
9. **Deviating from a decision** means a Decision-log row in `PLAN_DETAILED.md`, not a quiet change in code.

---

## 7. Open questions that block work

Owner and full list: `PLAN_DETAILED.md` §16. The ones that actually block something now:

| # | Question | Blocks |
|---|---|---|
| Q4 / D3 | Project AirSim v1.0.1 instead of Microsoft AirSim or Colosseum needs team sign-off and a Decision-log row | nothing technically, but it is an undocumented deviation |
| Q10 | CoDrone (arXiv 2512.19083): what are its scheduler inputs and reward, and does it account for collisions? If it does, revisit D16 | how we frame the contribution |
| Q3, Q6 | TA: is simulation-only evaluation accepted, and what are the exact final dates? | M7 planning |
| new | Does pause/step work in Project AirSim, and do wires appear in segmentation or ground-truth depth? | M4, and the whole wire story |
