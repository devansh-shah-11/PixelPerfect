# Pixel Perfect: Task Board

Companion to `PLAN.md` (design, decisions, and reasons live there; this file is only *what to do*).
Each person can work independently whenever they have time. Everything you need from someone else has a **stand-in** so you're never blocked waiting.

---

## How to use this file

1. **Pick a task** in your workstream whose dependencies are `DONE` (or that has a stand-in). Set status to `DOING` and put your initials.
2. **Work on a branch** named `<workstream>/<task-id>`, e.g. `A/A-03`. Small commits, daily if possible.
3. **Finish** = the "Done when" check passes. Set `DONE`, add the PR/commit link in Notes, and write any agent mistakes in `AGENT_LOG.md`.
4. **Blocked?** Set `BLOCKED`, write why in Notes, and post in the team chat. Then switch to another task or use the stand-in.
5. **Changing a design decision?** Don't do it here. Open a PR to `PLAN.md` (Decision log).
6. **Don't edit other people's rows** except to add a note. Shared rows (`S-`) can be claimed by anyone.

**Status values:** `TODO` · `DOING (XX)` · `DONE` · `BLOCKED` · `CUT` (dropped, with reason)

**Owners:** A = Simulator and flights (`____`) · B = Fast stack and gate (`____`) · C = VLM and evaluation (`____`)

---

## Sync rhythm

| When | What | Where |
|---|---|---|
| Whenever you work | Update your task rows in this file (status + notes) and push | repo |
| **Daily, async (by 11 pm)** | Post a 3-line update in the team chat using the template below | chat |
| **Mon and Thu, 15 min** | Live sync: blockers and handoffs only | call |
| **Sun, 1 hour** | Code walkthrough: each person walks the other two through their code; merge branches; update `PROGRESS LOG` at the bottom | call |
| Kill-switch dates (see PLAN.md §11) | Go/no-go decision, recorded in PLAN.md Decision log | call |

**Daily update template**
```
[Initials] [date]
Done: A-03 (link)
Next: A-04
Blocked / need from others: none | "need interface X from B by Thu"
```

---

## Handoffs (the only points where we depend on each other)

| ID | What | From → To | Needed by | Stand-in until then |
|---|---|---|---|---|
| H1 | Interface stubs (`perceive`, `act`, `score`, `should_ask`, `ask`) + flight-log schema | All → All | Thu Oct 22 | none (agree on the call) |
| H2 | Simulator that saves RGB + GT depth + seg + pose | A → B, C | Thu Oct 22 | DDOS images |
| H3 | Golden set (~20 saved episodes, train/val split) | A → B, C | Tue Oct 27 | DDOS images |
| H4 | `perception/` + `control/` modules usable in the loop | B → A | Sun Oct 25 | A uses GT depth as fake fast stack |
| H5 | `vlm.ask()` server + path-overlay/crop renderer | C → A | Fri Oct 30 | `vlm/surrogate.py` (random answers with set accuracy) |
| H6 | Never-ask training episodes (300 to 500, train seeds) | A → B | Fri Oct 30 | golden set |
| H7 | Trained gate checkpoint `gate.pt` + `score()` | B → A, C | Mon Nov 2 | threshold policy |
| H8 | Flight forking (replay from frame t) | A → B | Mon Nov 9 | none (L3 waits) |
| H9 | Cached VLM answers for fork frames | C → B | Mon Nov 16 | surrogate judge |

---

## Shared tasks (S)

| ID | Task | Due | Depends on | Done when | Status | Notes |
|---|---|---|---|---|---|---|
| S-01 | Create repo with PLAN.md layout, README, AGENT_LOG.md, TASKS.md; branch rules | Wed Oct 21 | | Everyone can push to their branch | TODO | |
| S-02 | Fill in owners for A, B, C (Q1) | Wed Oct 21 | | Names in header | TODO | |
| S-03 | Read **Stress-Query** paper ([arXiv 2610.23877](https://arxiv.org/html/2610.23877)) and AESOP; 5-line summary each in `report/related_work.md` | Fri Oct 23 | | Summary committed; Q7 answered | TODO | If Stress-Query already does learned gating for UAVs, flag for D16 |
| S-04 | Agree interfaces + flight-log schema (H1) and metric definitions (PLAN.md §5, §6) | Thu Oct 22 | | `interfaces.py` stubs merged | TODO | |
| S-05 | Simulator go/no-go; record D3 and D17 | Fri Oct 23 | A-01, A-02 | Decision-log rows added | TODO | |
| S-06 | Each person: confirm HPC account, partitions, GPU hours (Q5) | Wed Oct 21 | | Notes added here | TODO | |
| S-07 | Weekly walkthrough #1 | Sun Oct 25 | | Each module explained once | TODO | |
| S-08 | MVP milestone review memo (first curve, ~30 crash causes, next decisions); tag `mvp-v1` | Tue Nov 3 | A-10, B-08, C-09 | Memo in `report/`, tag pushed | TODO | |
| S-09 | Report skeleton in Overleaf (sections, owners, metric definitions) | Mon Nov 9 | | Skeleton compiles | TODO | |
| S-10 | Pick final 2 to 3 ablations | Mon Nov 16 | E2 results | Decision-log row | TODO | |
| S-11 | Merge Agent Logs into report appendix | Sun Dec 6 | | Appendix drafted | TODO | |
| S-12 | Slides + rehearsal; each person explains the other two's code | Week 7 | | Dry run done | TODO | |

---

## Workstream A: Simulator and flights

### Sprint 1 (to Tue Nov 3)

| ID | Task | Due | Depends on | Done when | Status | Notes |
|---|---|---|---|---|---|---|
| A-01 | GPU machine ready (GCP VM or HPC node): driver, Vulkan/OpenGL, prebuilt AirSim/Colosseum binary downloaded | Wed Oct 21 | | `nvidia-smi` works; binary launches headless | TODO | |
| A-02 | Python script: take off/teleport, save RGB + GT depth + seg + pose | Thu Oct 22 | A-01 | 10 frames saved to `data/` (H2) | TODO | Kill switch: no frame by Thu → start A-07 full time |
| A-03 | Apptainer try on HPC (time-boxed ≤ half day) | Thu Oct 22 | A-01 | Works or written "no" with reason | TODO | |
| A-04 | Obstacle spawner: wall/box, pole, thin wire (thickness, contrast), glass-like, fog | Fri Oct 23 | A-02 | Wire visible at 10 m in RGB | TODO | |
| A-05 | Own collision check (capsule/raycast vs. spawned geometry); cross-check engine flag | Fri Oct 23 | A-04 | Unit test: drone through wire = collision | TODO | |
| A-06 | Kinematic drone (fixed altitude band, speed v, 5 actions, hysteresis) + scenario generator; **freeze `configs/seeds.yaml`** (train/val/test) | Sat Oct 24 | A-04 | 100 seeded scenarios regenerate identically; seeds checksummed | TODO | |
| A-07 | Plan B simulator skeleton (raycast scene or Habitat-sim), same interface | Sun Oct 25 | | Flies one scripted episode with GT depth | TODO | Insurance; half a day |
| A-08 | `eval/run_episode.py` closed loop using GT depth as fake fast stack; then B's modules (H4) | Mon Oct 26 | A-06, (H4) | True-depth flights crash near zero | TODO | |
| A-09 | Never-ask + true-depth on ≥100 val episodes; measure sec/flight; write budget in README; headroom check (15 to 30% crashes) | Tue Oct 27 | A-08 | Numbers in README; difficulty retuned if needed | TODO | Kill switch Mon Oct 26 |
| A-10 | Save golden set (H3); VM snapshot | Tue Oct 27 | A-09 | 20 episodes on HPC scratch + VM | TODO | |
| A-11 | Paused-time delay loop: pause, call `ask()`, advance by `d` with brake-and-hold | Thu Oct 29 | A-08 | Same seed + same `d` gives identical flight | TODO | Uses surrogate until H5 |
| A-12 | Collect 300 to 500 never-ask episodes on train seeds with action noise (H6) | Fri Oct 30 | A-09 | Logs on HPC scratch, 2 copies | TODO | |
| A-13 | Smoke-test all MVP policies (10 val flights each); first sweep at 2 to 3 operating points | Mon Nov 2 | A-11, C-06, (H7) | `results/mvp_sweep.csv` | TODO | |

### Later (Weeks 3 to 7)

| ID | Task | Due | Depends on | Done when | Status | Notes |
|---|---|---|---|---|---|---|
| A-14 | Flight forking: replay from frame t, two branches (H8) | Mon Nov 9 | A-11 | Fork replays match original up to t | TODO | |
| A-15 | Throughput: parallel sim instances / HPC arrays; resumable `sweep.py` | Mon Nov 9 | A-13 | Restart skips finished pairs | TODO | |
| A-16 | Main grid E3 (all policies × 3 rates, d = 1 s, val-tuned, test once) | Sun Nov 15 | A-15, B-12, C-11 | Headline CSV | TODO | |
| A-17 | Delay sweep E5 (top 3 × d ∈ {0.25, 1, 3}); speed sweep if budget | Sun Nov 22 | A-16 | CSV + plot | TODO | |
| A-18 | Held-out environment runs (E6) | Sun Nov 22 | A-16 | CSV | TODO | |
| A-19 | Speed-limited waiting ablation (D11) | Sun Nov 22 | A-11 | CSV | TODO | |
| A-20 | Final test runs split across 3 HPC accounts; crash-vs-catch videos | Sun Nov 29 | A-16, A-17 | Videos in `results/` | TODO | |
| A-21 | Repo cleanup, reproduction instructions, versions in README; "Simulator and setup" section | Sun Dec 6 | | Fresh clone reproduces one figure | TODO | |

---

## Workstream B: Fast stack and gate

### Sprint 1 (to Tue Nov 3)

| ID | Task | Due | Depends on | Done when | Status | Notes |
|---|---|---|---|---|---|---|
| B-01 | HPC env on scratch; cache Depth Anything V2-Small (metric); sbatch template | Wed Oct 21 | | Test job prints `nvidia-smi` | TODO | |
| B-02 | Depth Anything V2 on DDOS: depth error, thin-wire failure examples, FPS | Thu Oct 22 | B-01 | Notebook + 5 failure images | TODO | Stand-in data: DDOS |
| B-03 | Rule controller (corridor min depth → stop/yaw/climb/continue, hysteresis) | Fri Oct 23 | | Unit tests on hand-made depth maps | TODO | |
| B-04 | `perception/` + `control/` behind the agreed interface (H4) | Sun Oct 25 | B-02, B-03, S-04 | A can import and run them in the loop | TODO | |
| B-05 | L1 label function (missed obstacle) + false-stop flag; unit tests | Sun Oct 25 | S-04 | Tests pass on hand-made frames | TODO | |
| B-06 | Scale check on sim frames; frozen global scale fit if needed; log scale error | Tue Oct 27 | H3 | Scale error plot by distance | TODO | |
| B-07 | Feature extraction on HPC (corridor stats, Canny/Frangi, encoder embedding, speed, temporal) | Fri Oct 30 | H6 | Feature table for train/val | TODO | |
| B-08 | Label sanity (eyeball 20 pos + 20 neg); gradient boosting + depth-variance baseline; AUROC/AUPRC/ECE on val | Sat Oct 31 | B-05, B-07 | Gate beats depth-variance baseline (kill switch) | TODO | |
| B-09 | Gate MLP; export `gate.pt` + `score()` (H7); < 1 ms per frame | Mon Nov 2 | B-08 | A can load it | TODO | |

### Later (Weeks 3 to 7)

| ID | Task | Due | Depends on | Done when | Status | Notes |
|---|---|---|---|---|---|---|
| B-10 | E2 gate study on sim frames: per obstacle family, held-out family, risk–coverage | Mon Nov 9 | B-09 | Table + figure | TODO | Fallback-paper result |
| B-11 | Uncertainty baseline (TTA/ensemble disagreement) | Mon Nov 9 | B-04 | `policies/uncertainty.py` | TODO | |
| B-12 | L2 labels (future collision) and comparison | Sun Nov 15 | B-09 | AUROC table L1 vs L2 | TODO | |
| B-13 | L3 replay labels from forks + cached VLM answers; gate-L3 (E4) | Sun Nov 22 | H8, H9 | gate-L3 vs gate-L1 at equal query rate | TODO | Headline 1 |
| B-14 | Generalization E6 (held-out family/environment) | Sun Nov 22 | B-10 | Table | TODO | |
| B-15 | Stretch: conformal threshold; DAgger round | Sun Nov 29 | B-13 | Result or CUT with reason | TODO | |
| B-16 | "Method" section draft | Sun Dec 6 | | In Overleaf | TODO | |

---

## Workstream C: VLM and evaluation

### Sprint 1 (to Tue Nov 3)

| ID | Task | Due | Depends on | Done when | Status | Notes |
|---|---|---|---|---|---|---|
| C-01 | Serve Qwen3-VL 4B and 8B (vLLM/HF) on HPC; measure latency (median, p95) | Wed Oct 21 | | Latency numbers in README | TODO | |
| C-02 | Ask TA: sim-only evaluation OK? (Q3); confirm presentation dates (Q6); GCP status (Q2) | Fri Oct 23 | | Answers in PLAN.md §16 | TODO | |
| C-03 | VLM on DDOS wires: with/without crop; yes/no via logit(yes) − logit(no) | Thu Oct 22 | C-01 | Precision/recall table | TODO | Stand-in data: DDOS |
| C-04 | Path-overlay + full-res crop renderer (draws next `T_h` s of flight path) | Sun Oct 25 | S-04 | Overlay images look right on 10 frames | TODO | |
| C-05 | `vlm/surrogate.py`: answers with set TPR/FPR per obstacle type | Sun Oct 25 | | Same interface as `ask()` | TODO | Unblocks A-11 |
| C-06 | `policies/` with one `should_ask` interface: never, always, random(p), cooldown | Sun Oct 25 | S-04 | Unit tests | TODO | |
| C-07 | Eval harness `eval/metrics.py`: crash rate/km, success, mission time, calls, latency p50/95/99, safety retained, Wilson/bootstrap CIs; auto results table | Tue Oct 27 | S-04 | Runs on golden set | TODO | |
| C-08 | VLM standalone on ~200 labeled sim frames per obstacle type (kill switch: wire recall ≥ ~60%) | Wed Oct 28 | H3, C-04 | Table by type and distance | TODO | |
| C-09 | `vlm.ask()` server reachable from the loop (H5); threshold(τ) and oracle policies | Fri Oct 30 | C-06, C-08 | A runs one gated flight | TODO | |
| C-10 | Crash-cause tool (gate miss / VLM wrong / delay / controller) on first ~30 crashes | Mon Nov 2 | A-13 | Pie/bar + examples | TODO | |

### Later (Weeks 3 to 7)

| ID | Task | Due | Depends on | Done when | Status | Notes |
|---|---|---|---|---|---|---|
| C-11 | VLM ablations: crop on/off, path on/off, 4B vs 8B, 2 prompt variants; cache answers per frame | Mon Nov 9 | C-08 | Table | TODO | |
| C-12 | Fixed-interval, scene-change, anomaly (AESOP-style) policies | Sun Nov 15 | C-06 | Policies pass smoke test | TODO | |
| C-13 | Cached VLM answers for fork frames (H9) | Mon Nov 16 | A-14 | Cache file on scratch | TODO | |
| C-14 | Headline figure, crash-cause figure, CI tables (scripts only) | Sun Nov 22 | A-16 | `results/figures/` regenerated by one command | TODO | |
| C-15 | Commercial VLM comparison (only if budgeted) | Sun Nov 22 | C-11 | Result or CUT | TODO | |
| C-16 | Gradio replay (frame, depth est vs GT, gate score, VLM markers); GIF fallback | Sun Nov 29 | A-12 | Loads a saved flight with no simulator | TODO | |
| C-17 | Venue/deadline check (CVPR/ICRA 2027 workshops, IROS 2027, RA-L) | Sun Nov 29 | | Dates in PLAN.md §15 | TODO | |
| C-18 | "Experiments and analysis" section draft; failure gallery | Sun Dec 6 | | In Overleaf | TODO | |

---

## Backlog pool (anyone, when ahead)

Pull from `PLAN.md` Appendix A. Copy the item here with an ID (`X-01`, ...) before starting.

| ID | Task | Owner | Status | Notes |
|---|---|---|---|---|
| X-01 | SAM 2 features ablation | | TODO | |
| X-02 | Explicit cost-aware gate score (speed × risk vs. delay) | | TODO | |
| X-03 | Alternate open VLM (InternVL / Qwen2.5-VL) | | TODO | |

---

## Progress log (updated at each Sunday walkthrough)

| Week | Date | Done this week | Slipped (and new date) | Decisions made | Risks / kill switches hit |
|---|---|---|---|---|---|
| 1 | Sun Oct 25 | | | | |
| 2 | Sun Nov 1 | | | | |
| 3 | Sun Nov 8 | | | | |
| 4 | Sun Nov 15 | | | | |
| 5 | Sun Nov 22 | | | | |
| 6 | Sun Nov 29 | | | | |
| 7 | Sun Dec 6 | | | | |
