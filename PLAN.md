# Pixel Perfect: Master Plan

**Working title:** *Ask Only When It Matters: Latency-Aware Deferral to Vision-Language Models for Safe Drone Flight*
**Course:** CSCI-GA 2271 Computer Vision, NYU, Fall 2026
**Track:** Agent (declared in the submitted proposal; locked) · **Format:** Research project
**Team:** Pixel Perfect, 3 members · **Version:** v1.1, October 29, 2026 (v1.0: October 20)

> This is the single source of truth for the team. If code, slides, or another doc disagrees with this file, this file wins until someone changes it through the process in Section 0.
> It merges three earlier drafts (a two-week sprint plan, a 7-week research plan, and a 14-day engineering plan). All three were drafted with AI assistants, so this file belongs in the Agent Log. Paper descriptions come from abstracts and search summaries: read each paper before citing it.

---

## 0. How to use and change this document

- **Read first:** Sections 1, 2, 4 and 13 (what we're building, the words we use, the design, and this week's tasks).
- **Change process:** any change to a locked decision (Section 3) goes through a small PR to this file, approved by at least one other member, with a new row in the Decision log. Don't silently change behavior in code.
- **Owner of this file:** rotates weekly with the walkthrough host (Section 12).
- **Status tags used below:** `LOCKED` (do not change without a decision-log entry), `DEFAULT` (our current choice, can change after a measured result), `OPEN` (to be decided by the date given).

---

## 1. The project in one page

### The idea (plain language)
A self-flying drone decides many times per second whether to keep going, climb, turn, or stop.

- The **fast stack** (quick thinker) is a fast depth model plus simple steering rules. It keeps up with flight, but it makes mistakes on thin wires, glass, and fog.
- The **VLM judge** (careful thinker) is a vision-language model. It understands scenes much better, but it's far too slow to ask about every frame.
- The **gate** (referee) is a tiny learned model that checks every frame and asks: "Is the fast stack probably wrong here, and is it worth the delay to find out?"
- Only when the gate fires does the drone ask the VLM **one yes/no question** about its flight path. **The VLM never flies the drone**; its answer only selects between continuing and avoiding.
- Everything runs in a simulator, so crashes cost nothing and every flight can be replayed exactly.

### Research question
**Main:** Can the drone be nearly as safe as if it always asked the VLM, while asking only a small fraction of the time?

**What makes it specific to flight:** asking is not free. A drone at 5 m/s waiting 2 s for a VLM covers 10 m. In text-model cascades, asking costs money; in flight, it costs distance or time. So the right question for the gate isn't only "is the fast stack wrong?" but "is it worth the delay to check, given my speed and how close things are?"

### Headline result we want
One figure: **crash rate vs. VLM calls per flight** (and vs. mission time) for every policy, at equal query rates, with 95% confidence intervals. Headline sentence:
> "The gate keeps X% of always-ask safety with Y% of the queries."

A modest X, or a gate that only matches a tuned threshold, is still a valid result if the crash-cause analysis explains why.

---

## 2. Glossary (use these words everywhere)

| Term in this doc and code | Plain alias (slides, layman pitch) | Meaning |
|---|---|---|
| **fast stack** | quick thinker | Depth model + rule controller; runs every frame |
| **controller** | steering rules | Rule-based code that maps the depth in the forward corridor to an action |
| **gate** | referee | Learned model that scores each frame; asks the VLM when the score exceeds a threshold τ |
| **VLM judge** | careful thinker | Open-weights VLM answering one yes/no question about the flight path |
| **policy** | asking strategy | Rule deciding when to ask (never, always, random, threshold, gate, oracle, ...) |
| **forward corridor** | flight path ahead | The 3D tube the drone will occupy over the next `T_h` seconds, and its projection on the image |
| **query rate** | how often it asks | VLM calls per frame (also reported per flight) |
| **delay `d`** | waiting time | Simulated seconds charged per VLM call |
| **episode / flight** | one flight | One start-to-goal run with a fixed seed |
| **oracle gate** | perfect referee | Asks exactly when the label says the fast stack is wrong (upper bound) |
| **true-depth flight** | perfect eyes | Controller fed simulator ground-truth depth (pipeline sanity check and upper bound) |

---

## 3. Decision log

| ID | Decision | Status | Why |
|---|---|---|---|
| D1 | Agent Track, Research format | LOCKED | Declared in proposal; cannot change |
| D2 | Simulation-only evaluation | DEFAULT | Confirm with TA (open question Q3) |
| D3 | Simulator: prebuilt **AirSim/Colosseum binary via Python API only. No ROS 2, no PX4, no OpenFly unless required** | DEFAULT, final by Mon Nov 2 | ROS 2 and sudo were the likeliest week-one sink; nothing in the study needs them |
| D4 | Drone motion: **kinematic stepping** (set pose each step), not full flight dynamics | DEFAULT | Deterministic replays (needed for replay labels), faster, no controller-tuning sink. We study perception, not control |
| D5 | Collisions: **own check against known obstacle geometry** (capsule/raycast), engine flag only as a cross-check | LOCKED | Thin wires may not trigger engine colliders |
| D6 | Obstacles: **spawned programmatically** with controlled type, thickness, color, distance, lighting, fog | LOCKED | Full control of difficulty; no asset hunting |
| D7 | Fast stack: **Depth Anything V2 (Small, metric checkpoint)** + rule controller. **SAM 2 is not in the MVP**; it's an ablation only | DEFAULT | Cheaper and simpler; the question is "is something in the path," not object identity |
| D8 | VLM judge: **Qwen3-VL-2B-Instruct by default; 8B as the "does size help?" reference**. Open weights, run in the same Cloud Bursting job as the simulator | DEFAULT | Small for real-world deployability (meeting with Prof. Xie). One-token yes/no output avoids the free-text hallucinations that broke small VLM copilots in Stress-Query (arXiv 2610.23877). Commercial VLM comparison is optional and must be budgeted |
| D9 | VLM question: **flight path drawn on the image + full-resolution crop of the corridor**, single yes/no token | DEFAULT | VLMs judge distance poorly; downscaling erases 1 to 2 px wires |
| D10 | Latency: **simulator pauses during the VLM call; then simulated time advances by delay `d` while the drone executes the waiting behavior. `d` is never zero** | LOCKED | Results independent of our GPU, exactly repeatable, lets us sweep delay. Real latency reported separately |
| D11 | Waiting behavior: **MVP = brake and hold**; Week 4+ = **speed-limited waiting** (ablation) | DEFAULT | Simple and safe first; research plan's slow-down behavior tested once the loop is stable |
| D12 | Data: **simulator is the training and test set**. DDOS only as an offline sanity test (depth and VLM on thin wires) and as a stand-in before the simulator works. Mid-Air dropped | DEFAULT | Open-loop datasets have a large domain gap |
| D13 | Splits by **scenario seed**, never by frame; train/val/test seed lists frozen at the simulator go/no-go (Mon Nov 2; was Sat Oct 24, see D19) | LOCKED | Prevents leakage |
| D14 | Thresholds and operating points tuned on **val** only; **test** seeds run once at the end | LOCKED | Prevents test leakage |
| D15 | Every policy runs on the **same scenarios and seeds** (paired comparison) | LOCKED | Fewer flights for the same statistical power |
| D16 | Two research headlines: **(1) replay labels, (2) delay as a controlled variable**. VLM failure analysis is a smaller third result | DEFAULT | Both are only possible in simulation; keeps scope realistic for 3 people in 7 weeks |
| D17 | Compute: **NYU HPC Cloud Bursting only** (the Torch cluster is forbidden by the course). Simulator, fast stack and VLM run **in one job on an L4** via Singularity; A100 (40 GB) for batch VLM evaluation and the 8B model. GCP VM only as a fallback | DEFAULT (confirmed once A-01 renders a frame) | Course announcement (Oct 2): 300 GPU-hours per student, L4 and A100. Nodes with low GPU use are shut down after 20 minutes, so a separate, mostly idle VLM server would be killed. L4 is a graphics GPU; A100 may not render Unreal [UNVERIFIED] |
| D18 | **Slow system is swappable; the gate is the contribution.** (a) VLM yes/no judge (MVP); (b) the same 2B VLM chooses one of the 5 actions directly (Sprint 2); (c) stretch: small flow-matching or diffusion policy trained on true-depth flights | DEFAULT | Prof. Xie: flight actions are low-dimensional, so natural-language output may not be needed; suggested VLA or diffusion. Testing the gate with more than one slow system makes the result stronger |
| D19 | **Schedule re-dated.** Sprint 1 had not started by Oct 29; it now runs Thu Oct 29 to Wed Nov 11. Final test runs still launch Mon Nov 30, so Sprint 2 is one week shorter | LOCKED | See Section 13 for what gets cut first |
| D20 | **Hardware demo is a stretch goal**, decided at the MVP review (Wed Nov 11): small indoor drone with a Python SDK and video stream, models running on a laptop next to it | DEFAULT | Prof. Xie called a live demo compelling if simulator results are strong. Nothing is bought before the decision |

**Changes from the submitted proposal** (state these in the report): no ROS 2/OpenFly by default (D3); SAM 2 moved to ablation (D7); simulator data replaces DDOS/Mid-Air as the main data (D12); kinematic drone (D4); 2B VLM by default (D8); swappable slow system (D18).

**Meeting with Prof. Xie (late Oct 2026):** notes in `report/meetings/prof-xie-meeting.md`. Led to D8, D18, D20, the broader related-work section (Section 9), and the presentation framing (Section 15).

---

## 4. Architecture

### 4.1 The decision loop (every frame)

```
camera frame (RGB)
   │
   ▼
FAST STACK: Depth Anything V2 → estimated depth
   │                                         │
   │ features (corridor depth stats,         ▼
   │ encoder embedding, temporal changes) CONTROLLER (rules) → proposed action
   ▼
POLICY / GATE: should_ask(features, drone_state, history)?
   │ no ─────────────────────────────────────────► execute proposed action
   │ yes (rarely; cooldown applies)
   ▼
pause sim → VLM JUDGE(image with drawn path + corridor crop) → yes/no
   │
   ▼
advance sim by delay d with waiting behavior (brake/hold or slow down)
   │
   ▼
yes → avoidance action (stop, climb, or yaw from controller fallback)
no  → execute proposed action
   │
   ▼
next frame
```

### 4.2 Components

**Simulator and drone (owner: A)**
- Prebuilt AirSim (or Colosseum) binary, headless (`-RenderOffScreen` or Xvfb), Python API.
- Kinematic drone: fixed altitude band, constant cruise speed `v` (default 5 m/s; sweep 3/5/8 m/s later), discrete actions `{continue, stop, yaw_left, yaw_right, climb}` with hysteresis.
- Sensors per step: RGB, ground-truth depth, segmentation IDs, pose, velocity.
- Scenario generator: random start/goal corridor; obstacle families:

| Family | Difficulty | Role |
|---|---|---|
| wall / box | easy | controller sanity check |
| pole (vs. bright sky) | medium | depth failure case |
| **thin wire / cable** (thickness and contrast varied) | hard | main failure case |
| glass-like / transparent panel | hard | depth failure case |
| low-texture surface, fog, low light | hard | perception degradation |

- One obstacle family is **held out** from gate training for the generalization test.
- Difficulty is tuned so **never-ask crashes in 15 to 30% of flights**. Below ~10% there is nothing to save; above ~80% always-ask is the only answer.

**Fast stack (owner: B)**
- Depth Anything V2-Small, metric checkpoint. If metric depth is off, fit one global scale+shift on the train split and freeze it (no ground truth at run time).
- Controller: forward-corridor minimum depth → `stop / yaw / climb` if below `d_safe`, else `continue`.
- Log the scale error separately so the gate doesn't learn "scale is wrong" instead of "obstacle missed."

**Gate (owner: B)**
- Input features:
  - corridor depth statistics (min, percentiles, local variance, edge/gradient density)
  - cheap thin-structure features (Canny/Hough/Frangi vesselness on the corridor crop)
  - frozen Depth Anything encoder embedding (the "learned" variant)
  - **drone speed** and distance to the nearest estimated obstacle
  - temporal features (last k frames: depth change, flicker)
- Models, in order: gradient boosting on handcrafted features (strong reference) → small MLP → MLP + embedding. Target under 1 ms per frame.
- Output: `score ∈ [0,1]`; ask if `score > τ`. τ chosen on val, optionally by conformal risk control (stretch).
- **Cooldown:** minimum frames between queries to prevent query storms (all policies).

**VLM judge (owner: C)**
- Qwen3-VL-2B-Instruct (default) and 8B (reference) via vLLM or HF, 4/8-bit if memory-bound. Separate process from the simulator, but **in the same Cloud Bursting job** (D17: idle nodes are shut down after 20 minutes).
- Input: frame with the next `T_h` seconds of flight path drawn as a highlighted region, plus a full-resolution crop of that region.
- Prompt (fixed; variants only in the prompt ablation):
  > *"Does any object, including thin wires, cables, poles or glass, cross the highlighted region? Answer yes or no."*
- Read `logit(yes) − logit(no)` to allow thresholding and confidence.
- Backups if recall on wires is weak: larger crop/zoom, prompt variants, 8B, another open VLM (InternVL, Qwen2.5-VL), and as a last resort a **surrogate judge** (Section 4.4).

### 4.3 Labels for the gate (most important design choice)

| Label | Definition | When |
|---|---|---|
| **L1: missed obstacle (primary MVP label)** | `y = 1` if ground-truth corridor min depth `< d_safe` while estimated corridor min depth `≥ d_safe` (the controller would continue when it should avoid). False stops (reverse case) logged separately | Week 1 to 2 |
| **L2: future collision** | `y = 1` if never-ask flight collides within `k` seconds of this frame | Week 2 to 3 (comparison) |
| **L3: replay label ("worth asking") (headline)** | At frame `t`, fork the flight twice for `H` seconds: (a) fast stack only, (b) VLM answer applied with delay `d` and waiting behavior. `y = 1` only if the outcomes differ (crash vs. no crash). Uses cached VLM answers | Weeks 3 to 5 |

Rules: labels are defined by **control consequence**, not raw depth error. Unit-test the label functions on hand-made frames; eyeball 20 random positives and 20 negatives per label before training.

### 4.4 Policies (one interface, all compared at equal query rates)

| Policy | Description | Phase |
|---|---|---|
| never | fast stack only | MVP |
| always | VLM every frame (with delay) | MVP |
| random(p) | ask with probability p | MVP |
| fixed-interval | ask every n frames | Week 3 |
| threshold(τ) | hand-tuned on corridor min depth or depth variance; **tuned as carefully as the gate** (strongest simple baseline) | MVP |
| uncertainty | depth-model disagreement (TTA/ensemble) or variance | Week 3 |
| scene-change | ask when the image changes enough (dual-process style) | Week 3 |
| anomaly (AESOP-style) | ask when the scene looks unusual vs. training | Week 3 to 4 |
| **gate-L1** | learned gate, L1 labels | MVP |
| **gate-L3** | learned gate, replay labels | Week 5 |
| oracle | asks exactly when L1 (or L3) is positive | MVP |
| true-depth flight | controller on ground-truth depth (not a policy; reference point) | MVP |

**Surrogate judge** (implemented by Nov 2 regardless): a noisy oracle that answers with the VLM's measured true/false positive rates per obstacle type. Used for fast sweeps and as a safety net; every surrogate result is labeled "surrogate," and a smaller real-VLM run validates it.

---

## 5. Metrics and statistics (locked by Fri Oct 30)

**Flight metrics**
- **Crash rate** per flight and **crashes per km**, 95% Wilson or bootstrap CI.
- **Success rate** and **mission time / average speed** (shows the drone isn't safe by not moving).
- **VLM calls per flight** and **query rate** (per frame).
- **Decision latency** per step: median, p95, p99 (real measured), and simulated delay charged.
- **Safety retained (headline):** `(crash_never − crash_policy) / (crash_never − crash_always)`.

**Gate metrics (on saved frames, held-out seeds):** AUROC, AUPRC, ECE, risk–coverage curve; per obstacle family.

**VLM metrics:** precision/recall of "object crosses the path," per obstacle type and distance, with and without crop and path drawing.

**Crash-cause breakdown (main analysis figure):** every crash is assigned to one of
1. gate didn't ask,
2. VLM answered wrong,
3. delay caused it (VLM right, too late),
4. controller issue (would crash even on true depth).

**Statistics**
- Telling 10% from 20% crash rate needs about **200 flights per setting** (α = 0.05, 80% power); 3% vs. 6% needs about **750**. Paired seeds (D15) reduce this.
- At least 3 seed blocks per configuration.
- Smoke-test every new configuration on 10 flights before launching hundreds.

---

## 6. Interfaces (agreed by Fri Oct 30; stand-ins until real parts arrive)

```python
# Flight log: one JSONL row per frame + image files
# fields: episode_id, seed, t, rgb_path, depth_gt_path, seg_gt_path, depth_est_path,
#         pose, velocity, corridor_min_gt, corridor_min_est, action_proposed,
#         asked (bool), vlm_answer, vlm_conf, vlm_latency_real_s, delay_charged_s,
#         action_taken, collision (bool), obstacle_family

fast_stack.perceive(rgb) -> DepthEstimate          # depth map + features dict
controller.act(depth, drone_state) -> Action
gate.score(features, drone_state) -> float          # higher = more reason to ask
policy.should_ask(t, features, drone_state, history) -> bool   # all baselines share this; cooldown inside
vlm.ask(image, path_overlay, crop) -> (answer: bool, confidence: float, latency_s: float)
```

- **Run config:** one YAML per experiment (`configs/`), including policy, knob value, delay `d`, speed `v`, and the seed list name.
- **Stand-ins:** A uses ground-truth depth as a fake fast stack; B and C use DDOS images until simulator frames exist.
- **Golden set:** ~20 saved episodes created as soon as the simulator works, so perception, gate, VLM, and demo work never wait on the simulator.

---

## 7. Repository layout

```
pixel-perfect/
  PLAN.md              # this file
  README.md            # setup, versions (driver, CUDA, UE, airsim), how to reproduce
  AGENT_LOG.md         # per-member entries, merged in Week 7
  configs/             # scenarios.yaml, seeds.yaml (frozen), policies/*.yaml
  sim/                 # launch, scenario generator, obstacle spawner, kinematic drone, collision check, episode runner, forking
  perception/          # Depth Anything V2 wrapper, scale fit, features (incl. thin-structure filters)
  control/             # rule controller
  gate/                # labels (L1/L2/L3), features, models, training, calibration
  vlm/                 # server, prompts, path overlay + crop, standalone eval, surrogate judge
  policies/            # never, always, random, interval, threshold, uncertainty, scene_change, anomaly, gate, oracle
  eval/                # run_episode.py, sweep.py (resumable), metrics.py, crash_causes.py, plots.py
  demo/                # Gradio replay (reads logs only)
  hpc/                 # sbatch templates, Apptainer def, env setup, rsync scripts
  data/                # episodes/<split>/<seed>/ (gitignored; synced, two copies)
  results/             # csv + figures (generated by scripts only)
  report/              # LaTeX (Overleaf mirror)
```

Rules: figures are produced by scripts, never by hand. Sweeps append per-episode results and skip finished `(policy, seed)` pairs on restart.

---

## 8. Compute plan

**Allocation:** NYU HPC **Cloud Bursting**, **300 GPU-hours per student** (900 total), Slurm account `csci_ga_2271-2026fa`, web portal Open OnDemand (`https://ood.burst.hpc.nyu.edu/`, NYU VPN off-campus). GPUs: NVIDIA **L4** and **A100 (40 GB)**. Setup guide: Brightspace, Content, "How to access HPC Cloud Bursting". Source: course announcement, Oct 2, 2026. GCP credits (Q2) are now only a fallback.

**Course rules that shape the design:**
- **Do not use the Torch cluster.** Everything runs on Cloud Bursting.
- **20-minute idle shutdown:** nodes with low GPU use for 20 minutes are terminated. So: the VLM runs in the same job as the simulator; reading, debugging and CPU-only work (thin-structure filters, gradient boosting) happen on laptops; every long job checkpoints and resumes.

| Work | Where |
|---|---|
| Simulator rendering, closed-loop flights | Cloud Bursting **L4** via Singularity (offscreen); GCP VM only if that fails and credits exist |
| Batch perception on saved frames, gate training, VLM standalone eval | Cloud Bursting (Slurm array jobs); A100 for the 8B model |
| VLM in the loop | Same L4 job as the simulator (separate process). Not a separate server: it would sit idle between gate calls and be shut down |
| Gradio demo | Any laptop (reads logs) |

**Budget (rough; replace with measured numbers on Sun Nov 8). Open: does an A100 hour cost more than an L4 hour? (Q8)**

| Item | Hours |
|---|---|
| Development and debugging | ~200 |
| Saved-frame studies (gate training, VLM accuracy, 4 prompt variants on ~20k frames) | ~30 |
| Main flight grid (~10 policies × 3 query rates × ~300 flights, d = 1 s) | ~150 |
| Delay sweep (top 3 policies × 3 delays × 300 flights) | ~45 |
| Replay labels (~20k forks × 2 short replays) | ~35 |
| Held-out family and environment runs | ~50 |
| One full rerun after a bug (assume it happens) | ~200 |
| **Total** | **~710 of 900** |

**Throughput warning:** thousands of flights at ~60 s each is tens of hours of simulator time before VLM cost. Measure seconds per flight by Wed Nov 4. Levers, in order: faster sim clock and lower resolution, shorter corridors, paused-time VLM (D10), multiple sim instances, caching VLM answers per frame, surrogate judge for sweeps, smaller grid with honest wider CIs.

**Rules to save hours:** batch jobs, not idle interactive sessions (they also get shut down after 20 minutes); L4 unless the job needs an A100; cache model weights and VLM answers; split final runs across all three accounts by configuration.

---

## 9. Related work and our position

**Updated Oct 29 after a broader search.** Full list, search log and reading assignments: `report/related_work.md`. Everything below is from abstracts only unless marked read.

**"Learned gate instead of a fixed threshold" is not new on its own.** Learned triggers for a slow model exist in driving (AdaDrive, arXiv 2511.06253; Cognitive Dual-Process Planning, arXiv 2607.19194), in robot manipulation (EMS, arXiv 2608.06434) and, for drones, in CoDrone (arXiv 2512.19083: a reinforcement-learning scheduler decides when to offload depth estimation to an edge server).

**Claim we make instead (verify before writing):** drone systems call slow models in parallel, at fixed rates, on uncertainty thresholds (UDAV), or through a learned offloading scheduler aimed at network delay (CoDrone). We learn when a drone's fast perception is wrong badly enough that a VLM's delay is worth paying: the gate is trained on whether asking would have changed the outcome (replay labels), and delay is a physical cost that depends on speed. **CoDrone and AESOP are the papers reviewers will compare us to; read both in full first.**

| Work | What it does | How we differ |
|---|---|---|
| **CoDrone** (Chen et al., IEEE IoT Journal, [arXiv 2512.19083](https://arxiv.org/abs/2512.19083), Dec 2025) | Drone navigation; a reinforcement-learning scheduler decides when to offload depth estimation (Depth Anything V2) to an edge server; cloud VLM for open-set reasoning | **Closest drone work. Read first.** Its slow model is a depth model, not a VLM; its stated motivation is network delay; the abstract doesn't say whether its reward includes collisions (Q10) |
| **Stress-Query** (Sodre et al., LARS 2026, [arXiv 2610.23877](https://arxiv.org/abs/2610.23877)) | Always-on VLM copilots (1.8B to 11B) for UAVs in Gazebo; heavy models overcorrect because of delay; small ones hallucinate | Measures the problem; no learned gating. Our motivation, not a scoop (Q7 answered) |
| **AdaDrive** (Zhang et al., ICCV 2025, [arXiv 2511.06253](https://arxiv.org/abs/2511.06253)) | Driving; an "adaptive activation loss" learns when to call the LLM, and its influence is scaled by scene complexity | Learned trigger, but driving, and trained on a scene-complexity proxy rather than outcome |
| **EMS** (Sun et al., [arXiv 2608.06434](https://arxiv.org/abs/2608.06434), Aug 2026) | VLA manipulation; a small switching policy trained with reinforcement learning picks the small or large model each step | Learned trigger via RL; our replay labels are a supervised alternative |
| **Cognitive Dual-Process Planning** (Yang et al., [arXiv 2607.19194](https://arxiv.org/abs/2607.19194), Jul 2026) | Driving; a lightweight classifier on vision-encoder features routes scenes to fast or slow reasoning, with an extra penalty for sending hard scenes to the fast path | Closest in mechanism to our gate-L1; reproduce as a baseline |
| **Astra** (ByteDance Seed, [arXiv 2506.06205](https://arxiv.org/abs/2506.06205)) and **Helix 02** ([Figure, Jan 2026](https://www.figure.ai/news/helix-02)) | Hierarchical System 1/2 robots; Helix 02 adds a 1 kHz "System 0" (System 1 runs at 200 Hz) | Framing: our System 2 = VLM on demand, System 1 = fast stack, System 0 = flight controller we abstract |
| **AESOP** (Sinha et al., RSS 2024, [arXiv 2407.08735](https://arxiv.org/abs/2407.08735)) | Fast anomaly detector triggers slow LLM reasoning on drones; accounts for LLM delay | Triggers on *unusual scenes*; we predict *fast-perception mistakes* and learn from consequences |
| **Slow Brain, Fast Planner** ([arXiv 2606.20458](https://arxiv.org/abs/2606.20458), June 2026) | VLM picks among planner proposals at a fixed 1 Hz; ground robots | Lists uncertainty-gated asking as future work; that's our project |
| **UDAV** ([arXiv 2609.16368](https://arxiv.org/abs/2609.16368), Sept 2026) | Heuristic dispersion threshold triggers VLM rethink of ground-vehicle routes | Heuristic, open-loop, no call budget or crash rate |
| Triggering Generalist Reasoning via Predictive Uncertainty (Yang et al., [arXiv 2610.05025](https://arxiv.org/abs/2610.05025)) | Uncertainty-gated heavy model in dual-system VLA | Manipulation; check numbers before citing |
| Dual-process indoor navigation ([arXiv 2601.21506](https://arxiv.org/abs/2601.21506)) | Calls VLM on scene change | Used as our scene-change baseline |
| Think Only When Needed ([arXiv 2608.23224](https://arxiv.org/abs/2608.23224)) | Selective slow path for manipulation | Delay has no physical cost there |
| Matched-budget gating in RL ([arXiv 2609.29548](https://arxiv.org/abs/2609.29548)) | When LM advice is worth buying, equal budgets | We borrow equal-budget comparison |
| Dynamic Selection of Perception Models (Ghosh et al., [arXiv 2207.06390](https://arxiv.org/abs/2207.06390)) | Switches perception models at run time in AirSim | No VLM; check overlap |
| When to Act, Ask, or Learn ([arXiv 2602.22474](https://arxiv.org/abs/2602.22474)); runtime failure monitoring ([arXiv 2410.04640](https://arxiv.org/abs/2410.04640)); FARM ([arXiv 2609.11445](https://arxiv.org/abs/2609.11445)) | Uncertainty-aware asking and failure monitoring | Background for the gate |
| KnowNo (Ren et al., CoRL 2023); Perceive With Confidence ([arXiv 2403.08185](https://arxiv.org/abs/2403.08185)); learnable conformal prediction ([arXiv 2509.21955](https://arxiv.org/abs/2509.21955)) | Conformal guarantees for asking and perception | Stretch: conformal threshold for the gate |
| Learning to defer (Madras et al. 2018; Mozannar & Sontag 2020) | Theory of handing off to an expert | We add a physical cost of asking |
| Introspective Perception (Daftry et al., IROS 2016) | Predicts vision failures for drones in forests | Closest earlier idea; no VLM to ask |
| DDOS (CVPR-W 2024, [arXiv 2312.12494](https://arxiv.org/abs/2312.12494)); UCorr ([arXiv 2509.14989](https://arxiv.org/abs/2509.14989)) | Thin obstacles break monocular depth | Motivates our failure cases |
| Eyes Wide Shut / MMVP, Cambrian-1, Thinking in Space (Tong et al.; Yang et al.) | VLM visual blind spots (instructor's group) | Motivates the VLM failure analysis and crop/path drawing |

---

## 10. Experiments

**E1. Pipeline sanity (Week 2):** true-depth flight crashes near zero; replays with a fixed seed match exactly; collision checker agrees with geometry.

**E2. Saved-frame studies (Week 3; fallback paper on its own):**
- Gate quality (AUROC, AUPRC, ECE, risk–coverage) vs. uncertainty and threshold baselines, per obstacle family, including the held-out family.
- VLM accuracy per obstacle type and distance; ablations: crop on/off, path drawing on/off, 2B vs. 8B, 2 prompt variants.

**E3. Main closed-loop grid (Week 4):** all policies at 3 matched query rates, `d = 1 s`, `v = 5 m/s`, val-tuned, test seeds once. Output: the headline curve + crash-cause breakdown.

**E4. Replay labels (Week 5, headline 1):** gate-L3 vs. gate-L1 at equal query rates.

**E5. Delay as a controlled variable (Week 5, headline 2):** top 3 policies × `d ∈ {0.25, 1, 3} s` (and speeds 3/5/8 m/s if budget allows). Shows when asking stops being worth it.

**E6. Generalization:** gate trained without one obstacle family and one environment, tested on them. Learned gates often lose to plain uncertainty on unfamiliar scenes; either outcome is reportable.

**E7. Slow-system comparison (D18):** the same gates with (a) the VLM yes/no judge and (b) the VLM choosing an action directly; (c) a flow-matching or diffusion policy only if the MVP review approves it. Question: does the best asking policy depend on what the slow system is?

**Ablations (pick 2 to 3 by Week 4):** gate features (depth-only / + thin-structure filters / + embedding); cooldown on/off; brake-and-hold vs. speed-limited waiting (D11); SAM 2 features; VLM 2B vs. 8B.

**Stretch:** conformal threshold that bounds missed hazards, and where that promise breaks (e.g. fog); commercial VLM comparison (budget first); DAgger-style round using gated rollouts.

---

## 11. Risks, kill switches, and backups

### Ranked blockers

| Rank | Blocker | Trigger | Backup chain |
|---|---|---|---|
| 1 | Simulator install / headless rendering | No frame captured by end of **Sat Oct 31** | Singularity on a Cloud Bursting L4 (AirSim or Colosseum binary, offscreen) → Xvfb/VirtualGL/EGL render path → GCP VM if credits exist → **Plan B simulator** |
| 2 | Flight throughput | sec/flight × planned flights exceeds budget (measure **Wed Nov 4**) | Faster clock, lower res, shorter corridors → paused-time VLM → parallel instances → cached VLM answers / surrogate → smaller grid |
| 3 | Closed-loop integration | never-ask fails on wall-only scenes | Fixed altitude + discrete actions → kinematic stepping (D4) → pause-step-resume loop |
| 4 | Thin wires don't render or collide | Not visible at 10 m or no collision by **Mon Nov 2** | Thicker / high-contrast variants → segmentation IDs for ground truth → own collision check (D5) → other hard cases (poles vs. sky, glass, fog) |
| 5 | No signal for the gate | never-ask crash rate outside 10 to 80% | Retune difficulty mix → degrade fast stack deliberately (document it) → add failure modes |
| 6 | VLM can't see thin wires | Wire recall < ~60% at relevant distance (**Fri Nov 6**) | Crop/zoom, prompt variants, 8B → another open VLM → depth overlay in prompt → surrogate judge |
| 7 | Depth scale pollutes labels | Positives correlate with distance, not obstacles | Metric checkpoint → frozen global scale fit → labels from segmentation ground truth |
| 8 | VLM serving: latency/memory | > 2 s per call or OOM | vLLM, 2B, quantization, one-token output → A100 for 8B → precomputed answers (not a separate idle server: 20-minute shutdown) |
| 9 | Gate doesn't beat the tuned threshold | AUROC ≈ depth-variance baseline | Fix labels, then features (temporal, thin-structure, embedding) → L2 labels → calibration → **report honestly with crash causes** |
| 10 | HPC queues, credits, access, idle shutdown | Test job pending > a day, or jobs killed by the 20-minute idle rule | Submit tiny jobs first, short checkpointed jobs, keep the GPU busy or end the job, Colab/Kaggle for VLM eval as last resort |
| 11 | Late label bugs / leakage | Odd label rates | Unit tests, seed-list checksums, re-runnable label scripts (< 1 hour to regenerate) |
| 12 | Report and demo underestimated | Writing starts Week 7 | Skeleton by Week 3; figures as scripts; static GIF replay if Gradio slips |

### Kill switches

| By | Check | If not met |
|---|---|---|
| Sat Oct 31 | A frame is captured from the simulator on a Cloud Bursting L4 | Start Plan B simulator full-time (one person), keep fixing in background |
| **Mon Nov 2 (go/no-go)** | Simulator chosen; wires visible and collidable, or alternate hard cases chosen | Commit to fallback; no more simulator debating |
| Wed Nov 4 | Never-ask closed loop runs; sec/flight measured; budget written | Plan B simulator + kinematic stepping |
| Fri Nov 6 | VLM (2B) recall on wires acceptable | Backup chain 6 (8B, crop/zoom, prompts); surrogate already exists |
| Mon Nov 9 | Gate-L1 beats depth-variance baseline on val | Chain 9; continue and report honestly |
| Wed Nov 11 | MVP: all MVP policies run end-to-end; first curve | Cut grid, freeze scope |
| Mon Nov 30 | Final test runs launched | No new features; analysis and writing only |

**Plan B simulator:** a lightweight scene with raycast depth (Open3D/trimesh) and simple shading, PyBullet, or Habitat-sim with a free-flying camera. Loses photorealism, gains control, speed, and laptop/HPC portability. Skeleton built Sun Nov 1 to Mon Nov 2 as insurance (≈ half a day). If used, the report says so plainly. **A complete simpler study beats a stalled realistic one.**

### Insurance to create now
Singularity image (or VM image) after the simulator works; daily push with tags at each milestone; two copies of all episode data (VM + HPC scratch, final test set also to Drive); lockfiles and versions in README; cached weights; frozen seeds and configs; golden set of ~20 episodes; Plan B skeleton; surrogate judge.

---

## 12. Team and ways of working

| | **A: Simulator and flights** | **B: Fast stack and gate** | **C: VLM and evaluation** |
|---|---|---|---|
| Name | _assign_ | _assign_ | _assign_ |
| Owns | scenarios, obstacles, kinematic drone, collision check, paused-time delay loop, forking/replays, sweeps, HPC/VM jobs, videos | depth model and scale, controller, features, labels L1/L2/L3, gate models, uncertainty baselines, conformal stretch | VLM serving, prompts, path drawing and crop, latency measurement, VLM accuracy, scene-change and anomaly baselines, surrogate judge, metrics, statistics, figures, Gradio demo |
| Stand-in until others deliver | ground-truth depth as fake fast stack | DDOS images, then golden set | DDOS images, then golden set |
| Hands off | flight logs, `run_episode.py`, `sweep.py` | `perception/`, `control/`, `gate/` | `vlm/`, `eval/`, `demo/` |
| Paper sections | Simulator and setup | Method | Experiments and analysis |

**Rituals**
- **Daily:** 10-minute sync (blockers only).
- **Weekly (Sunday):** 1-hour code walkthrough; each person walks the other two through their code. Required because anyone can be asked about any part at the final presentation.
- **Commits:** small, daily, one shared repo, a branch per workstream merged at least weekly. One bulk commit at the deadline is not accepted.
- **Agent Log:** each person logs what the AI tools did, what they got wrong (bugs, false claims, wrong approaches), what was fixed by hand, and how it was checked. Items 2 and 3 carry the most weight in Agent Track grading. Merged in Week 7.
- **Individual contributions** in the report come from the table above.

---

## 13. Timeline

> **Re-dated Oct 29 (D19).** Sprint 1 had not started by Oct 29, so it now runs Thu Oct 29 to Wed Nov 11. Final test runs still launch Mon Nov 30 and the report week doesn't move, so Sprint 2 is one week shorter. If late, cut in this order: speed sweep in E5, commercial VLM, conformal stretch, D18(c) diffusion/flow-matching slow policy, hardware demo (D20).

### Sprint 1 (Thu Oct 29 to Wed Nov 11): MVP

Goal by **Wed Nov 11**: the drone flies closed-loop through the full pipeline; never / always / random / threshold / gate-L1 / oracle and true-depth all run; first crash-vs-calls curve on val seeds.

**Thu Oct 29 to Fri Oct 30: unblock**
- [ ] All: assign workstream owners (Q1); repo skeleton (Section 7); interface stubs and metric definitions (Sections 5, 6; S-04)
- [ ] All: one tiny test job each on Cloud Bursting (account `csci_ga_2271-2026fa`); scratch env; cache weights (Depth Anything V2-Small, Qwen3-VL-2B and 8B)
- [ ] All: start reading CoDrone and AESOP in full (S-03); notes in `report/related_work.md`
- [ ] A: L4 session on Open OnDemand; download a prebuilt AirSim/Colosseum binary; Singularity image
- [ ] B: Depth Anything V2-Small (metric) on DDOS: accuracy and FPS on an L4; check A100 vs. L4 cost (Q8)
- [ ] C: serve Qwen3-VL-2B and 8B; measure latency; email the TA (Q3, Q6)

**Sat Oct 31: first simulator frame (kill switch)**
- [ ] A: binary renders offscreen in Singularity on an L4; Python script saves RGB + GT depth + segmentation
- [ ] B: controller on DDOS / GT depth frames; draft L1 label function with unit tests
- [ ] C: first VLM accuracy on DDOS wires, with and without crop

**Sun Nov 1 to Mon Nov 2: GO/NO-GO (Mon)**
- [ ] A: obstacle spawner (wires, poles, glass-like, fog); wires visible and collidable via own check; kinematic drone; scenario generator; **freeze seeds.yaml**; Plan B skeleton (insurance)
- [ ] B: fast-stack choice finalized (Prof. Xie's next step): short comparison on DDOS of thin-wire recall and FPS on an L4, Depth Anything V2-Small vs. one or two alternatives
- [ ] C: path-overlay + crop renderer; `policy.should_ask` interface with never/always/random; surrogate judge
- [ ] All: simulator decision (D3, D17) recorded in the Decision log; Sun: first weekly walkthrough

**Tue Nov 3 to Wed Nov 4: first closed loop**
- [ ] Never-ask and true-depth flights over ≥100 val episodes
- [ ] Measure sec/flight; write the episode budget into README
- [ ] Headroom check: never-ask crash rate in 15 to 30%; retune difficulty if not
- [ ] Save the golden set (~20 episodes); Singularity image saved
- [ ] C: evaluation harness (metrics, CIs, results table)

**Thu Nov 5 to Fri Nov 6: data + VLM**
- [ ] A: collect 300 to 500 never-ask episodes on train seeds with action noise (near-crash states); paused-time delay loop with the surrogate
- [ ] B: features + L1 labels; label sanity (eyeball 20+20); gradient boosting baseline (on a laptop: CPU-only)
- [ ] C: VLM standalone on ~200 labeled sim frames per obstacle type (**kill switch Fri**)

**Sat Nov 7 to Tue Nov 10: gated loop**
- [ ] B: gate-L1 MLP; AUROC/AUPRC/ECE vs. depth-variance baseline (**kill switch Mon Nov 9**)
- [ ] C: threshold(τ) and oracle policies; cooldown; real VLM reachable from the loop
- [ ] A: all MVP policies smoke-tested on 10 val flights each; first sweep at 2 to 3 operating points
- [ ] Sun Nov 8: walkthrough; replace compute estimates in Section 8 with measured numbers

**Wed Nov 11: milestone review**
- [ ] One-page memo: first curve, crash causes on ~30 crashes, decisions for Sprint 2 (including D18(c) and D20)
- [ ] Agent Log entries up to date; tag `mvp-v1`

### Sprint 2 (Nov 11 to Dec 6)

| Week | A: Simulator and flights | B: Fast stack and gate | C: VLM and evaluation | Together |
|---|---|---|---|---|
| **Nov 11 to 15** | Flights at scale; flight forking for replays | Gate on sim frames: full E2 study; uncertainty baseline; held-out family | VLM ablations (crop, path, 2B vs. 8B, prompts); D18(b) action-choosing VLM; scene-change and anomaly baselines | **Checkpoint: saved-frame results (fallback paper).** Report skeleton started |
| **Nov 16 to 22** | Main grid E3 at `d = 1 s` | Uncertainty and learned gates in flights; L2 and L3 labels | Crash-cause tooling; per-policy statistics | Pick final ablations |
| **Nov 23 to 29** (Thanksgiving Nov 26) | Delay sweep E5; held-out environment | Gate-L3 (E4); E6 generalization; conformal stretch | E7 slow-system comparison; first full figures | Walkthrough; paper sections drafted |
| **Nov 30 to Dec 6** | **Final test runs launched Mon Nov 30**, split across all three accounts; crash-vs-catch videos; repo cleanup | Final gate results; Method section final | CI tables, crash breakdown, failure gallery; Gradio replay; Experiments section final | Assemble 4 to 9 page LaTeX report + slides; each person explains the other two's code |

---

## 14. Verification checklist (where the Agent Track grade comes from)

- [ ] `python eval/run_episode.py --policy never --seed 0` flies and logs; the same seed reproduces the same flight exactly.
- [ ] Collision checker tested against known obstacle geometry, including thin wires.
- [ ] True-depth flight gives near-zero crashes (pipeline is correct).
- [ ] Label functions unit-tested; ~200 gate labels and ~200 VLM answers hand-checked.
- [ ] Gate beats the depth-variance baseline on held-out seeds, or the report explains why not.
- [ ] VLM standalone precision/recall reported before any gated flight result.
- [ ] `python eval/sweep.py` regenerates the headline CSV and figure from saved logs; rerunning gives identical numbers.
- [ ] Test seeds used once; thresholds tuned on val only.
- [ ] Every surrogate-judge result labeled and validated by a real-VLM run.
- [ ] Gradio (or GIF) replay loads a saved flight and shows frame, depth, gate score, VLM marker, with no simulator running.
- [ ] Every cited paper read in full and its claims checked.
- [ ] Each member can explain every module (weekly walkthroughs).
- [ ] Agent Log kept from day one; small, frequent commits.

---

## 15. Deliverables, presentation, and publication

**Course deliverables:** 4 to 9 page LaTeX report (track restated, individual contributions, Agent Log appendix), reproducible repo, final presentation with code walkthrough.

**Presentation framing (from the meeting with Prof. Xie):** show the idea and the findings; don't anchor on beating baselines. Negative results are explicitly valued. Baselines stay in the report for rigor.

**Presentation arc:** (1) side-by-side video: fast stack flies into a wire; the gate catches it and the VLM confirms; (2) the headline curve; (3) crash-cause breakdown; (4) replay labels and the delay sweep; (5) does the gate still help when the slow system changes (D18); (6) where the small VLM itself fails (ties to the instructor's work on VLM visual blind spots); (7) hardware demo clip, only if D20 is approved.

**Publication path:** the December report is about workshop-paper scope. January to February: more environments, possibly one real-flight clip; target a 2027 workshop (CVPR or ICRA) or IROS 2027 / RA-L. Confirm all deadlines in November; none are verified here.

---

## 16. Open questions

| # | Question | Owner | Due |
|---|---|---|---|
| Q1 | Who takes workstreams A, B, C? | All | Fri Oct 30 |
| Q2 | Were GCP credits approved? (Now only a fallback, D17) | C | Fri Oct 30 |
| Q3 | Does the TA accept simulation-only evaluation? (Prof. Xie said simulator-first is fine; confirm for grading) | C | Fri Oct 30 |
| Q4 | Simulator binary: AirSim, Colosseum, or Project AirSim (verify repo and license), or Plan B? | A | Mon Nov 2 |
| Q5 | ~~Are the 300 HPC hours GPU-hours?~~ **Answered:** yes, 300 GPU-hours per student on Cloud Bursting, L4 and A100 (40 GB); Torch forbidden; 20-minute idle shutdown. Still check scratch quota | B | Closed |
| Q6 | Exact final presentation and report dates? | C | Fri Oct 30 |
| Q7 | ~~Does Stress-Query already do learned gating for UAVs?~~ **Answered (abstract):** no, it evaluates always-on copilots. The closest drone work is CoDrone (Q10) | All | Closed |
| Q8 | Does an A100 hour cost more than an L4 hour on Cloud Bursting? | B | Fri Oct 30 |
| Q9 | Meeting notes: exact meeting date, and confirm "Astra" meant ByteDance's Astra (arXiv 2506.06205) | All | Fri Oct 30 |
| Q10 | CoDrone (arXiv 2512.19083): what are its scheduler's inputs and reward? Does it account for collisions or thin obstacles? If yes, revisit D16 | All | Mon Nov 2 |

---

## Appendix A. Backlog and parked items (nothing gets lost)

Everything from earlier drafts and discussions that isn't in the main plan lives here. Status: **Backlog** = do if time allows, **Fallback** = use if the main path fails, **Parked** = deliberately not doing now, with the reason. To pull an item into the plan, add a Decision-log row.

### A1. Simulator and environments

| Item | Status | Notes |
|---|---|---|
| OpenFly platform (prebuilt AirSim/UE scenes on HuggingFace, MIT code) | Fallback / Backup scenes | Its install needs sudo + ROS 2; GTA V scenes need the paid game and Windows; its instruction generator needs an OpenAI key. Use only its scene files if our own scenes are too plain |
| Colosseum (community AirSim fork) | Fallback (rank 1 chain) | Same Python API. Officially no Ubuntu 22.04 support; use Docker |
| Project AirSim (IAMAI) | Fallback | Claimed open-source under MIT; **verify repo and license** before relying on it (Q4) |
| Full flight dynamics + PX4 flight controller | Parked | We study perception, not control; adds tuning time and breaks exact replays |
| ROS 2 integration | Parked | Not needed for the study; likeliest setup time sink |
| Habitat-sim with a free-flying camera | Fallback (Plan B) | Headless, pip/conda installable, ground-truth depth |
| PyBullet or raycast scene (Open3D/trimesh) | Fallback (Plan B) | Runs on laptop and HPC arrays; low realism |
| Isaac Sim / Pegasus | Parked | Heavier setup; reconsider only for a post-course real-to-sim extension |
| Other aerial benchmarks: AerialVLN, CityNav, OpenUAV, OpenFly VLN tasks ([awesome-uav-vln list](https://github.com/earth-insights/awesome-uav-vln)) | Parked | Language-guided navigation, not our task; source of extra scenes if needed |
| Multiple simulator instances on one VM (different ports) | Backlog | Throughput lever (rank 2) |
| Speed sweep 3/5/8 m/s | Backlog | Part of E5 if budget allows |

### A2. Fast stack and perception

| Item | Status | Notes |
|---|---|---|
| SAM 2 segmentation | Backlog (ablation) | Run on downsampled frames or every k-th frame; smaller variant if slow. Mask count/area/fragmentation as gate features |
| Object detector (YOLO-class) alongside depth | Backlog | Optional extra fast-stack signal from the research plan |
| Depth Anything V2 relative checkpoint + scale alignment | Fallback | If the metric checkpoint is poor in our scenes |
| Scale from known altitude / ground plane online | Fallback | Rank 7 chain |
| Deliberately degraded fast stack (lower resolution, noise, smaller backbone) | Fallback | Only if never-ask rarely crashes; document it |
| Depth ensemble / test-time augmentation for uncertainty | Backlog | Feeds the uncertainty baseline |

### A3. Gate

| Item | Status | Notes |
|---|---|---|
| Conformal risk control for τ (bound missed hazards) | Backlog (stretch) | KnowNo, Perceive With Confidence, learnable conformal prediction; show where it breaks (fog) |
| DAgger-style data from gated rollouts | Backlog (stretch) | Fixes distribution shift between never-ask data and gated flights |
| Temperature / isotonic calibration | Backlog | Rank 9 chain |
| Explicit cost-aware score (expected crash cost × speed vs. delay cost) instead of learning cost through L3 | Backlog | Possible method variant if L3 is weak |
| False-stop labels (reverse of L1) as a second gate target | Backlog | Logged from Week 1 anyway |

### A4. VLM judge

| Item | Status | Notes |
|---|---|---|
| Commercial VLM comparison | Backlog | Costs money and adds network latency; budget first |
| Alternate open VLMs: InternVL, Qwen2.5-VL, LLaVA-OneVision | Fallback | If Qwen3-VL can't see wires |
| Larger VLM on HPC (bigger than 8B) | Fallback | Rank 6 chain |
| Depth overlay in the prompt | Fallback | Rank 6 chain |
| Alternate question "Is the path ahead clear?" | Fallback | Earlier draft wording; current default is path drawing (D9) |
| Majority vote over several answers | Fallback | Trades latency for reliability |
| "Always ask every k frames" variant of always-ask | Backlog | Cheaper upper reference |
| Asynchronous VLM (planner keeps flying, stale answers fused, as in Slow Brain, Fast Planner) | Parked | Different design; a possible extra baseline if reviewers ask |

### A5. Data

| Item | Status | Notes |
|---|---|---|
| DDOS ([HuggingFace](https://huggingface.co/datasets/benediktkol/DDOS), CC BY-NC 4.0, 300 flights / 30k train images) | Backlog (sanity test + week-1 stand-in) | Thin wires, weather; offline only |
| Mid-Air ([download](https://midair.ulg.ac.be/download.html), CC BY-NC-SA 4.0) | Parked | Large (hundreds of GB), domain gap; download only a few trajectories if ever needed |
| TartanAir ([site](https://theairlab.org/tartanair-dataset/), CC BY 4.0) | Parked | Good depth, but segmentation classes are random IDs |
| UCorr wire data/model ([arXiv 2509.14989](https://arxiv.org/abs/2509.14989)) | Backlog | Possible extra thin-wire baseline or test set |
| Hardware demo / real-flight clip | Stretch (D20) | Decided at the MVP review; otherwise Jan to Feb for the publication version |

### A6. Evaluation, demo, and writing

| Item | Status | Notes |
|---|---|---|
| Gradio replay demo | In plan (Week 6) | Static GIF/MP4 per flight as fallback |
| Crash-vs-catch side-by-side videos | In plan (Week 6) | Opening slide |
| Layman pitch text (driver asking an experienced passenger at confusing intersections) | Backlog | Reuse for slides, poster, and the intro paragraph |
| Paper Lantern feasibility note: "prototype, medium confidence" for a 2-week full system | Context | From the 14-day draft; consistent with an MVP-first plan |
| Unverified grounding numbers from the 14-day draft: uncertainty-gated heavy-model calls cut ~75% at 86.7% success ([arXiv 2610.05025](https://arxiv.org/abs/2610.05025)); calibrated safety margins cut obstacle misdetection 93% ([arXiv 2403.08185](https://arxiv.org/abs/2403.08185)) | Verify before citing | Came from a search tool summary |
| AirHunt ([arXiv 2601.12742](https://arxiv.org/html/2601.12742v1)) | Related work to read | VLM semantics + continuous planning for aerial object search |
| Workshop/venue deadlines (CVPR 2027 workshops, ICRA 2027 workshops, IROS 2027, RA-L) | Backlog | C checks in November |

### A7. Team and process

| Item | Status | Notes |
|---|---|---|
| 4-person split (C splits into "VLM" and "eval + paper") | Parked | If the team grows |
| 2-person split (A: sim, controller, eval, sweeps; B: perception, VLM, gate) | Fallback | If someone is unavailable for a stretch |
| Two-week plan doc (Claude doc + Word/PDF) | Superseded | This file replaces it |

---

## Appendix B. Implementation notes

**Simulation is the core of the project.** Every result (training data, labels, baselines, the headline curve) comes from closed-loop flights in the simulator. Note the naming: our course **track** is Agent; "simulation" is our evaluation environment, not a track.

**AirSim status:** Microsoft archived AirSim in 2022 (read-only, no fixes), but the binaries and the Python package (`pip install airsim`) still work and are widely used. Colosseum is the maintained community fork with the same API.

**Useful AirSim API calls:** `simSpawnObject` (obstacles), `simSetVehiclePose` (kinematic stepping), `simGetImages` (RGB, depth, segmentation), `simPause` / `simContinueForTime` (paused-time VLM delay), `ClockSpeed` setting (faster simulation), `moveByVelocityAsync` (only if we switch to dynamics).

**Rendering:** run headless with `-RenderOffScreen` or Xvfb (VirtualGL/EGL if needed). Snapshot the VM image once it works.

**HPC setup (Cloud Bursting; first day of work):**
1. Follow Brightspace, Content, "How to access HPC Cloud Bursting" (Open OnDemand, interactive GPU sessions, Conda/Singularity). Slurm account `csci_ga_2271-2026fa`. Never the Torch cluster.
2. `sinfo` for partitions and GPU types (L4, A100 40 GB); check storage and scratch quota.
3. Conda env and `HF_HOME` on scratch, not home; pre-download Depth Anything V2-Small, Qwen3-VL-2B and 8B (and SAM 2 for the ablation).
4. One reusable `hpc/job_template.sbatch` with `--account=csci_ga_2271-2026fa`, `--gres=gpu:1` and `--array`; jobs checkpoint because of the 20-minute idle shutdown.
5. Singularity definition in `hpc/` for the simulator.

**Data flow:**
```
simulator machine (VM or HPC) --episode logs--> storage
storage --rsync--> HPC scratch --SLURM--> features, labels, gate training, VLM eval
gate.pt --rsync--> simulator machine --> gated flights --> logs --> analysis + demo
```

**Licenses:** Unreal Engine is free for students and research; DDOS and Mid-Air are non-commercial licenses (fine for coursework, matters for any commercial follow-up); Qwen3-VL open weights are Apache 2.0 for the sizes we use (confirm on the model card).
