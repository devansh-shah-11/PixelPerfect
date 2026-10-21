# Pixel Perfect: Master Plan

**Working title:** *Ask Only When It Matters: Latency-Aware Deferral to Vision-Language Models for Safe Drone Flight*
**Course:** CSCI-GA 2271 Computer Vision, NYU, Fall 2026
**Track:** Agent (declared in the submitted proposal; locked) · **Format:** Research project
**Team:** Pixel Perfect, 3 members · **Version:** v1.0, October 20, 2026

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
| D3 | Simulator: prebuilt **AirSim/Colosseum binary via Python API only. No ROS 2, no PX4, no OpenFly unless required** | DEFAULT, final by Fri Oct 23 | ROS 2 and sudo were the likeliest week-one sink; nothing in the study needs them |
| D4 | Drone motion: **kinematic stepping** (set pose each step), not full flight dynamics | DEFAULT | Deterministic replays (needed for replay labels), faster, no controller-tuning sink. We study perception, not control |
| D5 | Collisions: **own check against known obstacle geometry** (capsule/raycast), engine flag only as a cross-check | LOCKED | Thin wires may not trigger engine colliders |
| D6 | Obstacles: **spawned programmatically** with controlled type, thickness, color, distance, lighting, fog | LOCKED | Full control of difficulty; no asset hunting |
| D7 | Fast stack: **Depth Anything V2 (Small, metric checkpoint)** + rule controller. **SAM 2 is not in the MVP**; it's an ablation only | DEFAULT | Cheaper and simpler; the question is "is something in the path," not object identity |
| D8 | VLM judge: **Qwen3-VL 4B and 8B**, open weights, served on NYU HPC (or VM) | DEFAULT | Reproducible, no per-call cost. Commercial VLM comparison is optional and must be budgeted |
| D9 | VLM question: **flight path drawn on the image + full-resolution crop of the corridor**, single yes/no token | DEFAULT | VLMs judge distance poorly; downscaling erases 1 to 2 px wires |
| D10 | Latency: **simulator pauses during the VLM call; then simulated time advances by delay `d` while the drone executes the waiting behavior. `d` is never zero** | LOCKED | Results independent of our GPU, exactly repeatable, lets us sweep delay. Real latency reported separately |
| D11 | Waiting behavior: **MVP = brake and hold**; Week 4+ = **speed-limited waiting** (ablation) | DEFAULT | Simple and safe first; research plan's slow-down behavior tested once the loop is stable |
| D12 | Data: **simulator is the training and test set**. DDOS only as an offline sanity test (depth and VLM on thin wires) and as a stand-in before the simulator works. Mid-Air dropped | DEFAULT | Open-loop datasets have a large domain gap |
| D13 | Splits by **scenario seed**, never by frame; train/val/test seed lists frozen by Sat Oct 24 | LOCKED | Prevents leakage |
| D14 | Thresholds and operating points tuned on **val** only; **test** seeds run once at the end | LOCKED | Prevents test leakage |
| D15 | Every policy runs on the **same scenarios and seeds** (paired comparison) | LOCKED | Fewer flights for the same statistical power |
| D16 | Two research headlines: **(1) replay labels, (2) delay as a controlled variable**. VLM failure analysis is a smaller third result | DEFAULT | Both are only possible in simulation; keeps scope realistic for 3 people in 7 weeks |
| D17 | Compute: **HPC for offline GPU work** (perception, gate, VLM eval, and the simulator if Apptainer rendering works); **GCP VM for the simulator** otherwise | OPEN, decide by Fri Oct 23 | HPC has no sudo or display; Apptainer is a time-boxed try |

**Changes from the submitted proposal** (state these in the report): no ROS 2/OpenFly by default (D3); SAM 2 moved to ablation (D7); simulator data replaces DDOS/Mid-Air as the main data (D12); kinematic drone (D4).

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
- Qwen3-VL 4B and 8B via vLLM or HF, 4/8-bit if memory-bound, separate process from the simulator.
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

## 5. Metrics and statistics (locked by Thu Oct 22)

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

## 6. Interfaces (agreed by Thu Oct 22; stand-ins until real parts arrive)

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

**Allocation:** NYU HPC, 300 hours per member (900 total; confirm whether these are GPU-hours, Q5). GCP credits: status unknown (Q2).

| Work | Where |
|---|---|
| Simulator rendering, closed-loop flights | GCP GPU VM (≥24 GB, Ubuntu 22.04), **or** HPC via Apptainer if the Day 2 time-boxed try works |
| Batch perception on saved frames, gate training, VLM standalone eval | HPC (SLURM, array jobs) |
| VLM in the loop | Same machine as the simulator (separate process) or HPC server over SSH tunnel; pick one and document it |
| Gradio demo | Any laptop (reads logs) |

**Budget (rough; replace with measured numbers on Mon Nov 2)**

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

**Throughput warning:** thousands of flights at ~60 s each is tens of hours of simulator time before VLM cost. Measure seconds per flight by Mon Oct 26. Levers, in order: faster sim clock and lower resolution, shorter corridors, paused-time VLM (D10), multiple sim instances, caching VLM answers per frame, surrogate judge for sweeps, smaller grid with honest wider CIs.

**Rules to save hours:** batch jobs, not idle interactive sessions; cache model weights and VLM answers on scratch; shut the VM down when idle; split final runs across all three accounts by configuration.

---

## 9. Related work and our position

**Gap we claim (verify before writing):** no work we found trains a gate to predict *perception* mistakes and decide when to ask a VLM, in closed-loop flight where the drone's actions change what it sees next, with the VLM's delay treated as a physical cost. The area is moving fast; re-check before the paper.

| Work | What it does | How we differ |
|---|---|---|
| **Stress-Query and Temporal Reliability of VLMs as UAV Copilots** ([arXiv 2610.23877](https://arxiv.org/html/2610.23877), Oct 2026) | VLM copilots for UAV navigation | **Not yet read. Possibly the closest. Read first (Week 1)** |
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
- VLM accuracy per obstacle type and distance; ablations: crop on/off, path drawing on/off, 4B vs. 8B, 2 prompt variants.

**E3. Main closed-loop grid (Week 4):** all policies at 3 matched query rates, `d = 1 s`, `v = 5 m/s`, val-tuned, test seeds once. Output: the headline curve + crash-cause breakdown.

**E4. Replay labels (Week 5, headline 1):** gate-L3 vs. gate-L1 at equal query rates.

**E5. Delay as a controlled variable (Week 5, headline 2):** top 3 policies × `d ∈ {0.25, 1, 3} s` (and speeds 3/5/8 m/s if budget allows). Shows when asking stops being worth it.

**E6. Generalization:** gate trained without one obstacle family and one environment, tested on them. Learned gates often lose to plain uncertainty on unfamiliar scenes; either outcome is reportable.

**Ablations (pick 2 to 3 by Week 4):** gate features (depth-only / + thin-structure filters / + embedding); cooldown on/off; brake-and-hold vs. speed-limited waiting (D11); SAM 2 features; VLM 4B vs. 8B.

**Stretch:** conformal threshold that bounds missed hazards, and where that promise breaks (e.g. fog); commercial VLM comparison (budget first); DAgger-style round using gated rollouts.

---

## 11. Risks, kill switches, and backups

### Ranked blockers

| Rank | Blocker | Trigger | Backup chain |
|---|---|---|---|
| 1 | Simulator install / headless rendering | No frame captured by end of **Thu Oct 22** | Colosseum or another prebuilt map → Xvfb/VirtualGL/EGL render path → Apptainer on HPC (≤ half day) → **Plan B simulator** |
| 2 | Flight throughput | sec/flight × planned flights exceeds budget (measure **Mon Oct 26**) | Faster clock, lower res, shorter corridors → paused-time VLM → parallel instances → cached VLM answers / surrogate → smaller grid |
| 3 | Closed-loop integration | never-ask fails on wall-only scenes | Fixed altitude + discrete actions → kinematic stepping (D4) → pause-step-resume loop |
| 4 | Thin wires don't render or collide | Not visible at 10 m or no collision by **Fri Oct 23** | Thicker / high-contrast variants → segmentation IDs for ground truth → own collision check (D5) → other hard cases (poles vs. sky, glass, fog) |
| 5 | No signal for the gate | never-ask crash rate outside 10 to 80% | Retune difficulty mix → degrade fast stack deliberately (document it) → add failure modes |
| 6 | VLM can't see thin wires | Wire recall < ~60% at relevant distance (**Wed Oct 28**) | Crop/zoom, prompt variants, 8B → another open VLM → depth overlay in prompt → surrogate judge |
| 7 | Depth scale pollutes labels | Positives correlate with distance, not obstacles | Metric checkpoint → frozen global scale fit → labels from segmentation ground truth |
| 8 | VLM serving: latency/memory | > 2 s per call or OOM | vLLM, 4B, quantization, one-token output → separate GPU/process → precomputed answers |
| 9 | Gate doesn't beat the tuned threshold | AUROC ≈ depth-variance baseline | Fix labels, then features (temporal, thin-structure, embedding) → L2 labels → calibration → **report honestly with crash causes** |
| 10 | HPC queues, credits, access | Test job pending > a day | Submit tiny jobs Day 1, short checkpointed jobs, Colab/Kaggle for VLM eval as last resort |
| 11 | Late label bugs / leakage | Odd label rates | Unit tests, seed-list checksums, re-runnable label scripts (< 1 hour to regenerate) |
| 12 | Report and demo underestimated | Writing starts Week 7 | Skeleton by Week 3; figures as scripts; static GIF replay if Gradio slips |

### Kill switches

| By | Check | If not met |
|---|---|---|
| Thu Oct 22 | A frame is captured from the simulator | Start Plan B simulator full-time (one person), keep fixing in background |
| **Fri Oct 23 (go/no-go)** | Simulator chosen; wires visible and collidable, or alternate hard cases chosen | Commit to fallback; no more simulator debating |
| Mon Oct 26 | Never-ask closed loop runs; sec/flight measured; budget written | Plan B simulator + kinematic stepping |
| Wed Oct 28 | VLM recall on wires acceptable | Backup chain 6; surrogate by Mon Nov 2 |
| Sat Oct 31 | Gate-L1 beats depth-variance baseline on val | Chain 9; continue and report honestly |
| Tue Nov 3 | MVP: all MVP policies run end-to-end; first curve | Cut grid, freeze scope |
| Mon Nov 30 | Final test runs launched | No new features; analysis and writing only |

**Plan B simulator:** a lightweight scene with raycast depth (Open3D/trimesh) and simple shading, PyBullet, or Habitat-sim with a free-flying camera. Loses photorealism, gains control, speed, and laptop/HPC portability. Skeleton built Sat Oct 24 to Sun Oct 25 as insurance (≈ half a day). If used, the report says so plainly. **A complete simpler study beats a stalled realistic one.**

### Insurance to create now
VM image after the simulator works; daily push with tags at each milestone; two copies of all episode data (VM + HPC scratch, final test set also to Drive); lockfiles and versions in README; cached weights; frozen seeds and configs; golden set of ~20 episodes; Plan B skeleton; surrogate judge.

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

### Sprint 1 (Wed Oct 21 to Tue Nov 3): MVP

Goal by **Tue Nov 3**: the drone flies closed-loop through the full pipeline; never / always / random / threshold / gate-L1 / oracle and true-depth all run; first crash-vs-calls curve on val seeds.

**Wed Oct 21 (Day 1): unblock**
- [ ] All: repo skeleton (Section 7), AGENT_LOG.md, read the Stress-Query paper and AESOP
- [ ] All: confirm HPC partitions/GPU types (`sinfo`), scratch envs, cache weights; GCP credit status
- [ ] A: VM or HPC node with working GPU + Vulkan/OpenGL; download prebuilt AirSim/Colosseum binary
- [ ] B: Depth Anything V2-Small (metric) on DDOS; accuracy and FPS
- [ ] C: serve Qwen3-VL 4B and 8B; measure latency; ask TA about sim-only evaluation

**Thu Oct 22 (Day 2): simulator + interfaces**
- [ ] A: Python script flies a waypoint, saves RGB + GT depth + segmentation; Apptainer try on HPC (≤ half day)
- [ ] All: agree interfaces (Section 6) and metric definitions (Section 5)
- [ ] B: controller on DDOS / GT depth frames; draft L1 label function with unit tests
- [ ] C: first VLM accuracy on DDOS wires, with and without crop

**Fri Oct 23 (Day 3): GO/NO-GO**
- [ ] A: obstacle spawner (wires, poles, glass-like, fog); wires visible and collidable via own check
- [ ] All: simulator decision (D3, D17) recorded in the Decision log

**Sat Oct 24 to Sun Oct 25 (Days 4 to 5)**
- [ ] A: kinematic drone, scenario generator, **freeze seeds.yaml**; Plan B skeleton (insurance)
- [ ] B: fast stack + controller in the loop on GT depth first (should almost never crash)
- [ ] C: path-overlay + crop renderer; evaluation harness (metrics, CIs, results table); `policy.should_ask` interface with never/always/random
- [ ] Sun: first weekly walkthrough

**Mon Oct 26 to Tue Oct 27 (Days 6 to 7): first closed loop**
- [ ] Never-ask and true-depth flights over ≥100 val episodes
- [ ] Measure sec/flight; write the episode budget into README
- [ ] Headroom check: never-ask crash rate in 15 to 30%; retune difficulty if not
- [ ] Save the golden set (~20 episodes); VM image snapshot

**Wed Oct 28 to Fri Oct 30 (Days 8 to 10): data + VLM**
- [ ] A: collect 300 to 500 never-ask episodes on train seeds with action noise (near-crash states)
- [ ] B: features + L1 labels on HPC; label sanity (eyeball 20+20); gradient boosting baseline
- [ ] C: VLM standalone on ~200 labeled sim frames per obstacle type (**kill switch Wed**); delay loop (pause, advance by `d`, brake-and-hold)

**Sat Oct 31 to Mon Nov 2 (Days 11 to 13): gated loop**
- [ ] B: gate-L1 MLP; AUROC/AUPRC/ECE vs. depth-variance baseline (**kill switch Sat**)
- [ ] C: threshold(τ) and oracle policies; cooldown; surrogate judge implemented
- [ ] A: all MVP policies smoke-tested on 10 val flights each; first sweep at 2 to 3 operating points
- [ ] Sun: walkthrough; Mon: replace compute estimates in Section 8 with measured numbers

**Tue Nov 3 (Day 14): milestone review**
- [ ] One-page memo: first curve, crash causes on ~30 crashes, decisions for Sprint 2
- [ ] Agent Log entries up to date; tag `mvp-v1`

### Sprint 2 onward (Weeks 3 to 7)

| Week | A: Simulator and flights | B: Fast stack and gate | C: VLM and evaluation | Together |
|---|---|---|---|---|
| **3** Nov 3 to 9 | Flights at scale; flight forking for replays | Gate on sim frames: full E2 study; uncertainty baseline; held-out family | VLM ablations (crop, path, 4B/8B, prompts); scene-change and anomaly baselines | **Checkpoint: saved-frame results (fallback paper).** Report skeleton started |
| **4** Nov 10 to 16 | Main grid E3 at `d = 1 s` | Plug uncertainty and learned gates into flights; L2 labels | Crash-cause tooling; per-policy statistics | Pick final ablations |
| **5** Nov 17 to 23 | Delay and speed sweep E5; held-out environment | Replay labels L3 and gate-L3 (E4); conformal stretch | First full figures; commercial VLM only if budgeted | Walkthrough |
| **6** Nov 24 to 30 (Thanksgiving Nov 26) | Final test runs split across all three accounts; crash-vs-catch videos | Final gate results; E6 generalization | CI tables, crash breakdown, failure gallery; Gradio replay | Each person drafts their paper sections and Agent Log |
| **7** Dec 1 to 7 | Repo cleanup; reproduction instructions | Method section final | Experiments section final | Assemble 4 to 9 page LaTeX report + slides; each person explains the other two's code |

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

**Presentation arc:** (1) side-by-side video: fast stack flies into a wire; the gate catches it and the VLM confirms; (2) the headline curve; (3) crash-cause breakdown; (4) replay labels and the delay sweep; (5) where the VLM itself fails (ties to the instructor's work on VLM visual blind spots).

**Publication path:** the December report is about workshop-paper scope. January to February: more environments, possibly one real-flight clip; target a 2027 workshop (CVPR or ICRA) or IROS 2027 / RA-L. Confirm all deadlines in November; none are verified here.

---

## 16. Open questions

| # | Question | Owner | Due |
|---|---|---|---|
| Q1 | Who takes workstreams A, B, C? | All | Wed Oct 21 |
| Q2 | Were GCP credits approved? | C | Wed Oct 21 |
| Q3 | Does the TA accept simulation-only evaluation? | C | Fri Oct 23 |
| Q4 | Simulator: AirSim binary, Colosseum, Project AirSim (verify the repo and license before relying on it), or Plan B? | A | Fri Oct 23 |
| Q5 | Are the 300 HPC hours GPU-hours? GPU types, partitions, Apptainer availability, scratch quota? | B | Wed Oct 21 |
| Q6 | Exact final presentation and report dates? | C | Fri Oct 23 |
| Q7 | Does the Stress-Query paper already do learned gating for UAVs? If yes, revisit D16 | All | Fri Oct 23 |

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
| Real-flight clip (Jan to Feb) | Backlog (post-course) | For the publication version |

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

**HPC setup (Day 1):**
1. `sinfo`, `sacctmgr show assoc user=$USER` for partitions, GPU types, limits.
2. Conda/venv on scratch, not home; `HF_HOME` on scratch; pre-download Depth Anything V2, Qwen3-VL 4B/8B (and SAM 2 for the ablation).
3. One reusable `hpc/job_template.sbatch` with `--gres=gpu:1` and `--array`.
4. Apptainer definition in `hpc/` for the time-boxed simulator try.

**Data flow:**
```
simulator machine (VM or HPC) --episode logs--> storage
storage --rsync--> HPC scratch --SLURM--> features, labels, gate training, VLM eval
gate.pt --rsync--> simulator machine --> gated flights --> logs --> analysis + demo
```

**Licenses:** Unreal Engine is free for students and research; DDOS and Mid-Air are non-commercial licenses (fine for coursework, matters for any commercial follow-up); Qwen3-VL open weights are Apache 2.0 for the sizes we use (confirm on the model card).
