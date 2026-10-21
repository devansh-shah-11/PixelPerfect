# Project Plan: "Ask the VLM Only When Unsure"

**Gated fast/slow perception for a simulated drone**
Window: **14 days, Tue Oct 20 → Tue Nov 3, 2026**
Team assumption: 1–2 people. Evaluation: simulation only.
Compute: **NYU HPC (access confirmed)** + one GCP GPU VM for the simulator.

---

## 0. TL;DR

- Fast stack (Depth Anything V2 + SAM 2 + rule controller) flies a simulated drone every frame.
- A small learned **gate** predicts when the fast stack is wrong (labels come free from simulator ground truth).
- Only when the gate fires do we ask **Qwen3-VL** a yes/no question ("is there a thin obstacle in the path?").
- Headline result: **crash rate vs. VLM calls/episode** for never / always / random / hand-tuned threshold / learned gate / oracle gate.
- Paper Lantern feasibility verdict: **PROTOTYPE, medium confidence** (80 papers surveyed). The concept is realistic; a polished full system in 2 weeks is not. The plan is built so a working end-to-end loop exists by **Day 5** and everything after is an upgrade.
- **The biggest risk is simulator + closed-loop integration, not the ML.** The ranked blockers, deadlines, and fallback chains are in **§12**.

### Research grounding
- Uncertainty-gated invocation of a heavy model cut heavy calls ~75% at 86.7% success (arxiv:2610.05025, Yang et al. – Paper Lantern, 119 papers surveyed).
- Safety margins calibrated on ground truth reduced obstacle misdetection 93% (arxiv:2403.08185, Mei et al.).
- Related: arxiv:2602.22474 (when to act/ask/learn), arxiv:2410.04640 (runtime failure monitoring), arxiv:2509.21955 (learnable conformal prediction), arxiv:2207.06390 (dynamic perception-model selection, AirSim), arxiv:2609.11445 (FARM failure readout).

---

## 1. Changes from the original proposal (and why)

| # | Proposal said | Plan does | Reason |
|---|---|---|---|
| 1 | OpenFly/AirSim + ROS 2 | **AirSim Python API only, no ROS 2** | ROS 2 + sudo is the likeliest week-one time sink; nothing in the study needs it. |
| 2 | Thin obstacles from datasets | **Spawn thin cylinders/boxes programmatically** (`simSpawnObject`) | No asset hunting; full control over thickness, color, distance. |
| 3 | DDOS + Mid-Air as datasets | **Simulator data is the training set**; DDOS only as an optional offline sanity check | Both are open-loop and have a large domain gap to AirSim; big downloads. |
| 4 | OpenFly | **Optional**, verify it installs first; default to plain AirSim (or community fork, since Microsoft's repo is archived) | OpenFly is a VLN benchmark platform with its own setup cost. |
| 5 | HPC for training + VLM | **HPC = offline GPU work; GCP VM = simulator** (see §2) | HPC has no sudo/display; simulator rendering needs a GPU VM you control. |

---

## 2. Compute allocation (HPC access is confirmed)

| Task | Where | Why |
|---|---|---|
| AirSim / Unreal rendering, closed-loop episodes | **GCP GPU VM** (L4/A10, ≥24 GB, Ubuntu 22.04) | Needs display/Vulkan/OpenGL stack and sudo. |
| Optional: simulator on HPC via **Apptainer/Singularity** container (headless UE, `-RenderOffScreen`) | HPC (stretch) | Would let the sweeps run in parallel as SLURM array jobs. Try on Day 2–3 as a *time-boxed (≤ half day)* experiment; abandon if Vulkan/GL is not available on compute nodes. |
| Batch perception inference on saved frames (Depth Anything V2, SAM 2) | **HPC** (SLURM GPU job) | Embarrassingly parallel over saved episodes. |
| Gate training (MLP / GBM), hyperparameter search, calibration | **HPC** | Small, but fast iteration and seeds-in-parallel. |
| Qwen3-VL standalone evaluation on labeled frames (Day 9) | **HPC** (A100/H100 preferred) | Needs large GPU memory; many frames. |
| Qwen3-VL inside the closed loop | Preferred: **same GCP VM** (separate process) or a **local HTTP server on HPC** reached over SSH tunnel | Latency must be measured and modeled explicitly (see §6.5). Keep one choice and document it. |
| Gradio replay demo | Laptop / any machine (reads saved logs) | No simulator needed. |

**Practical HPC setup (Day 1):**
1. Confirm account, allocation/partition names, GPU types (`sinfo`, `sacctmgr show assoc user=$USER`).
2. Create conda/venv in scratch (not home, to avoid quota); cache HF models to scratch (`HF_HOME`).
3. Write one reusable SLURM template (`hpc/job_template.sbatch`) with `--gres=gpu:1`, time limit, `--array` support.
4. Pre-download weights once: Depth Anything V2 (metric), SAM 2, Qwen3-VL (4B and 8B).
5. Sync code with git; sync episode logs with `rsync` between VM, HPC and laptop. Keep a single canonical `data/` layout (§5).

**Data flow:**
```
GCP VM (AirSim) --episode logs (rgb, gt depth, seg, collision, controls)--> storage
      storage --rsync--> HPC scratch --SLURM--> perception features, gate labels, gate training, VLM eval
      trained gate.pt --rsync--> GCP VM --> closed-loop gated episodes --> logs --> HPC/laptop analysis + Gradio
```

---

## 3. Scope

**Must have (MVP):**
- Closed-loop sim with thin-obstacle scenarios
- Fast stack (depth + rule controller)
- Gate trained from simulator ground truth
- Qwen3-VL yes/no judge in the loop
- 4 baselines + learned gate + oracle, crash-vs-query-rate curve
- Short report

**Should have:** SAM 2 in the fast stack, 2–3 ablations, Gradio replay.

**Cut first if late:** DDOS/Mid-Air, latency-realism extras, conformal threshold calibration, DAgger-style data aggregation, OpenFly, HPC-hosted simulator.

---

## 4. Metric definitions (lock on Day 1; do not change later)

- **Crash rate:** fraction of episodes with a collision before reaching the goal or hitting the time limit.
- **VLM calls/episode** and **query rate** (calls per frame).
- **Latency:** per-frame control latency (mean, p95) and total episode time including VLM holds.
- **Safety retained (headline):** `(crash_never − crash_gate) / (crash_never − crash_always)`.
- **Gate quality:** AUROC, AUPRC, ECE on held-out seeds.
- **Task success / time-to-goal** (to show the drone isn't "safe" by never moving).
- **Confidence intervals:** Wilson or bootstrap on all rates.
- **Oracle gate:** queries exactly when label `y = 1`; gives the upper bound on gating benefit.

---

## 5. Repository layout

```
project/
  PLAN.md
  README.md
  configs/            # yaml: scenarios, policies, seeds, thresholds
  sim/                # airsim launch, scenario generator, obstacle spawner, episode runner
  perception/         # depth (DA-V2), segmentation (SAM 2), feature extraction
  control/            # rule-based controller
  gate/               # labels, features, models, training, calibration
  vlm/                # Qwen3-VL wrapper, prompts, standalone eval
  policies/           # never, always, random, threshold, gate, oracle (one interface)
  eval/               # run_episode.py, sweep.py, metrics.py, plots.py, failure_analysis.py
  demo/               # Gradio replay app
  hpc/                # sbatch templates, env setup, sync scripts
  data/               # episodes/<split>/<seed>/ (rgb, depth_gt, seg_gt, depth_est, log.jsonl)
  results/            # csv, figures
  report/
```

Seed lists (train / val / test) live in `configs/seeds.yaml` and are never edited after Day 3.

---

## 6. Day-by-day plan

### Phase A – Foundations: closed loop, no ML yet (Days 1–5)

**Day 1 (Tue Oct 20) – unblock everything**
- Spin up GCP VM + credits; verify GPU driver, Vulkan/OpenGL.
- HPC: account check, env, SLURM template, pre-download weights (see §2).
- Decide AirSim vs fork; check whether OpenFly is required by the course.
- Create repo skeleton (§5); write metric definitions (§4) into README.
- Ask the instructor: is sim-only evaluation acceptable?
- Done when: VM reachable, HPC job runs `nvidia-smi`, repo exists.

**Day 2 – simulator running**
- Install a **prebuilt** AirSim binary (don't compile Unreal from source); run headless/virtual-display.
- Python API: fly a waypoint, capture RGB + depth + segmentation + collision flag.
- Time-boxed (≤ half day): try the same binary inside Apptainer on HPC.
- Done when: script flies and saves frames.

**Day 3 – scenarios**
- Scenario generator: random start/goal corridor; obstacle types = wall/box (easy), pole, **thin wire/cable** (hard), optional low-texture/glass-like surface; vary thickness, color, distance, lighting.
- Collision ends the episode. Fixed seeds → deterministic layouts.
- Freeze train/val/test seed lists.
- Done when: 100 reproducible seeded episodes generate.

**Day 4 – fast stack v1**
- Depth Anything V2 (**metric checkpoint**, or per-frame scale alignment) on the VM GPU.
- Rule controller: forward-corridor min-depth → stop / yaw / climb, with hysteresis.
- Done when: "never ask" flies, and crashes mostly on wires (this is the intended failure mode; if it doesn't crash on wires, make wires thinner or lower contrast).

**Day 5 – SAM 2 + logging + checkpoint**
- Add SAM 2 (downsampled frame or every k-th frame; smaller variant if too slow). If it doesn't pay for itself, keep it as an ablation and don't block.
- Per-timestep logging: RGB, estimated depth, GT depth, controller command, GT collision-imminent flag, timestamps.
- Measure baseline crash rate over ≥100 episodes.
- **CHECKPOINT:** closed loop not running by end of Day 5 → switch to Plan B (§9).

### Phase B – Gate (Days 6–8)

**Label design (most important decision)**
- `y = 1` if the fast stack's decision would be unsafe given ground truth: e.g. min GT depth in the forward corridor < `d_safe` while estimated min depth > `d_safe` (missed obstacle). Optionally also the reverse (false stop).
- Alternate label `y_future` = collision within next k seconds (compare both).
- Define labels by **control consequence**, not raw depth error.

**Data**
- ~300–500 "never ask" episodes plus randomized perturbations (to reach near-crash states).
- Split **by scenario seed**, never by frame. Hold out one obstacle family (e.g. one wire thickness) for a generalization test.
- Run batch perception + feature extraction on HPC (SLURM array over episodes).

**Features**
- Depth stats in forward corridor (min, percentiles, gradient/edge density, local variance).
- SAM mask count / area / fragmentation.
- Controller state.
- Optional: frozen Depth Anything encoder embedding (the "learned" variant).

**Training (HPC)**
- Start with gradient boosting on handcrafted features as a strong reference; then small MLP; then MLP + embedding.
- Class-weighted BCE; threshold selected from validation PR/ROC.
- Report AUROC, AUPRC, ECE.

**Sanity gate before closing the loop:** AUROC clearly above a trivial baseline (depth variance alone). If not, fix labels/features first, not the model.

### Phase C – VLM judge + gated loop (Days 9–11)

**Day 9 – VLM alone (HPC)**
- Qwen3-VL (4B and 8B) on ~200 labeled frames. Fixed prompt: *"Is there a thin obstacle (wire, pole, cable) in the drone's forward path within N meters? Answer yes or no."* Single-token output; use logit(yes) − logit(no) to allow thresholding.
- Report precision/recall on obstacle-in-path, by obstacle type and distance. Try crop/zoom of the forward corridor and 2 prompt variants.
- **If the VLM can't see thin wires, find out now**, not on Day 13; then adjust (zoom, larger model, prompt) or reframe the claim.

**Day 10 – integrate**
- VLM server (separate process) reachable from the simulator loop.
- **Latency model:** on a query the drone brakes/holds for the measured VLM latency (or sim is paused and latency charged to wall-clock cost). Choose one, document it, never let latency equal zero.
- VLM "yes" → avoidance maneuver; "no" → continue with fast stack.

**Day 11 – policies behind one interface**
- `never`, `always`, `random(p)`, `threshold(τ)` (hand-tuned on min-depth or depth variance), `gate(τ)`, `oracle`.
- **Cooldown** (min frames between queries) to prevent query storms.
- Smoke-test each policy on 10 val episodes.

### Phase D – Experiments (Days 11–13)

- Sweep each policy's knob (p, τ) on val seeds to choose operating points; evaluate on **test seeds** only once.
- Target: 3 seeds × ≥50 episodes per point (scale to compute; parallelize with SLURM arrays if the simulator runs on HPC, otherwise batch on the VM).
- Outputs: crash-rate vs. VLM-calls plot with all policies + oracle; per-obstacle-type crash rate; latency; task success.
- **Ablations (pick 2–3):** gate features (depth-only / +SAM / +embedding); cooldown on/off; VLM 4B vs 8B; held-out obstacle family.
- **Failure analysis:** categorize ~30 crashes — gate miss / VLM miss / controller issue / latency.
- Fairness: tune the hand-tuned threshold on the same validation seeds with equal effort; it is the strongest baseline.

### Phase E – Demo and write-up (Days 13–14)

- **Gradio replay:** dropdown of saved episodes; frame slider showing RGB, estimated vs. GT depth, gate score, VLM answer markers. Reads logs only.
- **Report:** method, definitions, curve, ablations, failure analysis, limitations (sim-only, sim-to-real gap).
- Freeze code, logs, seeds, configs.
- **Day 14 is buffer. No new work.**

### Milestone checks
| Day | Demo-able state |
|---|---|
| 5 | Closed loop flies; baseline crash rate |
| 8 | Trained gate with held-out AUROC |
| 11 | Full pipeline with all policies |
| 13 | Final plots + failure analysis |

### Two-person split (if applicable)
- **A:** simulator, scenarios, controller, eval harness, sweeps.
- **B:** perception models, VLM prompt/eval, gate training on HPC (can start from Day 3 on saved frames, before the loop is perfect).
- 10-minute daily sync.

---

## 7. Challenges, merits, demerits, mitigations

### 7.1 Simulator setup (highest risk)
- **Challenge:** AirSim archived; large UE binaries; headless GPU rendering on cloud VMs is fiddly; sudo/ROS 2 constraint from the proposal.
- **Merit:** Python API is simple; GT depth, segmentation, collision are built in, giving free labels.
- **Demerit:** dated tooling, occasional crashes/desyncs.
- **Mitigation:** prebuilt binary; skip ROS 2; `-RenderOffScreen` or Xvfb; save a VM image once it works; one OS/driver combo; Plan B at Day 5.

### 7.2 Fast perception stack
- **Challenge:** Depth Anything V2 outputs *relative* depth by default; wrong scale looks like "failure" and pollutes gate labels. SAM 2 automatic masking is slow.
- **Mitigation:** metric checkpoint or per-frame scale alignment; log scale error separately; run SAM 2 on downsampled or every k-th frame. All three models on one GPU may OOM — keep Qwen3-VL in a separate process, and use ≥24 GB.

### 7.3 Gate labels and learning
- **Challenge:** class imbalance; ambiguous "wrong" (depth error that doesn't matter); covariate shift between "never ask" data and states visited under gated policies.
- **Merit:** labels are exact and free; the supervised problem is small and fast.
- **Demerit:** gate may learn scenario artifacts and fail on new obstacle types.
- **Mitigation:** consequence-based labels; seed-level splits; held-out obstacle family; compare to oracle; DAgger-style round from gated rollouts as a stretch; conformal-style threshold calibration (arxiv:2403.08185, arxiv:2509.21955) as a cheap upgrade.

### 7.4 Datasets (DDOS, Mid-Air)
- **Challenge:** open-loop, large domain gap, big downloads.
- **Recommendation:** don't depend on them. Use simulator data; optionally DDOS for an offline thin-obstacle sanity test of depth/VLM; otherwise state honestly in the report that they were not used.

### 7.5 VLM judge
- **Challenge:** small VLMs can be poor on thin structures and distance judgments; prompt sensitivity; latency of ~0.3–2 s is long compared with drone dynamics.
- **Merit:** a strong demonstration of the trade-off if the VLM catches wires.
- **Mitigation:** evaluate alone first (Day 9); crop/zoom; prompt variants; logit thresholding; 4B vs 8B; explicit hold/brake latency model.

### 7.6 Evaluation validity
- **Challenge:** sim-to-real gap; small episode counts → noisy crash rates; threshold tuning on test seeds (leakage).
- **Mitigation:** fixed train/val/test seeds; confidence intervals; baselines tuned as carefully as the gate. If the gate only matches the tuned threshold, say so and analyze why — that is still a valid result.

### 7.7 Compute and access
- **Challenge:** HPC has no sudo/display, queue waits, quota limits.
- **Mitigation:** simulator on the VM; HPC for offline jobs (and optionally container-based sim as a time-boxed experiment); store caches on scratch; shut down the VM when idle; back up logs.

### 7.8 Time and team
- **Challenge:** sequential dependencies (sim → data → gate → VLM loop).
- **Mitigation:** Day 5 / 8 / 11 checkpoints; parallel work split; buffer Day 14.

---

## 8. What "good" looks like

- One figure: crash rate (y) vs. VLM calls/episode (x) for never / always / random / threshold / learned gate / oracle.
- Headline claim: *"the gate keeps X% of always-ask safety with Y% of the queries,"* with confidence intervals.
- A modest X, or a competitive hand-tuned threshold, is acceptable provided the failure analysis explains why.

---

## 9. Plan B (trigger: no closed loop by end of Day 5)

Replace AirSim with a lightweight simulator that still provides GT depth: PyBullet, or a custom scene with raycast depth (Open3D/trimesh) and simple shading; or another off-the-shelf navigation sim. You lose photorealism (the depth model and VLM will behave differently) but gain full control, no GPU-render dependencies, and fast iteration (runs on a laptop and in HPC SLURM arrays). State clearly in the report that it is a toy-realism simulation. A complete weak study beats a stalled realistic one.

---

## 10. Verification checklist

1. `python eval/run_episode.py --policy never --seed 0` flies and logs collision or success; the same seed reproduces the same result.
2. Gate: held-out-seed AUROC reported and beats a depth-variance-only baseline.
3. VLM alone: precision/recall on labeled frames reported before gated experiments.
4. `python eval/sweep.py` produces the crash-vs-query-rate CSV and plot from saved logs; re-running yields identical numbers.
5. Gradio demo loads a saved episode and shows frame, depth, gate score and VLM marker with no simulator running.

---

## 11. Open questions (resolve on Day 1)

- Team size?
- GCP credits approved? (HPC: confirmed.)
- Does the instructor accept sim-only evaluation?
- Is OpenFly required, or is plain AirSim fine?
- HPC specifics: GPU types, partitions, time limits, whether Apptainer is available, scratch quota.

---

## 12. Time sinks, blockers, and backups

Ranked by expected days lost (probability × cost). Each item has a **trigger** (how you know it's happening), a **decision deadline** (when you stop and switch), and an ordered **backup chain** (try A, then B, then C).

### Ranking at a glance

| Rank | Blocker | Prob. | Days at risk | Decision deadline |
|---|---|---|---|---|
| 1 | Simulator install / headless rendering | High | 1–3 | End of Day 2 (soft), Day 5 (hard) |
| 2 | Episode throughput (too many episodes, sim runs slow) | High | 1–3 | Day 5 (measure sec/episode) |
| 3 | Closed-loop integration + controller tuning | High | 1–2 | Day 5 |
| 4 | Thin wires not rendered / not in GT depth / no collision | Medium | 1–2 | Day 3 |
| 5 | Fast stack fails never or always (no signal for gate) | Medium | 1 | Day 5 |
| 6 | VLM cannot see thin wires | Medium–High | 1–2 | Day 9 |
| 7 | Depth scale (relative vs. metric) pollutes labels | Medium | 0.5–1 | Day 4 |
| 8 | VLM in the loop: latency, serving, memory | Medium | 1 | Day 10 |
| 9 | Gate doesn't beat the hand-tuned threshold | Medium | 0.5–1 | Day 8 |
| 10 | SAM 2 too slow / OOM | Medium | 0.5 | Day 5 |
| 11 | Queue waits, credits, access delays | Medium | 0.5–1 | Day 1 |
| 12 | Label bugs / data leakage found late | Low–Medium | 1 | Day 8 |
| 13 | Report/demo underestimated | High | 0.5–1 | Day 11 (start drafting) |

**The single most time-consuming part is not any one model. It is the simulator-in-the-loop work (ranks 1–4): install, getting thin obstacles to behave, making episodes fast enough, and wiring perception to control.** Everything downstream waits on it. Plan accordingly: start it first, time-box it, and use the backups early.

---

### B1. Simulator install / headless rendering (rank 1)
- **Why it hurts:** AirSim is archived; Unreal binaries are multi-GB; cloud VMs need a working GPU + Vulkan/OpenGL + display stack; version mismatches between UE, driver, and `airsim` Python package.
- **Trigger:** no frame captured by end of Day 2.
- **Backup chain:**
  - **A.** Try the other prebuilt option: community fork (Colosseum) binary, or a different prebuilt environment (Blocks → another map). Pin driver/CUDA version to a known-good combo.
  - **B.** Change the render path: `-RenderOffScreen`, Xvfb + VirtualGL, or an EGL-capable headless setup. Keep a screenshot-based smoke test.
  - **C.** Run the binary in Apptainer on HPC (time-boxed ≤ half day) — only if the VM route is dead.
  - **D. Plan B simulator (§9):** PyBullet / raycast scene. Develop this **skeleton in parallel from Day 3 as insurance** (≈ half a day), so switching costs hours, not days. Optional alternative: Habitat-sim (headless, pip/conda-installable, GT depth), using a free-flying camera as the "drone".
- **Rule:** if A and B both fail by end of Day 3, commit to D and don't look back.

### B2. Episode throughput (rank 2) — easy to underestimate
- **Why it hurts:** 3 seeds × 50 episodes × ~6 policies × ~5 operating points ≈ **4,500 episodes**. At 60 s each that is **~75 hours** of sim time, before VLM latency.
- **Trigger:** measured sec/episode on Day 5 × planned episodes > ~25 GPU-hours of available VM time.
- **Backup chain:**
  - **A.** Speed up the sim clock (AirSim `ClockSpeed`), lower render resolution, shorter corridors/time limits.
  - **B.** Pause the sim while perception/VLM runs (`simPause`), so wall-clock cost doesn't distort physics, and charge latency as simulated time instead.
  - **C.** Run multiple sim instances on one VM (ports) or a second VM; if the container route worked, SLURM arrays on HPC.
  - **D.** Cut the grid: 30 episodes/point, 3 operating points per policy, drop one baseline sweep (keep never/always as endpoints), and report wider CIs honestly.
  - **E.** Cache perception outputs per frame and reuse across policies where trajectories coincide (e.g. before the first query).
- **Day 5 action:** compute the episode budget explicitly and write it into README.

### B3. Closed-loop integration + controller tuning (rank 3)
- **Why it hurts:** drone dynamics oscillate; collisions flagged late/early; perception latency makes the loop unstable; non-deterministic physics breaks seed reproducibility.
- **Trigger:** "never ask" drone doesn't reach the goal on easy (wall-only) scenes.
- **Backup chain:**
  - **A.** Simplify control: fly at fixed altitude and constant forward speed; controller only chooses {continue, stop, yaw left/right, climb}.
  - **B.** Switch to **kinematic stepping** (set pose each step with `simSetVehiclePose`) instead of full flight dynamics; collision checks via sim collision flag or own raycast. Loses realism, gains determinism and speed.
  - **C.** Pause-step-resume loop (sim paused while models run) to remove the latency/stability coupling.
  - **D.** Use the AirSim built-in `moveByVelocityAsync` with short durations rather than custom low-level control.

### B4. Thin wires don't behave (rank 4)
- **Why it hurts:** wires thinner than a few pixels may not render, may vanish from GT depth, or may not trigger collisions (mesh/collider issues); the whole study depends on them being visible to a human but hard for the model.
- **Trigger:** by Day 3, wire not visible in RGB at 10 m, or drone passes through it without a collision flag.
- **Backup chain:**
  - **A.** Increase thickness until visible and collidable (e.g. 2–5 cm → 5–10 cm), use high-contrast vs. low-contrast variants; render at higher camera resolution.
  - **B.** Use segmentation IDs for obstacles so GT "obstacle present" doesn't depend on the depth buffer.
  - **C.** Own collision test: raycast / capsule-distance check between drone pose and spawned obstacle geometry (we know the geometry), independent of the engine.
  - **D.** Replace wires with other "hard for depth" cases: poles against bright sky, glass-like/transparent panels, low-texture white walls, fog. State this in the report; the study is about *fast-perception failure*, not wires specifically.

### B5. No signal for the gate (rank 5)
- **Why it hurts:** if the fast stack never fails, there's nothing to gate; if it fails everywhere, always-ask is the only answer.
- **Trigger:** "never ask" crash rate < ~10% or > ~80% on the mixed scenario set.
- **Backup chain:**
  - **A.** Retune the difficulty mix (more/fewer wires, thickness, distance, speed).
  - **B.** Degrade the fast stack deliberately: lower input resolution, lighting/fog/noise augmentation, smaller depth backbone. Document this honestly.
  - **C.** Add other failure modes (reflective/low-texture surfaces) so there's variety to learn.

### B6. VLM cannot see thin wires (rank 6)
- **Why it hurts:** if Qwen3-VL's recall on wires is near chance, "always ask" isn't safer, and the premise fails.
- **Trigger:** Day 9 standalone eval: recall < ~60% at relevant distances.
- **Backup chain:**
  - **A.** Crop/zoom the forward corridor, upsample, try 2 prompt variants, use logit thresholding, try 8B (or larger on HPC).
  - **B.** Try another open VLM (Qwen2.5-VL, InternVL, LLaVA-OneVision) as the judge; keep Qwen3-VL as the headline only if it works.
  - **C.** Ask a different question: "Is the path ahead clear?" with an annotated/zoomed region, or give the VLM the depth overlay too.
  - **D. Last resort — surrogate judge:** a *noisy oracle* with TPR/FPR measured from the real VLM on labeled frames, used for sweeps. Clearly label results as "surrogate VLM"; keep a smaller real-VLM run as validation. This preserves the gating study even if the VLM is weak.

### B7. Depth scale pollutes labels (rank 7)
- **Why it hurts:** relative depth with the wrong scale looks like "perception failure" everywhere, teaching the gate to detect scale error, not missed obstacles.
- **Trigger:** gate labels positive on most frames, or positives correlate with distance rather than obstacle type.
- **Backup chain:**
  - **A.** Use the metric Depth Anything V2 checkpoint (outdoor/indoor as appropriate).
  - **B.** Fit a single global affine map (scale+shift) from estimated to GT depth on the train split, and apply it frozen at test time (no GT at runtime).
  - **C.** Define labels on *obstacle presence in corridor* (via segmentation GT) rather than absolute distance error.
  - **D.** Use the ground-plane / known altitude to set scale online.

### B8. VLM in the loop: latency, serving, memory (rank 8)
- **Trigger:** a VLM call takes > ~2 s, or OOM when all models are loaded.
- **Backup chain:**
  - **A.** Serve with vLLM or HF with 4B, 4-bit/8-bit quantization; one-token output; resized image.
  - **B.** Separate process for the VLM; keep depth/SAM on the other GPU or run SAM 2 less often.
  - **C.** Sweeps use the surrogate judge (B6-D) or **precomputed VLM answers on stored frames** when trajectories coincide; run the real VLM in the loop only for final test runs.
  - **D.** Pause the sim during the VLM call and charge latency as simulated time (B2-B).

### B9. Gate doesn't beat the hand-tuned threshold (rank 9)
- **Trigger:** Day 8 AUROC ≈ depth-variance baseline, or Day 12 curve overlaps the threshold policy.
- **Backup chain:**
  - **A.** Fix labels first (consequence-based), then features: add temporal features (last k frames, depth change), SAM stats, encoder embedding.
  - **B.** Switch label to `y_future` (collision within k seconds).
  - **C.** Gradient boosting instead of MLP; calibrate (temperature/isotonic); pick τ by conformal-style coverage (arxiv:2403.08185, arxiv:2509.21955).
  - **D.** Add DAgger-style data from gated rollouts to fix covariate shift.
  - **E.** Accept and report: "learned gate ≈ tuned threshold" plus the failure analysis is a legitimate result.

### B10. SAM 2 too slow / OOM (rank 10)
- **Backup chain:** downsample + every k-th frame → smaller SAM 2 variant → keep it as an ablation only → replace with cheap line/edge features (Canny / Hough / Frangi vesselness), which suit thin structures and cost almost nothing.

### B11. Queue waits, credits, access (rank 11)
- **Backup chain:** submit HPC test jobs on Day 1 (short, small) to learn partition limits; request GCP credits / free-trial VM on Day 1; keep a local (Mac) dev path via Plan B simulator and CPU-sized models; Colab/Kaggle GPU as a last-resort sandbox for VLM eval; use short checkpointed jobs so preemption costs minutes.

### B12. Label bugs / leakage found late (rank 12)
- **Prevention:** unit tests for label function on hand-made frames (Day 6); split by seed with a checksum of the seed lists; plot label rate vs. distance/obstacle type; look at 20 random positives and 20 random negatives by eye.
- **Backup:** keep the data-collection script re-runnable on HPC so labels/features can be regenerated in under an hour.

### B13. Report and demo (rank 13)
- **Prevention:** start the report skeleton on Day 8 (method + metric definitions are already settled); write figures as scripts, so re-running is one command; Gradio demo reads logs only and can be built on Day 12 from any saved episodes.
- **Backup:** if time runs out, ship a static replay (saved GIF/MP4 per episode) instead of Gradio.

---

### Kill switches (decision table)

| By end of | Check | If not met |
|---|---|---|
| Day 2 | A frame is captured from the sim | Start Plan B skeleton full-time; keep trying fix in the background |
| Day 3 | Wires visible + collidable, or alternate hard cases chosen | Use B4-C/D |
| Day 5 | "Never ask" closed loop runs; sec/episode measured; budget written | Switch to Plan B (§9) and kinematic stepping (B3-B) |
| Day 8 | Gate beats trivial baseline | Apply B9 chain; if still no, carry on and report honestly |
| Day 9 | VLM recall acceptable | B6 chain; surrogate judge by Day 10 at the latest |
| Day 11 | All policies run end-to-end | Cut the grid (B2-D); freeze scope |
| Day 12 | Test-set run launched | No new features; only analysis and writing |

### Backups to create now (insurance for the work itself)

1. **VM snapshot/image** right after the simulator works (and again after models are installed).
2. **Git commit + push daily** to a remote repo; tag Day 5 / 8 / 11 milestones.
3. **Two copies of all episode data and logs:** VM disk plus HPC scratch (and a third to cloud storage/Drive for the final test set). `rsync` script in `hpc/`.
4. **Environment lockfiles:** `conda env export` / `pip freeze` for VM and HPC; record driver, CUDA, UE, AirSim versions in README.
5. **Cached model weights** on HPC scratch and VM disk (Depth Anything V2, SAM 2, Qwen3-VL 4B/8B), so no re-download during a crunch.
6. **Frozen seed lists and configs** (`configs/seeds.yaml`, policies, thresholds) committed before experiments.
7. **Checkpointed experiments:** every sweep appends per-episode results to a CSV/JSONL file and skips finished (policy, seed) pairs on restart.
8. **A tiny "golden" set of ~20 episodes** saved early so perception, gate, VLM, and demo can be developed without the simulator running.
9. **Plan B simulator skeleton** (half a day, Day 3–4) kept alive in the repo as the fallback.
10. **Surrogate VLM judge** implemented by Day 10 regardless, as a fast path for sweeps and a safety net.

---

## References (via Paper Lantern)

- Yang et al. 2026, *Triggering Generalist Reasoning via Predictive Uncertainty for Dual-System VLA* — arxiv:2610.05025
- Yuan et al. 2026, *When to Act, Ask, or Learn: Uncertainty-Aware Policy Steering* — arxiv:2602.22474
- Agia et al. 2024, *Unpacking Failure Modes of Generative Policies: Runtime Monitoring of Consistency and Progress* — arxiv:2410.04640
- Mei et al. 2024, *Perceive With Confidence* — arxiv:2403.08185
- Kumar et al. 2025, *Learnable Conformal Prediction with Context-Aware Nonconformity Functions* — arxiv:2509.21955
- Ghosh et al. 2022, *Dynamic Selection of Perception Models for Robotic Control* — arxiv:2207.06390
- Pei et al. 2026, *FARM* — arxiv:2609.11445
