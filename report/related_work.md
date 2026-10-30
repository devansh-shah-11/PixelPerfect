# Related work (task S-03)

**Status:** started Oct 29, 2026. **Everything below comes from abstracts, arXiv listings, or search summaries. No paper has been read in full yet.** Before citing anything in the report, read it and change its status to `read (initials)`.

Short version for PLAN.md: Section 9.

---

## 1. Our claim, given what exists

**Not new on its own:** a learned gate instead of a fixed threshold. Learned triggers for a slow model exist in driving (AdaDrive, Cognitive Dual-Process Planning), manipulation (EMS), and one drone system (CoDrone).

**What we claim instead (verify after reading CoDrone and AESOP in full):**

> Drone systems call slow models in parallel, at fixed rates, on uncertainty thresholds (UDAV), or through a learned offloading scheduler aimed at network delay (CoDrone). We learn when a drone's fast perception is wrong badly enough that a VLM's delay is worth paying. The gate is trained on whether asking would have changed the outcome, and delay is a physical cost that depends on speed.

Our distinguishing points:
1. **Delay has a physical cost in flight.** Delay and speed are inputs to the asking decision, and we sweep them.
2. **Replay labels.** The gate learns "would asking have changed the outcome?" from simulator forks: a supervised, outcome-based alternative to RL switching (EMS, CoDrone), scene-complexity proxies (AdaDrive, 2607.19194), and uncertainty thresholds (FASIONAD, UDAV).
3. **Small models for deployment**, and where they fail (thin wires, glass, fog).
4. **Swappable slow system** (PLAN.md D18): does the best asking policy depend on what the slow system is?

---

## 2. Read first

| Priority | Paper | Why | Owner | Status |
|---|---|---|---|---|
| 1 | **CoDrone** ([arXiv 2512.19083](https://arxiv.org/abs/2512.19083)) | Closest drone work: learned scheduler decides when to call a slower model. Need its scheduler inputs and reward (PLAN.md Q10) | | abstract only |
| 2 | **AESOP** ([arXiv 2407.08735](https://arxiv.org/abs/2407.08735)) | Drones; fast learned detector triggers slow LLM reasoning; accounts for delay | | abstract only |
| 3 | **Cognitive Dual-Process Planning** ([arXiv 2607.19194](https://arxiv.org/abs/2607.19194)) | Closest mechanism to our gate-L1; candidate baseline | | abstract only |
| 4 | **AdaDrive** ([arXiv 2511.06253](https://arxiv.org/abs/2511.06253)) | Best-known learned trigger (ICCV 2025) | | abstract only |
| 5 | **EMS** ([arXiv 2608.06434](https://arxiv.org/abs/2608.06434)) | RL-learned switching; contrast with replay labels | | abstract only |

---

## 3. Drone and UAV work

### 3.1 Learned or adaptive triggers

| Paper | What it does | How the slow model is called | How we differ | Status |
|---|---|---|---|---|
| **CoDrone: Autonomous Drone Navigation Assisted by Edge and Cloud Foundation Models.** Chen, Ouyang, Luo, Hong, Chen. IEEE IoT Journal (accepted). [arXiv 2512.19083](https://arxiv.org/abs/2512.19083), Dec 2025 | Onboard navigation from grayscale images; edge Depth Anything V2 for better depth; cloud VLM for open-set reasoning. Reports ~40% longer average flight distance | **Learned:** a deep-RL scheduler decides when to integrate edge depth estimation | Slow model is a depth model, not a VLM. Stated motivation is network delay. Abstract doesn't give the reward or inputs, or say whether flights are simulated or real | abstract only |
| **AESOP: Real-Time Anomaly Detection and Reactive Planning with Large Language Models.** Sinha, Elhafsi, Agia, Foutter, Schmerling, Pavone. RSS 2024. [arXiv 2407.08735](https://arxiv.org/abs/2407.08735) | Fast embedding-based anomaly detector, then slow LLM picks a fallback; MPC keeps fallbacks feasible during the LLM's delay; simulated drones and hardware | **Learned detector** flags unusual scenes | Triggers on unusual scenes, not fast-perception mistakes; not trained on outcomes | abstract only |
| **UDAV: Uncertainty-Driven Adaptive VLM Waypoint Planner.** [arXiv 2609.16368](https://arxiv.org/abs/2609.16368), Sep 2026 | VLM waypoint planning | **Threshold:** re-asks when the spread of sampled VLM routes exceeds a threshold | Fixed threshold; open loop. (PLAN.md lists it as ground-vehicle routes, but our arXiv "UAV" search returned it; check) | abstract only |
| **DisasterRoute: Evidence-Guided Dual-System Routing for Visual Disaster Question Answering.** Zhang, Deng, Zhang, Zhu, Hu. [arXiv 2610.29231](https://arxiv.org/abs/2610.29231), Oct 2026 | Question answering on low-altitude UAV images on edge hardware | Confidence plus disaster evidence decides whether to send a question to a slower reasoning model (abstract doesn't say whether learned) | Question answering, not flight control | abstract only |
| **AVERY: Intent-Driven Adaptive VLM Split Computing.** [arXiv 2511.18151](https://arxiv.org/abs/2511.18151), Nov 2025 | Disaster-response VLM split across edge and cloud | Onboard controller watches network conditions and operator intent | Not flight control; not error prediction | abstract only |
| **Introspective Perception: Learning to Predict Failures in Vision Systems.** Daftry, Zeng, Bagnell, Hebert. IROS 2016 | Predicts perception failures for drones flying through forests | Learned failure predictor | No slow model to ask | not checked online |

### 3.2 Slow model in parallel, at a fixed rate, or always on

| Paper | How the slow model is called | Status |
|---|---|---|
| **Stress-Query and Temporal Reliability of VLMs as copilots in Autonomous UAV Navigation.** Sodre et al. LARS 2026. [arXiv 2610.23877](https://arxiv.org/abs/2610.23877) | Always-on copilot (1.8B to 11B) in Gazebo; heavy models overcorrect from delay, small ones hallucinate; defines a "Temporal Reliability Boundary". **Our motivation, not a scoop** | abstract only |
| **VLMs as copilots for Autonomous UAV Navigation: Latency and Reliability in Degraded Environments.** [arXiv 2609.26084](https://arxiv.org/abs/2609.26084), Sep 2026 | Asynchronous VLM copilot next to a deterministic state machine | abstract only |
| **FSD-VLN: Fast-Slow Dual-System Modeling for Aerial Long-Horizon VLN.** [arXiv 2607.08359](https://arxiv.org/abs/2607.08359), Jul 2026 | Two asynchronous branches: slow VLM semantics, fast diffusion (DiT) flight commands | abstract only |
| **AeroDASH: Asynchronous VLA Control for Continuous UAV Flight.** [arXiv 2610.25407](https://arxiv.org/abs/2610.25407), Oct 2026 | Asynchronous; stale prediction prefixes dropped | abstract only |
| **VLM-MPPI: Grounding Natural Language in Behaviorally Diverse Trajectories for Aerial Navigation.** [arXiv 2609.18451](https://arxiv.org/abs/2609.18451), Sep 2026 | VLM picks among MPPI candidates asynchronously; MPPI replans at 20 Hz | abstract only |
| **APEX: Decoupled Memory-based Explorer for Asynchronous Aerial Object Goal Navigation.** [arXiv 2602.00551](https://arxiv.org/abs/2602.00551), Feb 2026 | Hierarchical asynchronous design hides VLM delay | abstract only |
| **AutoFly: VLA Model for UAV Autonomous Navigation in the Wild.** [arXiv 2602.09657](https://arxiv.org/abs/2602.09657), Feb 2026 | Vision and language pipelines run in parallel | abstract only |
| **AerialVLA: VLA for UAV Navigation via Minimalist End-to-End Control.** [arXiv 2603.14363](https://arxiv.org/abs/2603.14363), Mar 2026 | One model controls directly; lighter model to cut delay | abstract only |
| **AeroDPO** ([arXiv 2608.07557](https://arxiv.org/abs/2608.07557)); **Guided Diffusion with Distilled Vision-Language Reliability for Aerial Navigation** ([arXiv 2606.13883](https://arxiv.org/abs/2606.13883)); **UAV-Track VLA** ([arXiv 2604.02241](https://arxiv.org/abs/2604.02241)) | Smaller or distilled model instead of fewer calls | abstract only |

---

## 4. Learned triggers outside drones

| Paper | Domain | How the trigger is learned | Status |
|---|---|---|---|
| **AdaDrive: Self-Adaptive Slow-Fast System for Language-Grounded Autonomous Driving.** Zhang et al. ICCV 2025. [arXiv 2511.06253](https://arxiv.org/abs/2511.06253); [code](https://github.com/ReaFly/AdaDrive) | Driving (LangAuto) | "Adaptive activation loss" learns when to call the LLM; LLM influence scaled by scene complexity and confidence | abstract only |
| **EMS: Fast and Accurate: An Adaptive VLA Inference Framework through Environment-aware Model Selection.** Sun et al. [arXiv 2608.06434](https://arxiv.org/abs/2608.06434), Aug 2026 | VLA manipulation | Small switching policy trained with RL picks the small or large model | abstract only |
| **Cognitive Dual-Process Planning for Autonomous Driving with Structured Scene Knowledge and Verifiable Reasoning-Action Consistency.** Yang et al. [arXiv 2607.19194](https://arxiv.org/abs/2607.19194), Jul 2026 | Driving (NAVSIM) | Classifier on vision-encoder features, with an extra penalty for sending hard scenes to the fast path | abstract only |

## 5. Uncertainty or rule triggers outside drones

| Paper | Trigger | Status |
|---|---|---|
| **FASIONAD** ([arXiv 2411.18013](https://arxiv.org/abs/2411.18013)) | Fast planner's uncertainty | abstract only |
| **Triggering Generalist Reasoning via Predictive Uncertainty for Dual-System VLA** ([arXiv 2610.05025](https://arxiv.org/abs/2610.05025)) | Predicted uncertainty (check whether any part is learned) | abstract only |
| **Critic in the Loop: A Tri-System VLA Framework** ([arXiv 2603.05185](https://arxiv.org/abs/2603.05185)) | Calls the planner on subtask completion, anomalies, or stalled progress | abstract only |
| **ASSCG: Just-Right Gating over Chattering for Fast-Slow LLM Planning in Autonomous Driving** ([arXiv 2606.25509](https://arxiv.org/abs/2606.25509)) | Says most fast-slow driving planners use hand-designed triggers; uncertainty triggers are brittle; studies gates that flip on and off (related to our cooldown) | abstract only |
| **DualDriveVLA** ([arXiv 2602.10719](https://arxiv.org/abs/2602.10719)) | Calls the VLM in ~15% of scenarios (mechanism not checked) | abstract only |
| **Slow Brain, Fast Planner** ([arXiv 2606.20458](https://arxiv.org/abs/2606.20458)) | Fixed-rate VLM; lists uncertainty-gated asking as future work | abstract only |
| **Dual-process indoor navigation** ([arXiv 2601.21506](https://arxiv.org/abs/2601.21506)) | Calls the VLM on scene change (our scene-change baseline) | abstract only |
| **Think Only When Needed** ([arXiv 2608.23224](https://arxiv.org/abs/2608.23224)) | Selective slow path for manipulation (method not checked) | title only |

## 6. Hierarchical fast/slow architectures (framing)

| Work | Notes | Status |
|---|---|---|
| **Astra** (ByteDance Seed, [arXiv 2506.06205](https://arxiv.org/abs/2506.06205), Jun 2025) | Astra-Global: multimodal LLM for low-rate localization. Astra-Local: fast network, flow-matching planner with a masked ESDF loss. Probably the "Astra" from the meeting | abstract only |
| **Helix 02** ([Figure, Jan 2026](https://www.figure.ai/news/helix-02)) | System 2 slow VLM, System 1 at 200 Hz, System 0 at 1 kHz (trained in simulation) | company announcement |
| **Fast-in-Slow (FiS-VLA)** ([arXiv 2506.01953](https://arxiv.org/abs/2506.01953)) | Fixed dual-system VLA | search summary only |

Framing for the paper: our System 2 = VLM on demand; System 1 = fast stack; System 0 = the flight controller we abstract with kinematic stepping.

## 7. Background (already in PLAN.md Section 9)

- KnowNo (Ren et al., CoRL 2023, [arXiv 2307.01928](https://arxiv.org/abs/2307.01928)); Perceive With Confidence ([arXiv 2403.08185](https://arxiv.org/abs/2403.08185)): conformal guarantees for asking.
- Learning to defer: Madras et al. 2018; Mozannar & Sontag 2020.
- Thin obstacles: DDOS ([arXiv 2312.12494](https://arxiv.org/abs/2312.12494)); UCorr ([arXiv 2509.14989](https://arxiv.org/abs/2509.14989)).
- VLM blind spots (instructor's group): Eyes Wide Shut / MMVP, Cambrian-1, Thinking in Space.

---

## 8. Search log

| Date | Where | Query | Results | Notes |
|---|---|---|---|---|
| Oct 29 | arXiv, all fields, newest first | `UAV dual-system` | 124 (first 50 read) | Most are wireless-communications papers |
| Oct 29 | arXiv | `UAV fast slow` | 25 | |
| Oct 29 | arXiv | `drone vision-language model obstacle avoidance` | 5 | None selective |
| Oct 29 | arXiv | `UAV vision-language latency` | 18 | Found CoDrone |
| Oct 29 | arXiv | `UAV adaptive reasoning vision-language` | 17 | |
| Oct 29 | arXiv | `quadrotor vision-language model` | 9 | |
| Oct 29 | arXiv | `UAV uncertainty vision-language model` | 5 | Found UDAV |
| Oct 29 | arXiv | `drone event-triggered large model` | 0 | |
| Oct 29 | Web search | fast/slow robotics, driving, System 0, Astra, VLA for UAVs | n/a | Found AdaDrive, EMS, 2607.19194, Astra, Helix 02 |

Search results were summarized by an AI tool from abstracts; treat every entry as unverified until read.

**Still to search:** arXiv `UAV dual-system` results 51 to 124 (older); `drone System 2`; `aerial VLA diffusion policy`; venue proceedings (CoRL 2026, IROS 2026, ICRA 2026) for "fast-slow" and "when to ask".
