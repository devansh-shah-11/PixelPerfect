# Meeting with Prof. Saining Xie

**Date:** late October 2026 (exact date to fill in; PLAN.md Q9)
**Attendees:** Pixel Perfect team, Prof. Xie
**Source:** the team's meeting transcript summary, copied below unchanged. Terms such as "Post-Astra" and "system zero" come from that summary; see "Our reading" for what we think they refer to.

---

## Transcript summary (as recorded)

### Project Idea: Dual-Brain Autonomous Drone Pilot

- Core concept: two-system architecture for autonomous drone control
  - Fast perception system handles all real-time decisions at flight speed
  - Slow but smart VLM (e.g., Qwen-3 or similar scale) handles high-level reasoning
- Motivation for the split: VLMs are too slow to react at drone speeds
  - Running VLM continuously leads to latency that causes crashes
  - Current approaches that use VLM as a copilot (always-on monitoring) still too slow to be safe
- Learned gating mechanism: small model predicts when the fast stack is likely wrong
  - Only then does it escalate to the VLM
  - Edge cases targeted: thin wires, glass surfaces, fog, low-visibility conditions
- Both models intentionally kept small for real-world deployability
  - VLM target: around 2B parameters or similar lightweight variant
- Evaluation criteria: crash rate vs. VLM query rate
  - Tradeoff: fewer VLM queries is better for speed, but must not increase crashes
- Framing: adaptive gating vs. existing hard-coded confidence threshold approaches
  - Prior papers use fixed thresholds; this project aims for a learned, adaptive version
  - Not a completely novel idea, but a meaningful reframing as a research proposition

### Feedback and Suggestions from Saining

- Idea well-received overall; encouraged to pursue it with ambition
- Strongly recommended broadening literature review beyond drone/aerial simulation
  - Recent robotics and self-driving papers cover very similar fast/slow planner hierarchies
  - Post-Astra work in particular explores exactly this kind of dual-system setup
  - Papers from the last few months are most relevant; older papers less so
- Hierarchical planning framing to be aware of:
  - High-level slow planner using large vision-language models
  - Fast reactive policy for real-time control
  - Some work also proposes a "system zero" layer for ultra-fast locomotion at ~200 Hz
- Alternative approach suggested: VLA or diffusion-based methods for high-level command generation
  - Action space for flying is low-dimensional, so natural language output may not be necessary
  - Worth exploring as an alternative to pure VLM-based high-level reasoning
- Simulator-first approach is fine, but hardware demo flagged as a compelling option if results are strong
  - Saining recalled a live drone control demo from two years ago as a memorable example

### Expectations and Scope

- Explicit guidance: no pressure to beat state-of-the-art systems
  - Goal is learning, exploration, and surfacing new insights from the investigation
  - Negative results are explicitly valued and considered useful contributions
- Encouraged to be bold and creative rather than conservative
  - Showing innovation in the approach matters more than matching existing benchmarks
  - Even if the system underperforms, the exploration itself is the deliverable
- Project ambition check: team confirmed they have a clear idea and the approach is feasible
  - Saining confirmed: "It's definitely possible"
- Presentation framing: show the innovation, surface interesting findings, don't anchor on baseline comparisons

### Next Steps

- **Broaden literature review to recent robotics and self-driving papers**
- **Explore VLA and diffusion-based approaches as alternatives to VLM high-level planning**
- **Finalize the fast perception model choice**
- **Evaluate hardware demo feasibility if simulator results are promising**

---

## Our reading (to confirm)

- **"Astra"**: most likely ByteDance Seed's *Astra: Toward General-Purpose Mobile Robots via Hierarchical Multimodal Learning* (arXiv 2506.06205), a System 1/System 2 design for mobile robots. Confirm with whoever took notes (PLAN.md Q9).
- **"System zero"**: Figure's Helix 02 (Jan 2026). Figure says System 1 runs at 200 Hz and System 0 at 1 kHz, so the "~200 Hz" in the summary is probably System 1.
- **"Prior papers use fixed thresholds"**: only partly true. Learned triggers already exist in driving, manipulation, and one drone system (CoDrone). See `report/related_work.md`; this changes our claim.

## What we changed because of this meeting

| Feedback | Change | Where |
|---|---|---|
| VLM around 2B | Qwen3-VL-2B by default, 8B as size reference | PLAN.md D8 |
| VLA or diffusion as alternative slow system | Slow system swappable: VLM judge, VLM choosing actions, stretch flow-matching/diffusion policy; new experiment E7 | PLAN.md D18, Section 10 |
| Broaden literature review | arXiv and web searches across drones, driving, manipulation; claim reframed | PLAN.md Section 9, `report/related_work.md` |
| Finalize fast perception model | Short comparison on DDOS (thin-wire recall, FPS on L4), decided Mon Nov 2 | PLAN.md Section 13 |
| Hardware demo if results are strong | Stretch, decided at the MVP review on Wed Nov 11 | PLAN.md D20 |
| Negative results valued; don't anchor on baselines | Presentation leads with findings; baselines stay in the report | PLAN.md Section 15 |
