# PixelPerfect

Gated fast/slow perception for a simulated drone: fast perception (Depth Anything V2 + SAM 2 + rule controller) flies every frame, and a small learned gate asks a local VLM (Qwen3-VL) a yes/no question only when it predicts the fast perception is wrong.

Headline result: crash rate vs. VLM calls per episode for never / always / random / hand-tuned threshold / learned gate / oracle. Simulation only.

- Plan, schedule, metrics, blockers and fallbacks: [`PLAN.md`](PLAN.md)
- Working rules for the AI assistant: [`CLAUDE.md`](CLAUDE.md)
- Progress and AI logs: `logs/`

Status: planning complete, implementation starting (Day 1 of 14).
