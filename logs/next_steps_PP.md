# Next steps for PP (from Fri Oct 30)

Goal for this weekend: **one simulator frame saved on a Cloud Bursting L4 by Sat Oct 31** (`PLAN.md` §11 kill switch, tasks A-01 to A-03). Everything below leads there.

How the pieces connect: the Mac and HPC never talk directly. Code goes Mac → GitHub → HPC (`git pull`); results come back through the Open OnDemand file browser or small CSVs in git. The simulator, the Python loop, the fast stack and the VLM all run inside one HPC job (`PLAN.md` D17).

## 1. Team call: owners and PR #1 (today, 15 min)
- Assign owners for workstreams A, B and C (`PLAN.md` Q1), and fill in the names in `TASKS.md`.
- Ask one teammate to review and merge [PR #1](https://github.com/devansh-shah-11/PixelPerfect/pull/1) (`PLAN.md` §0 requires it). Use a merge commit, not squash.
- Ask whoever owns C to email the TA about simulation-only evaluation and the presentation date (Q3, Q6).

## 2. Set up the repo on the Mac and read the code (today, 30 min)
Anyone can be asked about any part of the code at the final presentation, so start now while it's small.
```bash
cd PixelPerfect
git pull
uv venv .venv --python 3.10 && uv pip install -e ".[dev]"   # skip if .venv already exists
.venv/bin/python -m pytest                                 # expect: 7 passed
```
Then read `interfaces.py` (about 170 lines). It defines what every workstream passes to the others (task S-04). Note anything you'd change before all three of you agree on it.

## 3. First Cloud Bursting session and smoke test (today, 30 min, task S-06 / A-01)
1. Turn on the NYU VPN if you're off campus, then open https://ood.burst.hpc.nyu.edu/.
2. Start an interactive session on an **L4** (account `csci_ga_2271-2026fa`). **Never the Torch cluster.**
3. In its terminal:
   ```bash
   cd $SCRATCH        # work in scratch, not home
   git clone https://github.com/devansh-shah-11/PixelPerfect.git && cd PixelPerfect
   git checkout S/plan-v1.1-skeleton   # until PR #1 is merged
   bash hpc/smoke_test.sh 2>&1 | tee smoke_$(hostname).txt
   sinfo -s           # partition names for hpc/job_template.sbatch
   ```
4. Paste both outputs to Claude. The important lines are **Vulkan/EGL** (can Unreal render without a screen?), **Singularity**, and **internet access** (can the node download the simulator and the model weights?).
5. End the session when you're done: idle GPU sessions are shut down after 20 minutes and still use up hours.

## 4. Simulator binary downloaded and launched headless (Sat Oct 31, tasks A-01 to A-03)
Based on the smoke-test output, Claude writes `sim/airsim_backend.py`, the AirSim `settings.json` (ComputerVision mode, for kinematic stepping), a launch script, and a frame-capture script. Then, in one L4 session:
- Download a prebuilt **Linux** AirSim environment (for example Blocks from AirSim's GitHub releases; check the release and its license first) to `$SCRATCH`.
- Launch it inside Singularity with `--nv` and `-RenderOffScreen`, then run the capture script. Success means 10 frames of RGB, ground-truth depth, segmentation and pose saved under `data/` (A-02).
- **If there's no frame by the end of Saturday**, switch to the Plan B simulator full time (A-07), per `PLAN.md` §11. Don't keep debugging rendering past Saturday.

Meanwhile, Claude builds the toy simulator backend for the Mac behind the same interface. The rest of the loop (controller, policies, stand-in VLM, metrics) can then be developed and tested on the laptop whatever happens with AirSim.

## 5. Read CoDrone and AESOP in full (by Mon Nov 2, about 1 hour each)
They're the two papers reviewers will compare us to. Our notes on them come from abstracts or an AI summary, which isn't enough to cite them in a paper.
- **CoDrone**, [arXiv 2512.19083](https://arxiv.org/abs/2512.19083): confirm that its learned scheduler only decides offloading and that its VLM is called by fixed rules (Q10).
- **AESOP**, [arXiv 2407.08735](https://arxiv.org/abs/2407.08735): note how its anomaly trigger works and how it accounts for the LLM's delay.

Record what you find in `report/related_work.md`. Change each entry's status from "abstract only" to "read in full". Any point where the AI summary was wrong goes in `AGENT_LOG.md`, since the Agent Log needs those examples.
