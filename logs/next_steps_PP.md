# Next steps for PP (from Fri Oct 30)

**Status, Fri Nov 13:** steps 3 and 4 are done a different way. L4 nodes never booted, and the team switched to Project AirSim v1.0.1 on the A100 partition. PP followed `hpc/instructions.md` and the simulator smoke test passed (job 31620: exit code 0, RGB 225x400, depth 3.47 to 7.56 m). Use `hpc/instructions.md` for HPC setup; the commands in steps 3 and 4 below are kept as a record only. Steps 1, 2 and 5 still apply.

Goal for this weekend: **one simulator frame saved on a Cloud Bursting L4 by Sat Oct 31** (`PLAN_DETAILED.md` §11 kill switch, tasks A-01 to A-03). Everything below leads there.

How the pieces connect: the Mac and HPC never talk directly. Code goes Mac → GitHub → HPC (`git pull`); results come back through the Open OnDemand file browser or small CSVs in git. The simulator, the Python loop, the fast stack and the VLM all run inside one HPC job (`PLAN_DETAILED.md` D17).

## 1. Team call: owners and PR #1 (today, 15 min)
- Assign owners for workstreams A, B and C (`PLAN_DETAILED.md` Q1), and fill in the names in `TASKS_DETAILED.md`.
- Ask one teammate to review and merge [PR #1](https://github.com/devansh-shah-11/PixelPerfect/pull/1) (`PLAN_DETAILED.md` §0 requires it). Use a merge commit, not squash.
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
Partitions and rules are in `hpc/instructions.md`. **Never the Torch cluster.**
1. Turn on the NYU VPN if you're off campus, then open https://ood.burst.hpc.nyu.edu/ (use a private window if login fails).
2. Start a Jupyter session and open a terminal inside it.
3. In that terminal, get the code onto scratch (until PR #1 is merged, use the PR branch):
   ```bash
   cd /scratch/$USER
   git clone https://github.com/devansh-shah-11/PixelPerfect.git
   cd PixelPerfect
   git checkout S/plan-v1.1-skeleton
   ```
4. Run the smoke test on an L4 as a batch job. The Jupyter terminal is itself a CPU job, so a plain `srun` from it fails (see `hpc/instructions.md`):
   ```bash
   sbatch hpc/job_template.sbatch
   squeue -u $USER
   ```
   The state goes `CF` (machine starting, a few minutes), then `R`, then the job disappears from the list. Then run `cat slurm-pp-<jobid>.out`.
5. Paste the output to Claude. The important lines are **nvidia-smi** (an L4 should be listed), **Vulkan/EGL** (can Unreal render without a screen?), **Singularity**, and **internet access** (can the node download the simulator and the model weights?).
6. Results so far (PP, Fri Oct 30, CPU node): Singularity 4.5.0 is installed; huggingface.co and github.com are reachable; scratch is at `/scratch/<NetID>`; the system Python is 3.9, so we'll use conda or a container for 3.10+. The NVIDIA EGL and Vulkan libraries are installed but still need confirming on an L4.

## 4. Simulator binary downloaded and launched headless (Sat Oct 31, tasks A-01 to A-03)
Based on the smoke-test output, Claude writes `sim/airsim_backend.py`, the AirSim `settings.json` (ComputerVision mode, for kinematic stepping), a launch script, and a frame-capture script. Then, in one L4 session:
- Download a prebuilt **Linux** AirSim environment (for example Blocks from AirSim's GitHub releases; check the release and its license first) to `$SCRATCH`.
- Launch it inside Singularity with `--nv` and `-RenderOffScreen`, then run the capture script. Success means 10 frames of RGB, ground-truth depth, segmentation and pose saved under `data/` (A-02).
- **If there's no frame by the end of Saturday**, switch to the Plan B simulator full time (A-07), per `PLAN_DETAILED.md` §11. Don't keep debugging rendering past Saturday.

Meanwhile, Claude builds the toy simulator backend for the Mac behind the same interface. The rest of the loop (controller, policies, stand-in VLM, metrics) can then be developed and tested on the laptop whatever happens with AirSim.

## 5. Read CoDrone and AESOP in full (by Mon Nov 2, about 1 hour each)
They're the two papers reviewers will compare us to. Our notes on them come from abstracts or an AI summary, which isn't enough to cite them in a paper.
- **CoDrone**, [arXiv 2512.19083](https://arxiv.org/abs/2512.19083): confirm that its learned scheduler only decides offloading and that its VLM is called by fixed rules (Q10).
- **AESOP**, [arXiv 2407.08735](https://arxiv.org/abs/2407.08735): note how its anomaly trigger works and how it accounts for the LLM's delay.

Record what you find in `report/related_work.md`. Change each entry's status from "abstract only" to "read in full". Any point where the AI summary was wrong goes in `AGENT_LOG.md`, since the Agent Log needs those examples.
