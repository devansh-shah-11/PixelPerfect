# AI behaviour log (append-only, newest at bottom)

## 2026-10-20 23:10  |  git  |  Initialize repo and push to PixelPerfect
- Request: "initialize it with local project" (repo URL given).
- Action: laptop. `git init -b main`, added origin, fetched, `git reset origin/main` (mixed), restored/expanded README.md, added project ignores, committed, pushed (fast-forward, no force). Inspected remote first via clone in scratchpad: it held only a stub README and the standard Python .gitignore.
- Assumptions: "initialize with local project" meant commit and push the local files to the existing remote main [UNVERIFIED but user-supplied URL and wording make it likely].
- Decisions & alternatives: kept remote history (rejected: force-push over it). Expanded the 14-byte README with a short project summary.
- Evidence: push output `bce9faf..c2df14b main -> main`.
- Outcome: success, after correcting one mistake (below).
- Mistake: first commit replaced the remote's 220-line .gitignore with only my 16 lines (the local folder had no .gitignore, so `cat >>` created a new file). Caught from the commit's diffstat before pushing; rebuilt .gitignore from `origin/main` plus the additions and amended the unpushed commit. Final diff for .gitignore: +16 lines.
- Follow-ups: this log and PROGRESS.md are uncommitted; commit when the user asks.

## 2026-10-31 02:30  |  sim  |  Project AirSim headless smoke test on HPC (A100)
- Request: set up and test AirSim on HPC because no GCP VM is available.
- Action: HPC (NYU cloud bursting, user dns5508). Wrote `hpc/instructions.md`, `hpc/smoke_test.sbatch`, `hpc/smoke_client.py` (laptop). Ran jobs 15418 (g2-standard-12, L4), 15490 (A100, nvidia-smi only), 15498 (A100, sim + hello_drone.py), and a later A100 job running `smoke_client.py`.
- Assumptions: [UNVERIFIED] how A100 and boot time are billed against the 300 GPU-hours; [UNVERIFIED] whether the sim can output segmentation (not in the sensor list).
- Decisions & alternatives: used Project AirSim v1.0.1 (PLAN.md lists it as Fallback, Q4, licence shown by pip as MIT) instead of Microsoft AirSim/Colosseum. DEVIATION from PLAN.md D3; needs team sign-off. Ran the binary natively (no container); container fallback not needed. Used the A100 partition because g2-standard-12 never booted.
- Evidence: job 15418 `CANCELLED+` after 00:10:41, node b-31-162 `drained#`, reason "15418 finishing" (user slurm); about 10 other g2-standard-12 nodes already `drained#`. Job 15498: sim up on RHEL 9.8, hello_drone aborted with exit 134 (`qt.qpa.xcb: could not connect to display`, OpenCV windows). Later job: `smoke_client.py` exit 0, RGB (225,400,3) mean 104.4 std 17.1, depth (225,400) finite 1.00, 3.50 to 7.61 m; rgb.png inspected by eye (non-black, shaded). Nvidia Vulkan ICD present: /usr/share/vulkan/icd.d/nvidia_icd.x86_64.json.
- Outcome: partial success. Sim renders and returns RGB + depth headless on A100. L4 partition unverified (boot failure). Segmentation, step time and pause/step not tested.
- Mistakes: (1) first told the user to `scancel` the stuck job and to run `srun` from the OOD terminal, which was already inside a CPU job; the real fix was `sbatch`. (2) first diagnosed the srun failure as a wrong GRES name; sinfo showed `gpu:l4:1`, so `gpu:1` was valid. (3) said wall-clock limit would not count during CF; scontrol's EndTime suggests it does [UNVERIFIED].
- Follow-ups: sign-off on Project AirSim vs PLAN.md default; test segmentation availability and pause/step API (D10); measure sec per step; contact NYU HPC about g2-standard-12 boots if L4 is wanted; check GPU-hour usage.
