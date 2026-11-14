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

## 2026-10-29 19:26  |  docs  |  Align CLAUDE.md with PLAN.md; move AI log to AGENT_LOG.md
- Member: PP
- Request: "go ahead" with the proposal to make PLAN.md win over CLAUDE.md and update CLAUDE.md to match, as a local `docs:` commit (not pushed).
- Action: laptop. Rewrote CLAUDE.md: SAM 2 as ablation only (D7); simulator machine open (D17) instead of "GCP VM"; section references fixed (§7 layout, §11 kill switches/insurance, §16 open questions; "Cut first if late" never existed, so scope now points to Appendix A); branch naming from TASKS.md; tag `mvp-v1` (S-08) instead of day5/day8/day11; removed stale "not yet a git repo" line; disagreement rule changed from "stop and ask" to "PLAN.md wins, tell the user, log it". `git mv logs/AI_LOG.md AGENT_LOG.md` (PLAN.md §7 and TASKS.md expect AGENT_LOG.md at the root); added a `Member:` field to the entry template; README link updated. Set repo-local git identity (not global) to "Parva Patel" with the GitHub no-reply address, because none was configured and the repo is public. Commit 70f3252.
- Assumptions: author name "Parva Patel" inferred from GitHub login `ParvaPatel` (profile name is empty) [UNVERIFIED]. Initials "PP" for this member [UNVERIFIED].
- Decisions & alternatives: kept one log file (rejected: separate assistant log and team Agent Log, which would duplicate entries). Did not touch PLAN.md (rule 0). Did not rewrite this file's earlier entry (append-only).
- Evidence: script check that every `PLAN.md` §, D-id, Q-id, and TASKS id cited in CLAUDE.md exists: all OK. `grep AI_LOG` over *.md: no leftover references. `git show --stat`: 3 files, 21+/20-, rename kept history.
- Outcome: success.
- Follow-ups: push when the user asks. README status line ("Day 1 of 14") is stale; left alone.

## 2026-10-29 19:26  |  plan  |  Sprint 1 schedule has slipped about 8 days
- Member: PP
- Request: user confirmed there is no unpushed work and the team has not started implementation.
- Action: none (record only). Compared PLAN.md §11 and §13 dates against today.
- Assumptions: none.
- Decisions & alternatives: recorded here instead of editing PLAN.md (rule 0). Re-dating is a team decision through the PLAN.md §0 change process.
- Evidence: all TASKS.md rows TODO, owners unassigned, `main` is the only branch, no PRs, last push 2026-10-21. Passed with no work started: first sim frame (Thu Oct 22), go/no-go (Fri Oct 23), seeds frozen (Sat Oct 24), closed-loop and sec/flight (Mon Oct 26), golden set (Tue Oct 27), VLM wire recall (Wed Oct 28). S-03 (read Stress-Query, arXiv 2610.23877) and Q2, Q3, Q5, Q6 also still open.
- Outcome: deviation from PLAN.md §13.
- Follow-ups: team to re-date Sprint 1 (MVP Tue Nov 3 to about Wed Nov 11) and decide what Sprint 2 loses. Simulator (blocker rank 1) is the critical path.

## 2026-10-29 20:00  |  docs  |  PLAN.md v1.1 (meeting, HPC facts, related work, re-dated schedule)
- Member: PP
- Request: "Go ahead" with the proposed batch: update PLAN.md (explicitly approved), CLAUDE.md, save meeting notes, start report/related_work.md; one local commit, not pushed.
- Action: laptop. PLAN.md edited by a script doing 33 exact-text replacements (each asserted to match exactly once) plus a full rewrite of Section 13: D8, D17 changed; D18, D19, D20 added; Sections 5, 6, 8, 9, 10, 11, 15, 16, Appendix A5 and B updated. CLAUDE.md environments rewritten for Cloud Bursting. New: report/meetings/prof-xie-meeting.md (transcript summary copied unchanged + our reading + changes table), report/related_work.md. Commit f27ee87.
- Assumptions: "Astra" in the meeting notes = ByteDance's Astra (arXiv 2506.06205) [UNVERIFIED, Q9]. "System zero at ~200 Hz" in the notes conflicts with Figure's description (S1 200 Hz, S0 1 kHz); recorded as Figure states it. A100 may not render Unreal [UNVERIFIED]. Meeting date unknown (Q9). All paper descriptions are from abstracts or AI-summarized search results [UNVERIFIED]; related_work.md marks every entry "abstract only".
- Decisions & alternatives: kept the gate as the contribution and made the slow system swappable (D18) instead of replacing the VLM with a VLA (rejected: too much scope after the slip). Kept the Mon Nov 30 final-run date and shortened Sprint 2 instead of moving the report week. Left relative "Week N" labels in Sections 4.3 and 10; the Sprint 2 table carries dates.
- Mistake (corrected by the user's arXiv search): I earlier told the user no learned triggers were used on drones. The arXiv keyword searches the user suggested found CoDrone (arXiv 2512.19083), a drone system with an RL-learned scheduler for calling a slower model. Earlier I also listed FASIONAD and 2610.05025 as "learned"; they are uncertainty triggers. Both corrected in PLAN.md Section 9 and related_work.md.
- Evidence: script output "OK: 33 replacements + Section 13"; check that every D/Q/§ reference in CLAUDE.md, PLAN.md and both new files exists and every Markdown table has consistent columns: ALL OK. `grep 4B PLAN.md`: none left.
- Outcome: success.
- Follow-ups: one teammate reviews before push (PLAN.md §0). TASKS.md due dates still use the old schedule. Read CoDrone and AESOP in full (Q10). Fill in the meeting date.

## 2026-10-29 20:09  |  eval  |  Repo skeleton, shared interfaces, smoke test, TASKS.md re-dated
- Member: PP
- Request: "Let's take whatever step next is required and start our work!"
- Action: laptop. Wrote interfaces.py (PLAN.md §6 types and Protocols), tests/test_interfaces.py, pyproject.toml, package folders from PLAN.md §7, configs/README.md, hpc/job_template.sbatch, hpc/smoke_test.sh. Re-dated TASKS.md by script (68 dates, 21 text changes). Created a local .venv with uv (Python 3.10, gitignored) because the laptop only has Python 3.9 without numpy. Commit 078e7ac.
- Assumptions: Slurm `--gres=gpu:1` works on Cloud Bursting and partition names are unknown (left as FILL_ME_IN) [UNVERIFIED]. Whether compute nodes have internet access is unknown; the smoke test checks it [UNVERIFIED].
- Decisions & alternatives: policies receive a history of RuntimeStep (no ground truth), not FrameRecord, to keep CLAUDE.md rule 5 enforceable by type; rejected passing full log rows. Gate.score takes a features dict (PLAN.md §6) and the encoder embedding goes under features["embedding"], instead of a separate argument. Added split and gate_score to FrameRecord (not in PLAN.md §6). Did not implement any workstream task beyond S-04/S-06 stubs.
- Evidence: pytest 7 passed. bash -n on both HPC scripts: ok. Smoke test dry run on the laptop degrades gracefully (no GPU, no Singularity). TASKS.md: 0 old dates left; all tables have consistent columns.
- Mistake: the TASKS.md script first failed because "### Sprint 1 (to Tue Nov 3)" appears three times; it stopped before writing (exactly-once check), then was fixed to replace all three.
- Outcome: success.
- Follow-ups: all three members must agree on interfaces.py (S-04). Each member runs the smoke test (S-06) and fills in the partition name.

## 2026-10-29 20:09  |  docs  |  CoDrone full text: VLM trigger is rule-based (Q10)
- Member: PP
- Request: same session; reading CoDrone was the agreed next step.
- Action: fetched arxiv.org/html/2512.19083; an AI tool summarized the first ~100k of ~120k characters. Updated report/related_work.md. Commit f0ebbeb.
- Assumptions: findings come from an AI summary of the HTML, not a human read; the last ~17% of the page was not read [UNVERIFIED].
- Decisions & alternatives: claim kept; CoDrone's rule trigger to be used as a reference baseline. PLAN.md Q10 not edited (rule 0: no explicit request); proposed to the user instead.
- Evidence: summary cites §IV-B3 (A2C scheduler: state = steering, collision rate, 5-step average, bandwidth; action = onboard/edge, depth on/off, compression), §IV-C1 (rule-based VLM invocation; drone suspended during analysis), §V-VI (AirSim Coastline; QoN, flight distance, latency; no collision metric).
- Outcome: success. This narrows the risk flagged in the previous entry: no drone paper found so far learns when to ask a VLM.
- Follow-ups: a human reads CoDrone and AESOP fully before the paper cites them.

## 2026-10-30 12:50  |  infra  |  Recovered lost session; pushed to a branch and opened a PR
- Member: PP
- Request: "go ahead create a PR" (after the project folder was renamed, which hid the previous Claude Code session).
- Action: laptop. Found the old session transcript under the old folder path and restored its project memory. Created branch S/plan-v1.1-skeleton from local main (4 commits ahead of origin/main), committed the pending AGENT_LOG.md and logs/PROGRESS.md entries, pushed the branch, opened a PR into main.
- Assumptions: none new.
- Decisions & alternatives: branch + PR instead of pushing to main, so a teammate reviews the PLAN.md changes as PLAN.md §0 requires. One combined branch rather than one per task, since the 4 commits were already stacked.
- Mistake: the previous session ended ("Not logged in") before acting on the user's "go ahead" to push; the PR was not opened until this session.
- Evidence: `git fetch` showed origin/main unchanged at 8b56bec; GitHub API reports push permission; pytest 7 passed after the folder rename.
- Outcome: success.
- Follow-ups: a teammate reviews and merges the PR. Use a merge commit (not squash) so local main fast-forwards.

## 2026-10-30 13:30  |  docs  |  Five starting action items for PP
- Member: PP
- Request: "give me 5 action items that i should start with - add that to some markdown file and push it".
- Action: laptop. Wrote logs/next_steps_PP.md: (1) team call for owners and PR #1 review, (2) Mac setup and read interfaces.py, (3) first Cloud Bursting L4 session and smoke test, (4) AirSim binary headless by Sat Oct 31 with Plan B fallback, (5) human read of CoDrone and AESOP. Placed in logs/ because PLAN.md §7 has no docs/ folder.
- Assumptions: Open OnDemand offers an interactive L4 session with a terminal [UNVERIFIED]. `$SCRATCH` is set on Cloud Bursting nodes [UNVERIFIED; the smoke test prints it]. A prebuilt Linux AirSim environment such as Blocks is on AirSim's GitHub releases [UNVERIFIED].
- Decisions & alternatives: the Mac never talks to HPC at runtime; code goes through GitHub and results come back via the OnDemand file browser (user pointed out the Mac can't reach HPC directly). Rejected: Python client on the Mac driving a simulator on HPC.
- Evidence: none needed (docs only).
- Mistake: the assistant's shell hung on every command (even `echo`), so the assistant could not commit or push; the user ran the git commands.
- Outcome: partial (file written; commit and push left to the user).
- Follow-ups: PP pastes the smoke-test and `sinfo` output; assistant then writes the AirSim backend, settings.json, launch and capture scripts, plus the toy backend.

## 2026-10-31 02:30  |  sim  |  Project AirSim headless smoke test on HPC (A100)
- Member: DS
- Request: set up and test AirSim on HPC because no GCP VM is available.
- Action: HPC (NYU cloud bursting, user dns5508). Wrote `hpc/instructions.md`, `hpc/smoke_test.sbatch`, `hpc/smoke_client.py` (laptop). Ran jobs 15418 (g2-standard-12, L4), 15490 (A100, nvidia-smi only), 15498 (A100, sim + hello_drone.py), and a later A100 job running `smoke_client.py`.
- Assumptions: [UNVERIFIED] how A100 and boot time are billed against the 300 GPU-hours; [UNVERIFIED] whether the sim can output segmentation (not in the sensor list).
- Decisions & alternatives: used Project AirSim v1.0.1 (PLAN.md lists it as Fallback, Q4, licence shown by pip as MIT) instead of Microsoft AirSim/Colosseum. DEVIATION from PLAN.md D3; needs team sign-off. Ran the binary natively (no container); container fallback not needed. Used the A100 partition because g2-standard-12 never booted.
- Evidence: job 15418 `CANCELLED+` after 00:10:41, node b-31-162 `drained#`, reason "15418 finishing" (user slurm); about 10 other g2-standard-12 nodes already `drained#`. Job 15498: sim up on RHEL 9.8, hello_drone aborted with exit 134 (`qt.qpa.xcb: could not connect to display`, OpenCV windows). Later job: `smoke_client.py` exit 0, RGB (225,400,3) mean 104.4 std 17.1, depth (225,400) finite 1.00, 3.50 to 7.61 m; rgb.png inspected by eye (non-black, shaded). Nvidia Vulkan ICD present: /usr/share/vulkan/icd.d/nvidia_icd.x86_64.json.
- Outcome: partial success. Sim renders and returns RGB + depth headless on A100. L4 partition unverified (boot failure). Segmentation, step time and pause/step not tested.
- Mistakes: (1) first told the user to `scancel` the stuck job and to run `srun` from the OOD terminal, which was already inside a CPU job; the real fix was `sbatch`. (2) first diagnosed the srun failure as a wrong GRES name; sinfo showed `gpu:l4:1`, so `gpu:1` was valid. (3) said wall-clock limit would not count during CF; scontrol's EndTime suggests it does [UNVERIFIED].
- Follow-ups: sign-off on Project AirSim vs PLAN.md default; test segmentation availability and pause/step API (D10); measure sec per step; contact NYU HPC about g2-standard-12 boots if L4 is wanted; check GPU-hour usage.

## 2026-11-13 21:40  |  sim  |  Project AirSim smoke test reproduced on a second account
- Member: PP
- Request: Devansh asked teammates to set up Project AirSim on their own HPC accounts from `hpc/instructions.md`.
- Action: HPC (Cloud Bursting, user psp8502). Followed `hpc/instructions.md` §1 and §2, with one change: the scripts came from `git checkout sim/capability-test` in the existing clone, not `scp`. Conda env `/scratch/psp8502/envs/pa` (Python 3.11.18), `projectairsim==1.0.1`, `Blocks-Linux-1.0.1.zip`. Job 31620 on `c12m85-a100-1`.
- Assumptions: none new.
- Decisions & alternatives: A100, because L4 nodes never booted for either member (PP job 15186 on Oct 30 sat in `CF`; DS job 15418).
- Evidence: job 31620: `smoke_client exit code: 0`; RGB (225, 400, 3) mean 104.3 std 17.2; depth (225, 400) finite 1.00, 3.47 to 7.56 m; `rgb.png` and `depth.npy` saved. DS's run gave mean 104.4, std 17.1, depth 3.50 to 7.61 m, so the setup reproduces. About 2.5 min in `CF`, about 1.5 min running. `ls /scratch/dns5508/frame_bank/CityEnviron` from psp8502: Permission denied.
- Mistakes (assistant, Oct 30): told PP to run `srun --gres=gpu:1` from the OOD Jupyter terminal, which is already inside a CPU job, so it failed with "Invalid generic resource (gres) specification"; `sbatch` was the fix. Also planned for the L4 from the course page without a boot test.
- Outcome: success.
- Follow-ups: DS to open `/scratch/dns5508/frame_bank` and `airsim/envs` for reading (`chmod o+rX`), so all three use the same frames. PP has not yet looked at `rgb.png` by eye.

## 2026-11-13 21:50  |  docs  |  Merged main into the PR #1 branch
- Member: PP
- Request: "ok go ahead" (fix PR #1's conflicts with main instead of opening a new PR).
- Action: laptop. Committed the pending HPC edits (job template now defaults to the A100 with `--requeue`; smoke test falls back to `/scratch/$USER`; CLAUDE.md lists the partitions). Merged origin/main (d339395) into S/plan-v1.1-skeleton. Conflicts: `.gitignore` (kept both sides) and `logs/AI_LOG.md` (deleted here, modified on main).
- Assumptions: "DS" as Devansh's initials for the Member field [UNVERIFIED].
- Decisions & alternatives: moved DS's Oct 31 entry into this file unchanged except for the added Member line, and removed `logs/AI_LOG.md`, since PLAN.md §7 and TASKS.md name `AGENT_LOG.md` as the single log. Rejected: keeping two log files. Dropped the assistant's separate `hpc/README.md` and put its partition table and the `srun` workaround into DS's `hpc/instructions.md`; its two `logs/AI_LOG.md` references now point here.
- Deviations from PLAN.md, recorded here because PLAN.md is read-only for the assistant: (1) simulator is Project AirSim v1.0.1, not AirSim/Colosseum (D3; needs team sign-off, raised by DS); (2) the A100 is the working GPU, not the L4 (D17); (3) `sim/capability-test` has a week plan that orders frame-bank vision work before the closed loop, and names Qwen3-VL 4B/8B where D8 says 2B/8B.
- Evidence: `pytest`: 7 passed after the merge. `bash -n` on the HPC scripts: ok.
- Outcome: success.
- Follow-ups: DS reviews and merges PR #1, then updates `sim/capability-test` on top of it (its `TASKS.md` and `.gitignore` changes will need merging). Week-plan dates (Nov 2 to 8) have passed and need re-dating.

## 2026-11-13 22:10  |  perception  |  Depth clipping, corridor statistics, L1 labels, Depth Anything V2 wrapper
- Member: PP
- Request: "Lets start!" on the perception and gate workstream (week-plan tasks W-S2, W-B1 code, W-B3), then "commit and create PR".
- Action: laptop. New branch B/W-B1-perception on top of S/plan-v1.1-skeleton (needs interfaces.py). Wrote perception/depth_utils.py, gate/labels.py, perception/depth_anything.py and three test files; added a `perception` extra (torch, transformers, pillow) to pyproject.toml.
- Assumptions: Hugging Face checkpoint `depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf` is the Small metric outdoor model (it loaded and ran; the model card was not read) [UNVERIFIED]. The frame bank's depth is distance along the pixel ray (`DEPTH_PERSPECTIVE` in hpc/capability/capability_test.py on sim/capability-test), while Depth Anything predicts planar depth [UNVERIFIED for the model; camera field of view unknown].
- Decisions & alternatives: (1) corridor = fixed central window (rows 0.4 to 0.6, columns 0.45 to 0.55), the same as `corridor_min()` in the capability test, so values match `corridor_min_depth_m` in frames.jsonl; rejected for now: the projected flight tube of PLAN.md §2, which needs intrinsics and T_h. (2) depth clipped at 100 m; sky, saturated (65504 m) and invalid pixels count as far, not near. (3) sky masked by range, not by segmentation ID as the week plan suggests, because the sky's ID is unknown until the frames are readable. (4) `d_safe` is an argument with no default: PLAN.md gives no value, and it sets the positive rate, so the team must choose it. (5) "obstacle" means strictly closer than `d_safe`. (6) a corrected estimate at or below zero is clamped to near, not treated as invalid.
- Evidence: pytest 29 passed (7 old, 22 new). Real model on the Mac (MPS, torch 2.14.1, transformers 5.20.2): depth (225, 400) float32, 5.51 to 17.94 m on a synthetic image, median 22 ms per frame over 10 calls after a 9.5 s first call.
- Outcome: partial. Code and unit tests done; the model has not been run on a simulator frame, so accuracy is unknown.
- Follow-ups: run on real bank frames once /scratch/dns5508 is readable; eyeball 20 positives and 20 negatives (W-B3); scale check with the perspective-to-planar conversion (W-B2); team picks `d_safe`.

## 2026-11-13 22:45  |  perception  |  Frame-bank reader, episode split, resumable bank runner
- Member: PP
- Request: "go ahead!" on writing the bank runner (rest of W-B1) and the split script (W-S1) before the frames are readable, then "go ahead" on commit and push.
- Action: laptop. Wrote perception/frame_bank.py, perception/bank_split.py, perception/run_bank.py and tests/test_frame_bank.py; moved pillow into the core dependencies (the reader needs it); added two quick commands to CLAUDE.md.
- Assumptions: the bank's file layout and JSONL fields, read from hpc/capability/capability_test.py on sim/capability-test and not from the real files [UNVERIFIED until run on the bank]. `_rgb.png` files hold correct colours, i.e. the simulator's image was BGR when cv2 wrote it [UNVERIFIED; a red/blue swap would show in the eyeball check].
- Decisions & alternatives: (1) split by episode, stratified by (obstacle, condition), 70/15/15 through an interleaved 20-slot pattern with a random offset per stratum, so small strata still reach val and test; rejected: per-stratum rounding, which sends strata of 3 or fewer entirely to train. (2) kept conditions default to clear, dusk and rain; fog and snow excluded, following hpc/instructions.md §3.5; this is the "decide kept conditions" part of W-S1 and the team can override it with `--keep`. (3) the split script refuses to overwrite its output (PLAN.md D13). (4) the runner stores both corridor minimums and no label, so labels can be recomputed for any `d_safe`. (5) estimated depth maps are saved only with `--save-depth` (about 3.5 GB for 19k frames as float16). (6) the runner processes all conditions; filtering by split and condition happens downstream.
- Evidence: pytest 39 passed (10 new). End to end on the Mac with the real model on a fake bank of 30 episodes and 120 frames: split file written and a second run refused; runner did 50 frames with `--limit`, then 70 on resume, 120 rows with no duplicates, median 21.8 ms per frame; two lines appended to logs/runs.jsonl, then removed because they were not experiments.
- Outcome: partial. Works on a fake bank; not yet run on real frames.
- Follow-ups: on HPC once /scratch/dns5508 is readable: write configs/bank_split.yaml, commit it, run the bank job (ask before launching), then the eyeball check and the scale check. Episode ids in the split refer to the CityEnviron bank only.

## 2026-11-14 00:20  |  perception  |  First run on the real frame bank (5 frames, CPU); runner ordering fix
- Member: PP
- Request: check the reader against the real bank before a GPU job; then "lets fix, pull and start our work".
- Action: HPC (psp8502, CPU node b-10-16, env `pa`): `pip install -e ".[perception]"`; `python -m perception.bank_split` wrote configs/bank_split.yaml on HPC; `python -m perception.run_bank --limit 5`. Laptop: in perception/run_bank.py the depth map is now saved before the frame's row is written.
- Assumptions: none new.
- Decisions & alternatives: fixed before the full run, not after, at the user's request.
- Evidence: bank has episodes.jsonl, frames.jsonl and 57,801 files in frames/ (19,267 x 3); field names match the reader. Split: train 202, val 40, test 39, excluded 130 (= fog 53 + snow 77), checksum 6e3af228c9f6. Frame 0000_000: corridor_min_gt 41.71875, identical to the bank's `corridor_min_depth_m`; corridor_min_est 35.13 (about 16% short). CPU inference about 91 ms per frame after a 25 s first call. pytest 39 passed after the fix.
- Mistake (assistant): the runner wrote a frame's row before saving its depth map, so a job killed between the two would have marked a frame done with no depth map. Found on re-reading the code when the user asked about checkpoints; no data was affected.
- Surprise, not yet explained: corridor_min_gt is exactly 41.71875 in frames 0000_000 to 0000_004 while the recorded pose advances 0.4 m per step. Possible stale frames after a pose change (hpc/instructions.md §3.5 reports them for job 15689) [UNVERIFIED]; a full head-on episode is being checked.
- Outcome: success for the reader and split; the stale-frame question is open.
- Follow-ups: commit configs/bank_split.yaml from the HPC copy; full bank job (launched by the user); resolve the stale-frame question before trusting labels.
