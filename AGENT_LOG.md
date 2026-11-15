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

## 2026-11-15 14:55  |  docs  |  Split the plan: one-page PLAN.md + milestone TASKS.md; old files renamed
- Member: SB
- Request: SB asked to replace the two very specific plan files with a high-level overview plus a task split "in chunks so that we can keep track", keeping the old files under renamed filenames. SB chose: milestones with gates, and no dates (dependency order only). SB also asked to build this on DS's branch plus the commits from PP's branch relevant to the plan.
- Action: laptop, branch `S/plan-v2-milestones` off `main`. Merged `origin/sim/capability-test` (5 commits, capability tests and `hpc/instructions.md` findings) and `origin/S/plan-v1.1-skeleton` (8 commits, PLAN v1.1, repo skeleton, `interfaces.py`, smoke test). Both merged with no conflicts; the `TASKS.md` auto-merge kept PP's re-dated rows plus DS's `W-*` week section. Then `git mv PLAN.md PLAN_DETAILED.md`, `git mv TASKS.md TASKS_DETAILED.md`, wrote a new one-page `PLAN.md` and a new milestone `TASKS.md` (M0 to M7, each with a gate and a fallback), and updated the filename references in CLAUDE.md, README.md, configs/README.md, interfaces.py, hpc/*, report/*, logs/next_steps_PP.md, logs/REFERENCES.md.
- Assumptions: workstream owners are A = DS, B = PP, C = SB, inferred from who authored which branch, which closes Q1 [UNVERIFIED until the team confirms]. All 8 commits of PR #1 were treated as "relevant to the plan", not only the three that touch PLAN.md, because dropping the skeleton commit would leave CLAUDE.md and PP's perception branch pointing at files that do not exist.
- Decisions & alternatives: `PLAN_DETAILED.md` stays the single source of truth and the new `PLAN.md` deliberately contains no decision, metric or risk of its own, only pointers, so the two files cannot drift. Rejected: `PLAN_old.md` as the name (it implies stale, and the file is still authoritative); rejected keeping the dates with an OVERDUE flag (SB chose order only); rejected a per-person Now/Next/Later board (it loses the "done when" checks the Agent Track grade leans on). Kill switches that were calendar dates are now conditions on a gate, which also fixes their having all expired.
- Evidence: a reference checker over both new files resolved every `D*`, `§*`, `Q*`, `E*`, old task id and file path against `PLAN_DETAILED.md` and `TASKS_DETAILED.md`, with one expected miss (`frames.jsonl`, which lives on HPC scratch and is gitignored). `interfaces.py` parses (`ast.parse`). `pytest` could not run: no venv on this laptop yet and system Python has no pytest, so the interface tests are unverified here; the only code changes are docstrings and comments.
- Mistake (assistant, earlier in this session): deleted the local `PLAN.md` and `AGENT_LOG.md` on SB's instruction before checking the folder was not a clone of the repo, and before checking git history existed. It did not: the folder was an unversioned copy. Both files were recovered only because the repo had them. Should have looked at what the files were and warned first.
- Mistake (assistant, earlier in this session): told SB that `S/plan-v1.1-skeleton` was his branch, inferring ownership from the `S/` prefix. The `S/` means shared tasks (`S-01` to `S-12`) and all 8 commits are PP's. Corrected after checking commit authors.
- Risky action flagged to SB: this commit rewrites two shared, team-agreed documents and renames them, which will conflict with PP's `B/W-B1-perception` branch (its `TASKS.md` and `CLAUDE.md` edits) and with DS's `sim/capability-test`. CLAUDE.md rule 0 makes `PLAN.md` read-only for the assistant; SB asked for this change explicitly, and `PLAN_DETAILED.md` §0 still requires a teammate's approval before it lands on `main`.
- Outcome: success, not pushed.
- Follow-ups: team to confirm the owner table (Q1) and sign off on Project AirSim (D3, Q4, chunk M0.5); PP to rebase `B/W-B1-perception` onto this and land M1.5; the `W-*` rows from DS's week plan are now folded into M1, M2 and M4, so `TASKS_DETAILED.md` is reference only.

## 2026-11-15 15:35  |  vlm  |  Qwen3-VL judge, overlay and crop, surrogate, latency benchmark (commit 7c59ece)
- Member: SB
- Request: "ok lets do m2 now". SB also said earlier that M1 and M2 were done; a check of every branch found `vlm/` empty, so M2 was started from nothing (recorded in the previous entry).
- Action: laptop, branch `C/M2.1` off `S/plan-v2-milestones`. Wrote `vlm/judge.py`, `vlm/overlay.py`, `vlm/surrogate.py`, `vlm/bench_latency.py`, `hpc/vlm_latency.sbatch`, `tests/test_vlm.py`; added the `vlm` extra and pillow to `pyproject.toml`; README section with a "pending" latency table; TASKS.md M2 rows set to DOING/DONE as the evidence allows. Created the laptop venv (`uv venv --python 3.11 .venv`, `pip install -e ".[dev]"`).
- Assumptions, all marked until the A100 job runs: (1) `processor.apply_chat_template` with `{"type": "image"}` placeholders followed by `processor(text=[...], images=[...])` expands the image tokens for `qwen3_vl` the same way as for Qwen2-VL [UNVERIFIED on a live model; the processor source has `replace_image_token`, which is the mechanism]. (2) bfloat16 on the A100 holds both the 2B and the 8B in 40 GB [UNVERIFIED; the benchmark catches an OOM per model and keeps the other's numbers]. (3) Model ids verified on the Hugging Face API on 2026-11-15: Qwen/Qwen3-VL-2B-Instruct, 4B, 8B, all HTTP 200, Apache 2.0, architecture `Qwen3VLForConditionalGeneration`.
- Decisions & alternatives: HF transformers, not vLLM, for the first measurement (matches PP's working env; gives the logits directly; vLLM stays as the rank-8 latency lever). One forward pass reading next-token logits instead of `generate()` (deterministic, cheaper, and the `confidence` field of `VLMAnswer` is exactly `logit(yes) - logit(no)`); several surface forms of yes/no are pooled with log-sum-exp so the tokenizer's choice of " yes" vs "yes" cannot flip the sign. The highlighted region for the MVP is the fixed corridor window from `perception.depth_utils` (rows 40 to 60%, cols 45 to 55%), not a projected flight tube, so the VLM question and the L1 label refer to the same pixels; constants are duplicated in `vlm/overlay.py` with a test that asserts equality once B's branch is merged. The surrogate takes its truth through `set_truth()` from the evaluation loop, because `ask()` sees only images and must never read ground truth itself (CLAUDE.md rule 5). Rejected: making `ask()` take an obstacle-family argument (would change the shared interface).
- Evidence: `pytest`: 19 passed, 1 skipped (the cross-check against `perception.depth_utils`, absent on this branch). Overlay rendered on a synthetic scene with a 1 px wire at row 108 and viewed: box on the corridor, wire visible through the tint, crop shows the wire as a 4 px band. transformers 5.23.0 source, installed in a scratch venv without torch: `class Qwen3VLForConditionalGeneration` at `modeling_qwen3_vl.py:1241`, exported via `models/__init__.py:439`, `hasattr(transformers, "Qwen3VLForConditionalGeneration")` is True; `ProcessorMixin.__call__(images=, text=)` at `processing_utils.py:671`; `modeling_utils.py:1404` warns that `torch_dtype` is deprecated in favour of `dtype`, which the code uses.
- Risky behaviour: none. No HPC job launched; nothing pushed.
- Outcome: partial. Code and tests done; M2.1's "done when" (latency numbers in README) is not met because the GPU run has not happened.
- Follow-ups: SB runs `hpc/vlm_latency.sbatch` on the A100 (setup in its header) and pastes the printed table into README. If the 2B's yes-fraction on noise frames is 0.0 or 1.0 the prompt or token pooling needs a look before M2.4. M2.4 needs `perception.frame_bank` and `configs/bank_split.yaml` from PP's branch (M1.5). DS can build M4.6 against `vlm.surrogate.SurrogateJudge` now.
