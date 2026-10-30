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
