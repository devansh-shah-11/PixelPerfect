# Next steps for DS

## Update Fri Nov 20: overall picture, pending tasks, immediate actions

### Overall picture
| Milestone | Status now |
|---|---|
| M0 Foundations | Nearly done. Open: SB's smoke test (M0.3), Project AirSim sign-off (M0.5, D3). |
| M1 Frame bank | Gate passed. `da2_small.jsonl` (19,267 rows, job 36206) is now on the laptop at `results/bank_runs/` (gitignored). Open: M1.5 merge to `main`, M1.8 clean labels, M1.9 depth diagnosis. |
| M2 VLM | Measured (jobs 33620, 35101, 35102). Every standalone row fails the M2 gate. The verdict that matters (VLM recall on the frames depth misses) still needs clean GT from M1.8. D23 revision needs a teammate's approval. |
| M3 Saved-frame gate | Code ready and tested, not run on the real bank. Blocked on M1.8. |
| M4 Closed loop | AirSim works: 10/10 true-depth `never` flights reached the goal, 0 crashes (job 36229, n=10, a sanity check only). Headroom with the real depth model (target 15 to 30% crashes) is unmeasured. |
| M5 to M7 | Not started. |

### da2_small findings, FIRST LOOK ON RAW LABELS (superseded: clean-label, benchmark and gate results are in the 2026-11-20 entry of `AGENT_LOG.md`, `results/benchmark.md`, `results/depth_diag/`, `results/gate_val.json/gate_report.md`)
- The rerun reproduces the old numbers exactly (val AbsRel raw 2.418, median scale 0.790, affine 0.525 and +3.83 m, val 1.588), so the bank run is deterministic and the earlier analysis stands.
- Miss rate at d_safe 6 / 8 / 10 m: train 76.0 / 76.4 / 77.3%, val 86.5 / 85.5 / 85.3% (raw labels; scaled depth 86.1 / 85.2 / 85.0%).
- By type: ball, glass, pole and both wires 100% missed; 1 m boxes about 90%; building_block and white wall 0 to 5%, with their false stops (building 24 to 29%, wall 9 to 13%). Rain misses most (83 to 84%), dusk least (67 to 70%).
- Caveats: these are raw-label figures. Episodes with no obstacle still show 53 train / 24 val "obstacle" frames, so labels are noisy until M1.8. Val has 0 positive frames for box_1m_black and only 2 to 10 for the white wall, so val per-type numbers are thin.

### High-level pending tasks
1. Clean labels and depth diagnosis (M1.8, M1.9): needed before any miss rate or gate is trusted.
2. M2 verdict and judge pick (M2.4, M2.6, M2.7): VLM recall on depth's misses, quantized accuracy, `results/benchmark.md`.
3. Gate training and held-out family (M3.3 to M3.5).
4. Closed-loop headroom and policies (M4.7, M5.1 to M5.3): never-ask crash rate with real depth, threshold and oracle policies, real VLM in the flight job.
5. Crash-cause tool, then the two headline experiments (M5.4, M6.3, M6.4).
6. Final test runs, report, demo (M6.8, M7).
### To look at and improve later (not for now; not a plan item until a Decision-log row says so)
- **Thin val coverage:** val has 0 positive frames for box_1m_black and 2 to 10 for the white wall, so per-type val miss rates and gate metrics for those types are unreliable. Options when there is time: report those types with train-fold cross-validation by episode, or note them as not evaluable. The split is frozen (`configs/bank_split.yaml`), so this cannot be fixed by resplitting.
- **Raw-label noise:** obstacle-free episodes show 53 train / 24 val "obstacle" frames. Check after M1.8 whether `--gt` removes them; if some remain, find the cause (rain, ground or sky in the corridor window) before training any gate.
- **Building block and white wall:** depth sees them (0 to 5% missed) but they cause the false stops; a gate that only targets misses may hide that cost.
Team: D23 approval, D3 sign-off (M0.5), M1.5 merge, trim result PNGs and the AirSim config artifacts from PR #3.

### Reproducing gitignored files (for the next person or their Claude)
The three `results/bank_runs/*.jsonl` files ARE now committed (2026-11-20); `*.pkl` is the only ignored item. If they are missing or stale, regenerate with the commands below, in order. HPC paths use `/scratch/dns5508`; set `BANK=/scratch/dns5508/frame_bank/CityEnviron`. Activate first: `eval "$(conda shell.bash hook)"; conda activate /scratch/dns5508/envs/pa; cd /scratch/dns5508/PixelPerfect`. Everything is deterministic (frozen seeds and split), so reruns should match the logged numbers.

| Missing file (gitignored) | Needed by | Regenerate (where) |
|---|---|---|
| `results/bank_runs/da2_small.jsonl` (depth over the bank) | analyze_bank, diagnose_depth, benchmark, gate.train | `sbatch hpc/run_bank.sbatch` (A100, about 11 min), writes `/scratch/dns5508/bank_runs/da2_small.jsonl` |
| `results/bank_runs/bank_gt.jsonl` (clean labels) | analyze_bank `--gt`, gate.train | `python -m gate.bank_gt --bank $BANK --out /scratch/dns5508/bank_runs/bank_gt.jsonl` (HPC CPU, minutes) |
| `results/bank_runs/image_features.jsonl` | gate.train `--image-features` | `python -m gate.image_features --bank $BANK --out /scratch/dns5508/bank_runs/image_features.jsonl` (HPC CPU, minutes) |
| `results/gate_*/gate_*.pkl` (trained gates, served by `gate/model.py`) | the closed loop with a learned gate (M5) | `python -m gate.train --run <da2_small> --gt <bank_gt> --split configs/bank_split.yaml --image-features <image_features> --d-safe 8 --out results/gate_val.json` (seconds, CPU; `--out` makes a directory). The best model is `gate_mlp_depth+temporal+image.pkl` |

Copy the three bank files to the laptop with `rsync -avP dns5508@<hpc-login-host>:/scratch/dns5508/bank_runs/<file> results/bank_runs/`. The frame bank itself (images) lives only at `$BANK` on HPC scratch; the laptop `Framebank/` has metadata only. `gate_report.md` and `gate_results.json` are small and are kept in git; only the pickles are ignored.

### Run-book (run one step at a time; paste the output back before the next step)
Paths: `HPC` = `/scratch/dns5508`, bank = `/scratch/dns5508/frame_bank/CityEnviron`. The laptop `Framebank/` holds metadata only (no images), so image-reading steps run on the HPC. Replace `<hpc-login-host>`.

**Step 1. HPC CPU node, minutes, no GPU: clean labels (M1.8).** Reads segmentation masks. Do not leave an idle session open.
```
cd /scratch/dns5508/PixelPerfect && git pull
python -m gate.bank_gt --bank /scratch/dns5508/frame_bank/CityEnviron --out /scratch/dns5508/bank_runs/bank_gt.jsonl
```
Then, on the laptop:
```
rsync -avP dns5508@<hpc-login-host>:/scratch/dns5508/bank_runs/bank_gt.jsonl results/bank_runs/
```
Check: 19,267 rows (`wc -l`). Impact: every later miss rate, label and gate uses it.

**Step 2. Laptop, seconds: depth diagnosis and clean miss rates (M1.9 Part A, M1.8).**
```
.venv/bin/python -m perception.diagnose_depth --run results/bank_runs/da2_small.jsonl --gt results/bank_runs/bank_gt.jsonl --split configs/bank_split.yaml --pixels 0
.venv/bin/python -m perception.analyze_bank --run results/bank_runs/da2_small.jsonl --gt results/bank_runs/bank_gt.jsonl --split configs/bank_split.yaml --d-safe 6 8 10
```
Look for: is the 2.42 AbsRel still large on scene (non-obstacle) frames (a depth bug) or only on obstacle frames (miss-inflated)? Do the 53 train / 24 val obstacle-free "obstacle" frames disappear? Part B (`--pixels N --bank`, contact sheet) needs the HPC bank; only if Part A is unclear.

**Step 3. Laptop, seconds: benchmark and the D23 verdict (M2.4, M2.7).** (The old line with `--summarize-only` was wrong; that flag belongs to `vlm.eval_bank`, not `eval.benchmark`.)
```
.venv/bin/python -m eval.benchmark --vlm results/vlm_eval.jsonl --run results/bank_runs/da2_small.jsonl --latency results/vlm_latency_35101.json --out results/benchmark.md
```
Trust it only after step 2. Look for: VLM recall on the frames depth misses at least 0.60 with false alarms at most 0.15 (D23); which variant wins on the rule (D22). A `results/benchmark.md` from before the clean labels exists; overwrite it.

**Step 3b. HPC A100, needs your OK, over an hour (quantized accuracy, so the benchmark can score 8B@8bit and 8B@4bit).** First `pip install -e ".[quant]"` in the `pa` env. Small run first (add `,PER_TYPE=50` and use `OUT=results/vlm_eval_quant_quick.jsonl`), then:
```
sbatch --export=ALL,MODELS="Qwen/Qwen3-VL-8B-Instruct@8bit Qwen/Qwen3-VL-8B-Instruct@4bit",CONDITIONS="both both_blocked",ONE_AT_A_TIME=1,OUT=results/vlm_eval_quant.jsonl hpc/vlm_eval_bank.sbatch
```
Then rsync it to `results/` and rerun step 3 with `--vlm results/vlm_eval.jsonl results/vlm_eval_quant.jsonl`. Steps 1, 2, 4 and 5 do not need a rerun.

**Step 4. HPC CPU node, minutes: image features (M3.4).**
```
python -m gate.image_features --bank /scratch/dns5508/frame_bank/CityEnviron --out /scratch/dns5508/bank_runs/image_features.jsonl
```
Then rsync it to `results/bank_runs/` as in step 1.

**Step 5. DONE 2026-11-20 on the laptop (paths below are the laptop copies; on the HPC use `/scratch/dns5508/bank_runs/...`). Note `--out` creates a directory named `*.json`. Train and evaluate the gate (M3.3 to M3.5).**
```
.venv/bin/python -m gate.train --run results/bank_runs/da2_small.jsonl --gt results/bank_runs/bank_gt.jsonl --split configs/bank_split.yaml --image-features results/bank_runs/image_features.jsonl --d-safe 8 --out results/gate_val.json
.venv/bin/python -m gate.train --run results/bank_runs/da2_small.jsonl --gt results/bank_runs/bank_gt.jsonl --split configs/bank_split.yaml --image-features results/bank_runs/image_features.jsonl --d-safe 8 --holdout-family wire_3cm_axis_z --out results/gate_holdout_wire3.json
```
Val only; never the test split. AUROC above 0.99 means suspect leakage, not success. The gate must beat the depth-variance baseline on val.

**Step 6. Team (no commands).** Ask for D23 approval, D3 sign-off (M0.5), review and merge of PR #3 (M1.5), and trim the PNGs and AirSim configs from the PR.

**Step 7. A100, needs your OK first, about 15 min: never-ask headroom (M4.7).** Command to be written once steps 1 to 5 are read; it uses `eval.sweep` with the depth stack on val seeds and `configs/sweeps/airsim_bench.yaml`.

---

# Earlier notes (written Tue Nov 17)

What changed in the working tree on `C/M2.1`, how to test it, how to commit it in small steps, what can go wrong, and what comes after. Nothing here has been committed, and **nothing has run on a GPU or in Project AirSim yet**, so there are no new numbers.

Contents: [1. Where we are](#1-where-we-are) · [2. What was built](#2-what-was-built) · [3. Test on the laptop](#3-test-on-the-laptop) · [4. Commit in four steps](#4-commit-in-four-steps) · [5. Run on the HPC](#5-run-on-the-hpc) · [6. Errors you may hit](#6-errors-you-may-hit) · [7. What comes next](#7-what-comes-next)

---

## 1. Where we are

| Milestone | Status |
|---|---|
| M0 Foundations | Nearly done. Open: SB's smoke test, the Project AirSim sign-off (row D21). |
| M1 Frame bank | Gate passed. Open: merge into `main` (M1.5); confirm the clean labels and the depth diagnosis ran (M1.8, M1.9). |
| M2 VLM sees obstacles | Measured (job 33620). **Verdict open:** it waits for the team to approve D23 and for the fast-miss numbers to be logged. New code for variants and the benchmark is written (M2.6, M2.7), not run. |
| M3 Saved-frame results | Code written and tested. Not run on the real bank. |
| M4 Closed loop | Works on the Plan B (raycast) simulator. On Project AirSim: wires and pause/step verified (job 33776), **seeds frozen today**, start poses not yet checked, true-depth sweep not yet run. |
| M5 First curve | Not started. |
| M6 Research results, M7 Report | Not started. |

**The one hard deadline:** final test runs launch Mon Nov 30. That is 13 days away.

**Waiting on a teammate:** rows D21 to D24 in `PLAN_DETAILED.md` are marked OPEN until one of them approves. The seed file is already frozen (D24), so tell them today.

---

## 2. What was built

Three parts. Each one is its own commit in section 4.

### Part A: VLM variants and prompts (D22, D23)

| File | What changed |
|---|---|
| `vlm/judge.py` | A model name ending in `@4bit` or `@8bit` loads the quantized model (bitsandbytes), e.g. `Qwen/Qwen3-VL-8B-Instruct@4bit`. Two new prompts: `fill` (says a wall filling the region counts as blocked) and `free` (asks the opposite question; the sign is flipped). A cut `tau`: the judge says yes when confidence > `tau`. |
| `vlm/eval_bank.py` | New conditions `both_fill` and `both_free`. New flag `--one-at-a-time` (load a model, evaluate, free it), because the four variants don't fit the GPU together. |
| `vlm/bench_latency.py` | Accepts the `@4bit` / `@8bit` names. `--depth` adds the depth model as a row. |
| `eval/factory.py` | Sweep configs can name a prompt (`prompt_key`) and a quantized model. New fast stack `clear`. |
| `perception/stand_ins.py` | `ClearStack`: no depth model, always says "clear". Used to fly "VLM only". |
| `pyproject.toml` | New extra `quant` (bitsandbytes). HPC only. |
| `hpc/vlm_latency.sbatch`, `hpc/vlm_eval_bank.sbatch` | New defaults and commands in the headers. |

Why each model has its own cut: a 4-bit model keeps its accuracy but its confidence scores can shift (arXiv 2607.24440, 2609.06922), so one shared cut would be unfair. 8-bit changes the scores very little (arXiv 2208.07339).

### Part B: Seeds, start poses, scenario freeze (D24)

| File | What changed |
|---|---|
| `configs/seeds.yaml` | **New and frozen.** train 100000 to 100999, val 101000 to 101499, test 101500 to 101999. Never edit it. |
| `sim/scenario.py` | Defaults are now 1000 / 500 / 500. New second freeze for scenario settings: `--freeze-scenario`. |
| `sim/airsim_backend.py` | `starts`: a list of start poses `[x, y, yaw_deg, clear_m]`. A flight uses pose `seed % number of poses`. A flight longer than the pose's clear distance is refused. |
| `eval/sweep.py` | `--set key=value` overrides a config value. Test seeds are refused unless the scenario comes from the frozen file. |
| `eval/run_episode.py` | Logs which start pose a flight used. Default seed is now a val seed. |
| `configs/sweeps/airsim_starts_smoke.yaml` | New. Checks the five start poses (10 flights). |
| `configs/sweeps/airsim_true_depth.yaml` | Now 40 m flights from five poses (was 20 m from one). |
| `configs/sweeps/airsim_bench.yaml` | New. Flight side of the benchmark. |
| `configs/sweeps/raycast_smoke.yaml` | Now takes val seeds from the frozen file. |

### Part C: Benchmark table

| File | What it does |
|---|---|
| `eval/benchmark.py` | New. Builds one table: depth only, VLM only, depth + VLM, for every VLM variant. Columns: recall, false alarms, recall on depth's misses, latency, GPU memory. Ends with a pick. Reads result files only, so it runs on the laptop. |
| `tests/test_benchmark.py` | New. 7 tests on made-up rows. |

### Docs
`README.md`, `TASKS.md`, `CLAUDE.md` (quick commands), `PLAN_DETAILED.md` (D24 row: new counts), `AGENT_LOG.md`, this file.

---

## 3. Test on the laptop

All of this is CPU only. The tests take about 15 seconds; check 3 below takes about a minute.

```bash
cd "<repo>"
uv venv .venv --python 3.11 && uv pip install -e ".[dev]"   # skip if .venv exists. Do NOT add [quant] on the Mac
.venv/bin/python -m pytest                                   # expect: 137 passed, 1 skipped
```

The one skipped test needs torch. That is expected on the laptop.

Then four quick checks by hand:

```bash
# 1. The frozen seed file loads (checksum and no overlap are checked inside)
.venv/bin/python -c "from sim.scenario import load_seed_lists as l; print({k: len(v) for k, v in l('configs/seeds.yaml').items()})"
#    expect: {'train': 1000, 'val': 500, 'test': 500}

# 2. A scenario prints
.venv/bin/python -m sim.scenario --show 101000

# 3. A small sweep on the Plan B simulator, with an override (writes to /tmp, logs nothing)
.venv/bin/python -m eval.sweep --config configs/sweeps/raycast_smoke.yaml --no-log --set out=/tmp/pp_smoke
#    expect: a table with never / always / random / threshold / oracle, judge "surrogate VLM"

# 4. The benchmark's flight table from that sweep
.venv/bin/python -m eval.benchmark --flights /tmp/pp_smoke --out /tmp/pp_benchmark.md --no-log
```

These use the stand-in depth and the surrogate judge. **They are plumbing checks, not results.**

---

## 4. Commit in four steps

This order was checked: with only the files of each step added on top of the last commit, the tests pass.

| After | Tests |
|---|---|
| today's `HEAD` | 118 passed |
| commit 1 | 124 passed, 1 skipped |
| commit 2 | 130 passed, 1 skipped |
| commit 3 | 137 passed, 1 skipped |

Note: `pytest` runs on your working tree, which already has everything, so it will say 137 at every step. The table is what a teammate would see checking out each commit.

Before each commit: `git diff --cached --stat` and check nothing large or secret is staged.

**Commit 1: VLM variants and prompts**
```bash
git add vlm/judge.py vlm/eval_bank.py vlm/bench_latency.py eval/factory.py perception/stand_ins.py \
        pyproject.toml hpc/vlm_latency.sbatch hpc/vlm_eval_bank.sbatch tests/test_vlm.py tests/test_eval_bank.py
git commit -m "vlm: quantized 8B variants, second prompt round, per-model cut" \
           -m "On-prem deployment needs a small-memory judge. Adds @4bit/@8bit model specs"
```

**Commit 2: seeds, start poses, scenario freeze**
```bash
git add sim/scenario.py sim/airsim_backend.py eval/sweep.py eval/run_episode.py configs/seeds.yaml \
        configs/sweeps/airsim_true_depth.yaml configs/sweeps/airsim_starts_smoke.yaml \
        configs/sweeps/airsim_bench.yaml configs/sweeps/raycast_smoke.yaml tests/test_sim_loop.py
git commit -m "sim: freeze seed lists, five start poses, scenario freeze, sweep overrides"
```
Check that `configs/seeds.yaml` is in this commit. It is a new file, so `git add -u` would miss it.

**Commit 3: benchmark table**
```bash
git add eval/benchmark.py tests/test_benchmark.py
git commit -m "eval: benchmark table for depth only, VLM only, depth + VLM"
```

**Commit 4: docs**
```bash
git add README.md TASKS.md CLAUDE.md PLAN_DETAILED.md AGENT_LOG.md logs/next_steps_DS.md
git commit -m "docs: benchmark and seed-freeze instructions, D24 counts, agent log"
```

After the commits, per `CLAUDE.md`: add a `logs/PROGRESS.md` entry per commit, check `git status` is clean, and back up (commit 2 touches `configs/`). Push only when you decide to. The HPC needs the push before it can `git pull`.

---

## 5. Run on the HPC

All on Cloud Bursting, partition `c12m85-a100-1`. **Ask the team before each job; job 4 may pass an hour.** Run them in this order, since each one is cheap and tells you whether the next will work.

**Setup, once**
```bash
cd /scratch/$USER/PixelPerfect && git pull
source $(conda info --base)/etc/profile.d/conda.sh     # every new shell/job; never `conda init`
conda activate /scratch/$USER/envs/pa                   # the existing env; do not create a new one
pip install -e ".[dev,vlm,quant,perception]"
python -c "import bitsandbytes, torch; print(bitsandbytes.__version__, torch.cuda.is_available())"   # on a GPU node
```

| # | Job | Command | Time | Gives |
|---|---|---|---|---|
| 1 | Latency and memory of all five systems | `sbatch hpc/vlm_latency.sbatch` | about 20 min | `results/vlm_latency_<job>.json`. **Also the first proof that the quantized models load.** |
| 2 | New prompts on 2B and 8B | `sbatch hpc/vlm_eval_bank.sbatch` (same `OUT` as job 33620, so only the two new prompts are asked) | about 25 min | more rows in `results/vlm_eval.jsonl` |
| 3 | Quantized 8B, small sample | the `PER_TYPE=50` command in the header of `hpc/vlm_eval_bank.sbatch` | unknown, short | how many seconds per call |
sbatch --export=ALL,MODELS="Qwen/Qwen3-VL-8B-Instruct@8bit Qwen/Qwen3-VL-8B-Instruct@4bit",CONDITIONS="both both_blocked",ONE_AT_A_TIME=1,PER_TYPE=50,OUT=results/vlm_eval_quant_quick.jsonl hpc/vlm_eval_bank.sbatch
| 4 | Quantized 8B, full | the last command in that header | **unknown, maybe over 1 h** | rows for `@8bit` and `@4bit` |
sbatch --export=ALL,MODELS="Qwen/Qwen3-VL-8B-Instruct@8bit Qwen/Qwen3-VL-8B-Instruct@4bit",CONDITIONS="both both_blocked",ONE_AT_A_TIME=1 hpc/vlm_eval_bank.sbatch
| 5 | Start-pose check | `sbatch --export=ALL,PA_ENV=/scratch/dns5508/envs/pa,CMD="python -m eval.sweep --config configs/sweeps/airsim_starts_smoke.yaml" hpc/airsim_job.sbatch` | about 15 min | which of the five poses are really open |

Then build the table (laptop or a CPU node):
```bash
python -m eval.benchmark --vlm results/vlm_eval.jsonl --run <bank runs>/da2_small.jsonl \
    --latency results/vlm_latency_*.json --out results/benchmark.md
```

First `sbatch hpc/run_bank.sbatch` if you don't have `da2_small.jsonl` (see section 7, step 1). D23 (revised) defines the three tables and the pass and pick rules; write them down before scoring.

**How to read job 5.** Open `results/sweeps/airsim_starts_smoke/results.jsonl`. Each line has `extra.start_index` (0 to 4) and `extra.collision_source`. A pose passes if both of its flights reached the goal. If a flight crashed with `collision_source: "scenery"`, the map is in the way: delete that pose from the `starts` list in all three AirSim configs, then rerun.

**Log every job** in `AGENT_LOG.md` with its job id and commit. Two earlier results are still not logged: the fast-miss numbers (2B 15 to 18%, 8B 79%) and the 5-flight AirSim smoke run.

---

## 6. Errors you may hit

### Laptop

| You see | Cause | Fix |
|---|---|---|
| `.venv/bin/python: no such file` | No venv on this machine | The `uv venv` line in section 3 |
| `pip install` fails on `bitsandbytes` | It is for CUDA GPUs | Don't install `[quant]` on the Mac |
| `1 skipped` in pytest | The test needs torch | Expected |
| `FileExistsError: configs/seeds.yaml exists and is frozen` | You ran `--write-seeds` again | Nothing to do. The file is meant to be written once |
| `ValueError: seed lists do not match the stored checksum` | `configs/seeds.yaml` was edited or merged badly | `git checkout configs/seeds.yaml`. Never edit it by hand |
| `refusing to fly test seeds with unfrozen scenario settings` | A config has `split: test` | Correct. Test seeds are for the final run only |
| `--set expects key=value` | Shell split the value | Quote it: `--set 'seeds={file: configs/seeds.yaml, n: 8}'` |

### HPC, quantized models (the least tested part)

| You see | Cause | Fix |
|---|---|---|
| Row says `failed: ImportError` (bitsandbytes) | Extra not installed in that env | `pip install -e ".[quant]"`. The other rows in the file are still good |
| `failed: RuntimeError ... needs a CUDA GPU` | Ran on a CPU node | Use the A100 partition |
| An error about `device_map` or `accelerate` | `accelerate` missing or old | `pip install -U accelerate` |
| `failed: ... OutOfMemoryError` in `eval_bank` | All variants loaded together | Add `ONE_AT_A_TIME=1` |
| bitsandbytes complains about CUDA libraries | Its build doesn't match the node's CUDA | `pip install -U bitsandbytes`, and paste the error in `AGENT_LOG.md` |
| A quantized model answers "no" to everything | Its scores are shifted, so cut 0 is wrong for it | Expected. The benchmark tunes a cut per model. Look at AUROC, not recall at 0 |
| The quantized run is very slow | bitsandbytes is often slower than bfloat16 | Stop after job 3 and decide. Options: larger `STRIDE`, only one condition, or drop 8-bit |
| Job ends at the time limit | Run was longer than `--time` | Resubmit with the same `OUT`. It continues where it stopped |
| Weights download fails | Not cached, or `HF_HUB_OFFLINE=1` | The quantized variants reuse the 8B weights already under `HF_HOME`; check `HF_HOME` points to scratch |

If the quantized models can't be made to load at all, that is a finding, not a blocker: the table still compares 2B and 8B. Write it in `AGENT_LOG.md`.

### HPC, benchmark table

| You see | Cause | Fix |
|---|---|---|
| `the frame-bank table needs both --vlm and --run` | One of the two is missing | Give both. `--run` is the `perception.run_bank` file (`da2_small.jsonl`) |
| Can't read `da2_small.jsonl` | It is in a teammate's scratch | Ask for read access, or rerun `perception.run_bank` (about 10 min on an A100) |
| "only N of M evaluated frames have a row for every system" | One model was run on fewer frames (e.g. `PER_TYPE=50`) | Fine for a first look. For the real table run every model on the same frames |
| Every cell is `n/a` | No train frames in the rows | Use `--score-split val`, and note the report then says "optimistic" |
| Memory column is `n/a` | No `--latency` file for that model | Run job 1 and pass its file |
| `refusing to benchmark on test frames` | Test rows in the input | Correct. Remove them |

### HPC, Project AirSim

| You see | Cause | Fix |
|---|---|---|
| `start pose 2 ... has 30 m clear; a 40 m flight needs 45 m` | The flight is longer than that pose allows | Shorten `goal_m`, or drop the pose |
| Crash with `collision_source: "scenery"` | The map blocks that pose | Drop the pose (section 5) |
| `backend_kw.starts differs from the frozen start poses` | Config and frozen scenario file disagree | Remove `starts` from the config; the frozen file wins |
| The simulator can't find its configs | `PP_SIM_CONFIGS` not set | Run through `hpc/airsim_job.sbatch`, which sets it |
| Node shut down mid-job | 20-minute idle rule | Sweeps resume: resubmit the same command |
| True-depth flights crash on spawned obstacles | Something is wrong in the loop on AirSim | Stop. This is the M4 gate. Don't run any other sweep until it is near zero |

---

## 7. What comes next

In order. Each step unblocks the one after.

1. **Team approves D21 to D24** and closes M2 with three tables (D23, revised 2026-11-18). Order:
   1. Get `da2_small.jsonl` (ask the owner of job 32117, or `sbatch hpc/run_bank.sbatch`, about 10 min).
   2. **Write the pass rule and the pick rule down first** (already in D23: recall >= 0.60 at FPR <= 0.15 on depth's misses, cut tuned on val, smallest memory among the passing variants).
   3. Job 3 (quantized sample) so table 1 has accuracy for every variant.
   4. Table 1: all variants, every obstacle type, independent of depth. Table 2: recall on depth's misses plus FPR on all negatives, on val, to pick the judge. Table 3: the picked judge on another split (`--score-split train`), depth only, VLM only, depth + VLM. Add the learnings to `results/benchmark.md`.
   5. Report the all-frames table next to table 2, and the proxy caveat: the end result is crash rate against calls per flight.
2. **Start poses checked** (job 5), then the **true-depth sweep on Project AirSim** (`airsim_true_depth.yaml`, 100 val seeds). This is the M4 gate: near-zero crashes, and the same seed gives the same flight.
3. **M3 on the real bank:** clean labels, then `gate.train`. This gives the first learned gate and the "fallback paper" table. CPU only.
4. **M4.7 headroom:** fly never-ask with Depth Anything on Project AirSim. Target: 15 to 30% crashes. If not, retune the scenario settings on val.
5. **Freeze the scenario settings** (`python -m sim.scenario --freeze-scenario ...`). After this, nothing about the flights changes.
6. **M5 first curve:** every policy (never, always, random, threshold, gate, oracle) on val seeds, crash rate against calls per flight, about 30 crashes with a cause each. Tag `mvp-v1`.
7. **M6 research results:** replay labels and the delay sweep. Then the **final test runs by Mon Nov 30**, test seeds used once.
8. **M7:** report, demo, slides.

Given 13 days, steps 1 to 6 need to be done within about a week. If time runs short, `PLAN_DETAILED.md` §13 gives the cut order.

Still unverified and worth remembering: bitsandbytes with Qwen3-VL, the five start poses, whether the new prompts help, and the two unlogged results named in section 5.
