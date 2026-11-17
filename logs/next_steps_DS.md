# Next steps for DS (written Tue Nov 17)

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
git commit -m "sim: freeze seed lists, five start poses, scenario freeze, sweep overrides" \
           -m "D24: seeds frozen at 1000/500/500. One start pose left no room for a 2 s delay, so flights are 40 m from five poses picked by seed. Test seeds now need frozen scenario settings. Start poses are unverified until airsim_starts_smoke runs."
```
Check that `configs/seeds.yaml` is in this commit. It is a new file, so `git add -u` would miss it.

**Commit 3: benchmark table**
```bash
git add eval/benchmark.py tests/test_benchmark.py
git commit -m "eval: benchmark table for depth only, VLM only, depth + VLM" \
           -m "One table per VLM variant with accuracy, latency and memory, and a pick by the D23 rule, so the team can choose the judge (D22). Reads result files only."
```

**Commit 4: docs**
```bash
git add README.md TASKS.md CLAUDE.md PLAN_DETAILED.md AGENT_LOG.md logs/next_steps_DS.md
git commit -m "docs: benchmark and seed-freeze instructions, D24 counts, agent log" \
           -m "Records what was built, what is unverified, and the HPC jobs still to run."
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
| 4 | Quantized 8B, full | the last command in that header | **unknown, maybe over 1 h** | rows for `@8bit` and `@4bit` |
| 5 | Start-pose check | `sbatch --export=ALL,PA_ENV=/scratch/dns5508/envs/pa,CMD="python -m eval.sweep --config configs/sweeps/airsim_starts_smoke.yaml" hpc/airsim_job.sbatch` | about 15 min | which of the five poses are really open |

Then build the table (laptop or a CPU node):
```bash
python -m eval.benchmark --vlm results/vlm_eval.jsonl --run <bank runs>/da2_small.jsonl \
    --latency results/vlm_latency_*.json --out results/benchmark.md
```

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

1. **Team approves D21 to D24** and closes M2 with the benchmark table (jobs 1 to 4). Decide the judge: 8B, or a quantized 8B if it passes.
2. **Start poses checked** (job 5), then the **true-depth sweep on Project AirSim** (`airsim_true_depth.yaml`, 100 val seeds). This is the M4 gate: near-zero crashes, and the same seed gives the same flight.
3. **M3 on the real bank:** clean labels, then `gate.train`. This gives the first learned gate and the "fallback paper" table. CPU only.
4. **M4.7 headroom:** fly never-ask with Depth Anything on Project AirSim. Target: 15 to 30% crashes. If not, retune the scenario settings on val.
5. **Freeze the scenario settings** (`python -m sim.scenario --freeze-scenario ...`). After this, nothing about the flights changes.
6. **M5 first curve:** every policy (never, always, random, threshold, gate, oracle) on val seeds, crash rate against calls per flight, about 30 crashes with a cause each. Tag `mvp-v1`.
7. **M6 research results:** replay labels and the delay sweep. Then the **final test runs by Mon Nov 30**, test seeds used once.
8. **M7:** report, demo, slides.

Given 13 days, steps 1 to 6 need to be done within about a week. If time runs short, `PLAN_DETAILED.md` §13 gives the cut order.

Still unverified and worth remembering: bitsandbytes with Qwen3-VL, the five start poses, whether the new prompts help, and the two unlogged results named in section 5.
