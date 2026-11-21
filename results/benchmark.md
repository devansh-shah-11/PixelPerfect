# Benchmark: depth only, VLM only, depth + VLM

## Frame bank

Task: an obstacle within 8 m in the corridor window. Positive: nearest true surface closer than 8 m. Negative: nothing closer than 30 m. In between: left out.
Scored on 1773 train frames (657 positive from 121 flights, 1116 negative), the same frames for every system. The fast stack misses 489 of the positives. Intervals: 95%, resampling whole flights.
Each VLM's condition and cut were chosen on val (cut: lowest with val false alarms at most 0.10), never on the frames scored here.

| System | recall [95% CI] | false alarms | precision | AUROC | recall on depth's misses | latency median / p95 (ms) | peak GPU (GB) |
|---|---|---|---|---|---|---|---|
| depth only | 0.26 [0.18 to 0.33] | 0.00 | 1.00 | 0.73 |  | 15 / 15 | 0.2 |
| VLM only: Qwen/Qwen3-VL-2B-Instruct | 0.47 [0.40 to 0.54] | 0.09 | 0.76 | 0.76 | 0.48 | 76 / 81 | 4.4 |
| VLM only: Qwen/Qwen3-VL-8B-Instruct | 0.67 [0.60 to 0.73] | 0.14 | 0.73 | 0.79 | 0.80 | 91 / 98 | 17.7 |
| depth + VLM: Qwen/Qwen3-VL-2B-Instruct | 0.61 [0.55 to 0.67] | 0.09 | 0.81 | n/a |  | 91 / 97 | 4.5 |
| depth + VLM: Qwen/Qwen3-VL-8B-Instruct | 0.85 [0.81 to 0.89] | 0.14 | 0.78 | n/a |  | 106 / 113 | 17.8 |

### VLM settings (chosen on val)

| Model | condition | val AUROC | cut on the confidence | passes the M2 gate (D23) |
|---|---|---|---|---|
| Qwen/Qwen3-VL-2B-Instruct | both_blocked (of both, both_blocked, both_fill, both_free, both_hit, crop, overlay, raw) | 0.69 | +0.50 | no |
| Qwen/Qwen3-VL-8B-Instruct | both (of both, both_blocked, both_fill, both_free, both_hit, crop, overlay, raw) | 0.83 | -4.02 | yes |

### Recall by obstacle type (positive frames)

| System | ball_20cm_red (n=31) | box_1m_black (n=67) | box_1m_grey (n=54) | building_block (n=83) | glass_pane_alpha20 (n=38) | pole_10cm_axis_z (n=108) | wall_white_featureless (n=69) | wire_1cm_axis_z (n=84) | wire_3cm_axis_z (n=123) |
|---|---|---|---|---|---|---|---|---|---|
| depth only | 0.00 | 0.13 | 0.13 | 1.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 |
| VLM only: Qwen/Qwen3-VL-2B-Instruct | 0.19 | 0.94 | 0.56 | 0.02 | 0.32 | 0.55 | 0.84 | 0.26 | 0.46 |
| VLM only: Qwen/Qwen3-VL-8B-Instruct | 0.84 | 0.57 | 0.63 | 0.11 | 0.87 | 0.85 | 0.43 | 0.87 | 0.84 |
| depth + VLM: Qwen/Qwen3-VL-2B-Instruct | 0.19 | 0.94 | 0.56 | 1.00 | 0.32 | 0.55 | 1.00 | 0.26 | 0.46 |
| depth + VLM: Qwen/Qwen3-VL-8B-Instruct | 0.84 | 0.63 | 0.69 | 1.00 | 0.87 | 0.85 | 1.00 | 0.87 | 0.84 |

**Pick, by rule (the team decides):** Qwen/Qwen3-VL-8B-Instruct (both, cut -4.02): the smallest-memory variant that passes the M2 gate (D23), recall on depth's misses 0.80, false alarms 0.14.
Rule: recall on depth's misses at least 0.60 with false alarms at most 0.15, then the least GPU memory.

Latency and memory sources: depth only: vlm_latency_35101.json; VLM only: Qwen/Qwen3-VL-2B-Instruct: vlm_latency_35101.json; VLM only: Qwen/Qwen3-VL-8B-Instruct: vlm_latency_35101.json.
