Scene read: frame and object level (truth = true depth of every surface, placed or scenery)

| condition | pos n | neg n | recall | false alarm | all ones | parse fail | thin found | big found | scenery found | depth-missed found | box precision | IoU | range within 1 | latency s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen/Qwen3-VL-8B-Instruct|json_raw | 259 | 209 | 0.05 | 0.01 | 0.00 | 0.00 | 0.00 | 0.07 | 0.00 | 0.02 | 0.80 | 0.86 | 1.00 | 0.14 |
| Qwen/Qwen3-VL-8B-Instruct|grid_raw | 259 | 209 | 0.96 | 1.00 | 0.83 | 0.00 | 1.00 | 0.92 | 0.93 | 0.99 | 0.28 | 0.01 | 1.00 | 0.52 |
| Qwen/Qwen3-VL-8B-Instruct|yesno_both | 259 | 209 | 0.51 | 0.29 | 0.00 | 0.00 | 0.25 | 0.12 | 0.04 | 0.18 | 0.56 | 0.02 | 1.00 | 0.09 |
| oracle|- | 259 | 209 | 1.00 | 0.00 | 0.03 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | n/a |
| all_near|- | 259 | 209 | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.55 | 0.00 | 1.00 | n/a |

Steering arm (boxes turned into a move; same answer key as the direction pilot)

| model | condition | avoid n | safe [95% CI] | exact | regret m | straight n | false steer |
|---|---|---:|---|---:|---:|---:|---:|
| depth_gt_choose | - | 129 | 0.50 [0.36, 0.61] | 0.18 | 1.22 | 170 | 0.07 |
| goal_side | - | 129 | 0.87 [0.67, 1.00] | 0.66 | 1.37 | 170 | 0.00 |
| oracle | - | 129 | 1.00 [1.00, 1.00] | 1.00 | 0.00 | 170 | 0.00 |
| random | - | 129 | 0.66 [0.55, 0.75] | 0.22 | 1.29 | 170 | 0.80 |
| scene_grid_raw | boxes | 129 | 0.81 [0.61, 1.00] | 0.35 | 2.52 | 170 | 1.00 |
| scene_json_raw | boxes | 129 | 0.01 [0.00, 0.02] | 0.00 | -0.60 | 170 | 0.08 |
