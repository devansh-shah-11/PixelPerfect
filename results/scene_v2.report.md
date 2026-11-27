Scene read: frame and object level (truth = true depth of every surface, placed or scenery). Yes/no cuts tuned on train (FPR <= 0.1): {"v2_both": 0.72, "v2_raw": 0.5, "v2near_both": -2.28, "v2thin_both": 1.62, "yesno_both": -3.25}

| condition | pos n | neg n | recall | false alarm | all ones | parse fail | in-path found | in-path thin | in-path depth-missed | box precision | IoU | range exact | latency s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen/Qwen3-VL-8B-Instruct|yesno_both | 259 | 209 | 0.31 | 0.12 | 0.00 | 0.00 | 0.21 | 0.25 | 0.23 | 0.70 | 0.02 | 0.74 | 0.09 |
| Qwen/Qwen3-VL-8B-Instruct|v2_both | 259 | 209 | 0.09 | 0.14 | 0.00 | 0.00 | 0.01 | 0.00 | 0.00 | 0.19 | 0.00 | 1.00 | 0.09 |
| Qwen/Qwen3-VL-8B-Instruct|v2near_both | 259 | 209 | 0.19 | 0.12 | 0.00 | 0.00 | 0.08 | 0.00 | 0.08 | 0.47 | 0.08 | 0.83 | 0.09 |
| Qwen/Qwen3-VL-8B-Instruct|v2thin_both | 259 | 209 | 0.19 | 0.12 | 0.00 | 0.00 | 0.11 | 0.13 | 0.11 | 0.58 | 0.02 | 0.71 | 0.09 |
| Qwen/Qwen3-VL-8B-Instruct|v2_raw | 259 | 209 | 0.09 | 0.11 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.19 | n/a | n/a | 0.08 |
| Qwen/Qwen3-VL-8B-Instruct|jsonv2_both | 259 | 209 | 0.02 | 0.02 | 0.00 | 0.00 | 0.01 | 0.00 | 0.00 | 0.27 | 0.18 | 1.00 | 0.14 |
| Qwen/Qwen3-VL-8B-Instruct|jsonv2_raw | 259 | 209 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | n/a | n/a | n/a | 0.13 |
| oracle|- | 259 | 209 | 1.00 | 0.00 | 0.03 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | n/a |
| all_near|- | 259 | 209 | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | 0.55 | 0.00 | 0.36 | n/a |

AUROC by obstacle type, yes/no questions (object within 15 m vs none)

- Qwen/Qwen3-VL-8B-Instruct|v2_both: ball_20cm_red 0.479, box_1m_black 0.542, box_1m_grey 0.437, building_block 0.043, pole_10cm_axis_z 0.375, wall_white_featureless 0.744, wire_1cm_axis_z 0.416, wire_3cm_axis_z 0.506
- Qwen/Qwen3-VL-8B-Instruct|v2_raw: ball_20cm_red 0.5, box_1m_black 0.53, box_1m_grey 0.526, building_block 0.117, pole_10cm_axis_z 0.387, wall_white_featureless 0.754, wire_1cm_axis_z 0.364, wire_3cm_axis_z 0.455
- Qwen/Qwen3-VL-8B-Instruct|v2near_both: ball_20cm_red 0.516, box_1m_black 0.59, box_1m_grey 0.532, building_block 0.539, pole_10cm_axis_z 0.38, wall_white_featureless 0.926, wire_1cm_axis_z 0.466, wire_3cm_axis_z 0.532
- Qwen/Qwen3-VL-8B-Instruct|v2thin_both: ball_20cm_red 0.456, box_1m_black 0.357, box_1m_grey 0.464, building_block 0.344, pole_10cm_axis_z 0.521, wall_white_featureless 0.662, wire_1cm_axis_z 0.754, wire_3cm_axis_z 0.642
- Qwen/Qwen3-VL-8B-Instruct|yesno_both: ball_20cm_red 0.603, box_1m_black 0.437, box_1m_grey 0.488, building_block 0.314, pole_10cm_axis_z 0.635, wall_white_featureless 0.62, wire_1cm_axis_z 0.929, wire_3cm_axis_z 0.787

Steering arm (boxes turned into a move; same answer key as the direction pilot)

| model | condition | avoid n | safe [95% CI] | exact | regret m | straight n | false steer |
|---|---|---:|---|---:|---:|---:|---:|
| depth_gt_choose | - | 129 | 0.50 [0.36, 0.61] | 0.18 | 1.22 | 170 | 0.07 |
| goal_side | - | 129 | 0.87 [0.67, 1.00] | 0.66 | 1.37 | 170 | 0.00 |
| oracle | - | 129 | 1.00 [1.00, 1.00] | 1.00 | 0.00 | 170 | 0.00 |
| random | - | 129 | 0.66 [0.55, 0.75] | 0.22 | 1.29 | 170 | 0.80 |
| scene_jsonv2_both | boxes | 129 | 0.02 [0.00, 0.04] | 0.01 | 1.74 | 170 | 0.03 |
| scene_jsonv2_raw | boxes | 129 | 0.00 [0.00, 0.00] | 0.00 | nan | 170 | 0.00 |
