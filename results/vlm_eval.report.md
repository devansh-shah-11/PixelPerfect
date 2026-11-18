Positive: nearest true surface in the window closer than 15 m. Negative: farther than 30 m. In between: ambiguous, excluded. Intervals: 95%, resampling whole flights.

### All frames, cut at confidence 0

| Model / condition | n_pos (flights) | n_neg | recall [95% CI] | precision | FPR | AUROC |
|---|---|---|---|---|---|---|
| Qwen/Qwen3-VL-2B-Instruct / both | 1457 (168) | 1353 | 0.37 [0.31 to 0.43] | 0.88 | 0.06 | 0.67 |
| Qwen/Qwen3-VL-2B-Instruct / both_blocked | 1457 (168) | 1353 | 0.55 [0.50 to 0.61] | 0.77 | 0.18 | 0.75 |
| Qwen/Qwen3-VL-2B-Instruct / both_fill | 1457 (168) | 1353 | 0.92 [0.89 to 0.95] | 0.54 | 0.83 | 0.72 |
| Qwen/Qwen3-VL-2B-Instruct / both_free | 1457 (168) | 1353 | 0.02 [0.01 to 0.04] | 0.83 | 0.01 | 0.56 |
| Qwen/Qwen3-VL-2B-Instruct / both_hit | 1457 (168) | 1353 | 0.69 [0.64 to 0.75] | 0.53 | 0.66 | 0.57 |
| Qwen/Qwen3-VL-2B-Instruct / crop | 1457 (168) | 1353 | 0.41 [0.35 to 0.46] | 0.79 | 0.12 | 0.65 |
| Qwen/Qwen3-VL-2B-Instruct / overlay | 1457 (168) | 1353 | 0.09 [0.07 to 0.11] | 0.74 | 0.03 | 0.55 |
| Qwen/Qwen3-VL-2B-Instruct / raw | 1457 (168) | 1353 | 0.13 [0.10 to 0.17] | 0.71 | 0.06 | 0.53 |
| Qwen/Qwen3-VL-8B-Instruct / both | 1457 (168) | 1353 | 0.14 [0.10 to 0.18] | 1.00 | 0.00 | 0.79 |
| Qwen/Qwen3-VL-8B-Instruct / both_blocked | 1457 (168) | 1353 | 0.28 [0.23 to 0.33] | 0.87 | 0.04 | 0.73 |
| Qwen/Qwen3-VL-8B-Instruct / both_fill | 1457 (168) | 1353 | 0.30 [0.25 to 0.35] | 0.89 | 0.04 | 0.72 |
| Qwen/Qwen3-VL-8B-Instruct / both_free | 1457 (168) | 1353 | 0.00 [0.00 to 0.00] | 0.67 | 0.00 | 0.57 |
| Qwen/Qwen3-VL-8B-Instruct / both_hit | 1457 (168) | 1353 | 0.12 [0.09 to 0.15] | 0.63 | 0.07 | 0.49 |
| Qwen/Qwen3-VL-8B-Instruct / crop | 1457 (168) | 1353 | 0.34 [0.28 to 0.40] | 0.96 | 0.01 | 0.72 |
| Qwen/Qwen3-VL-8B-Instruct / overlay | 1457 (168) | 1353 | 0.01 [0.01 to 0.02] | 0.94 | 0.00 | 0.58 |
| Qwen/Qwen3-VL-8B-Instruct / raw | 1457 (168) | 1353 | 0.09 [0.07 to 0.12] | 0.71 | 0.04 | 0.47 |

### M2 gate: recall on visible types (not wires, not empty) >= 0.60

| Model / condition | visible n_pos (flights) | recall at 0 [95% CI] | cut tuned on val | train recall / FPR at that cut | verdict |
|---|---|---|---|---|---|
| Qwen/Qwen3-VL-2B-Instruct / both | 999 (122) | 0.30 [0.24 to 0.36] | -0.12 | 0.31 / 0.12 | FAIL (recall 0.30 at cut 0) |
| Qwen/Qwen3-VL-2B-Instruct / both_blocked | 999 (122) | 0.59 [0.53 to 0.66] | 0.50 | 0.44 / 0.12 | FAIL (recall 0.59 at cut 0) |
| Qwen/Qwen3-VL-2B-Instruct / both_fill | 999 (122) | 0.90 [0.85 to 0.94] | 1.63 | 0.35 / 0.10 | FAIL: recall 0.90 but FPR 0.83 at cut 0 (says yes too often) |
| Qwen/Qwen3-VL-2B-Instruct / both_free | 999 (122) | 0.02 [0.01 to 0.04] | -1.13 | 0.14 / 0.12 | FAIL (recall 0.02 at cut 0) |
| Qwen/Qwen3-VL-2B-Instruct / both_hit | 999 (122) | 0.61 [0.54 to 0.68] | 0.38 | 0.21 / 0.08 | FAIL: recall 0.61 but FPR 0.66 at cut 0 (says yes too often) |
| Qwen/Qwen3-VL-2B-Instruct / crop | 999 (122) | 0.37 [0.31 to 0.43] | 0.00 | 0.32 / 0.11 | FAIL (recall 0.37 at cut 0) |
| Qwen/Qwen3-VL-2B-Instruct / overlay | 999 (122) | 0.11 [0.08 to 0.14] | -0.25 | 0.18 / 0.11 | FAIL (recall 0.11 at cut 0) |
| Qwen/Qwen3-VL-2B-Instruct / raw | 999 (122) | 0.18 [0.14 to 0.23] | -0.14 | 0.23 / 0.13 | FAIL (recall 0.18 at cut 0) |
| Qwen/Qwen3-VL-8B-Instruct / both | 999 (122) | 0.10 [0.06 to 0.14] | -4.02 | 0.58 / 0.21 | FAIL (recall 0.10 at cut 0) |
| Qwen/Qwen3-VL-8B-Instruct / both_blocked | 999 (122) | 0.35 [0.28 to 0.41] | -1.68 | 0.48 / 0.21 | FAIL (recall 0.35 at cut 0) |
| Qwen/Qwen3-VL-8B-Instruct / both_fill | 999 (122) | 0.36 [0.30 to 0.43] | -1.00 | 0.47 / 0.19 | FAIL (recall 0.36 at cut 0) |
| Qwen/Qwen3-VL-8B-Instruct / both_free | 999 (122) | 0.00 [0.00 to 0.01] | -6.24 | 0.15 / 0.11 | FAIL (recall 0.00 at cut 0) |
| Qwen/Qwen3-VL-8B-Instruct / both_hit | 999 (122) | 0.14 [0.09 to 0.18] | -0.19 | 0.19 / 0.14 | FAIL (recall 0.14 at cut 0) |
| Qwen/Qwen3-VL-8B-Instruct / crop | 999 (122) | 0.27 [0.20 to 0.34] | -1.49 | 0.47 / 0.20 | FAIL (recall 0.27 at cut 0) |
| Qwen/Qwen3-VL-8B-Instruct / overlay | 999 (122) | 0.02 [0.01 to 0.03] | -4.50 | 0.27 / 0.12 | FAIL (recall 0.02 at cut 0) |
| Qwen/Qwen3-VL-8B-Instruct / raw | 999 (122) | 0.13 [0.10 to 0.17] | -1.49 | 0.20 / 0.17 | FAIL (recall 0.13 at cut 0) |

### Qwen/Qwen3-VL-2B-Instruct / both_fill: by obstacle type

| Obstacle | n_pos (flights) | recall [95% CI] | n_neg | FPR | AUROC vs all negatives |
|---|---|---|---|---|---|
| ball_20cm_red | 110 (17) | 0.99 [0.97 to 1.00] | 102 | 0.78 | 0.71 |
| box_1m_black | 137 (19) | 0.99 [0.97 to 1.00] | 165 | 0.85 | 0.92 |
| box_1m_grey | 145 (19) | 0.99 [0.98 to 1.00] | 112 | 0.83 | 0.84 |
| building_block | 131 (17) | 0.32 [0.19 to 0.44] | 61 | 0.85 | 0.21 |
| glass_pane_alpha20 | 106 (11) | 0.98 [0.95 to 1.00] | 79 | 0.87 | 0.75 |
| none | 1 (1) | 1.00 [n/a] | 780 | 0.83 | 0.78 |
| pole_10cm_axis_z | 233 (26) | 0.99 [0.97 to 1.00] | 15 | 0.93 | 0.73 |
| wall_white_featureless | 137 (13) | 0.98 [0.95 to 1.00] | 23 | 0.65 | 0.92 |
| wire_1cm_axis_z | 187 (18) | 0.96 [0.93 to 0.99] | 4 | 1.00 | 0.65 |
| wire_3cm_axis_z | 270 (27) | 0.97 [0.96 to 0.99] | 12 | 0.75 | 0.74 |

### Qwen/Qwen3-VL-2B-Instruct / both_fill: by distance of the nearest true surface

| Distance | n_pos (flights) | recall | n_neg | FPR |
|---|---|---|---|---|
| 0-3 m | 227 (115) | 0.80 | 0 | n/a |
| 3-6 m | 326 (132) | 0.90 | 0 | n/a |
| 6-10 m | 438 (147) | 0.94 | 0 | n/a |
| 10-20 m | 466 (136) | 0.98 | 0 | n/a |
| >20 m | 0 (0) | n/a | 1353 | 0.83 |

### Qwen/Qwen3-VL-2B-Instruct / both: by obstacle type

| Obstacle | n_pos (flights) | recall [95% CI] | n_neg | FPR | AUROC vs all negatives |
|---|---|---|---|---|---|
| ball_20cm_red | 110 (17) | 0.81 [0.72 to 0.89] | 102 | 0.10 | 0.94 |
| box_1m_black | 137 (19) | 0.15 [0.06 to 0.27] | 165 | 0.10 | 0.61 |
| box_1m_grey | 145 (19) | 0.26 [0.15 to 0.39] | 112 | 0.14 | 0.67 |
| building_block | 131 (17) | 0.00 [0.00 to 0.00] | 61 | 0.05 | 0.02 |
| glass_pane_alpha20 | 106 (11) | 0.15 [0.09 to 0.23] | 79 | 0.09 | 0.69 |
| none | 1 (1) | 1.00 [n/a] | 780 | 0.02 | 0.99 |
| pole_10cm_axis_z | 233 (26) | 0.53 [0.40 to 0.65] | 15 | 0.20 | 0.83 |
| wall_white_featureless | 137 (13) | 0.08 [0.03 to 0.14] | 23 | 0.09 | 0.44 |
| wire_1cm_axis_z | 187 (18) | 0.57 [0.43 to 0.72] | 4 | 0.00 | 0.82 |
| wire_3cm_axis_z | 270 (27) | 0.49 [0.36 to 0.61] | 12 | 0.00 | 0.79 |

### Qwen/Qwen3-VL-2B-Instruct / both: by distance of the nearest true surface

| Distance | n_pos (flights) | recall | n_neg | FPR |
|---|---|---|---|---|
| 0-3 m | 227 (115) | 0.31 | 0 | n/a |
| 3-6 m | 326 (132) | 0.29 | 0 | n/a |
| 6-10 m | 438 (147) | 0.41 | 0 | n/a |
| 10-20 m | 466 (136) | 0.41 | 0 | n/a |
| >20 m | 0 (0) | n/a | 1353 | 0.06 |

### Qwen/Qwen3-VL-8B-Instruct / both: by obstacle type

| Obstacle | n_pos (flights) | recall [95% CI] | n_neg | FPR | AUROC vs all negatives |
|---|---|---|---|---|---|
| ball_20cm_red | 110 (17) | 0.00 [0.00 to 0.00] | 102 | 0.00 | 0.86 |
| box_1m_black | 137 (19) | 0.00 [0.00 to 0.00] | 165 | 0.00 | 0.72 |
| box_1m_grey | 145 (19) | 0.00 [0.00 to 0.00] | 112 | 0.00 | 0.81 |
| building_block | 131 (17) | 0.00 [0.00 to 0.00] | 61 | 0.00 | 0.22 |
| glass_pane_alpha20 | 106 (11) | 0.02 [0.00 to 0.05] | 79 | 0.00 | 0.87 |
| none | 1 (1) | 0.00 [n/a] | 780 | 0.00 | 0.94 |
| pole_10cm_axis_z | 233 (26) | 0.40 [0.26 to 0.53] | 15 | 0.00 | 0.90 |
| wall_white_featureless | 137 (13) | 0.01 [0.00 to 0.02] | 23 | 0.00 | 0.77 |
| wire_1cm_axis_z | 187 (18) | 0.29 [0.16 to 0.44] | 4 | 0.00 | 0.90 |
| wire_3cm_axis_z | 270 (27) | 0.19 [0.10 to 0.30] | 12 | 0.00 | 0.87 |

### Qwen/Qwen3-VL-8B-Instruct / both: by distance of the nearest true surface

| Distance | n_pos (flights) | recall | n_neg | FPR |
|---|---|---|---|---|
| 0-3 m | 227 (115) | 0.19 | 0 | n/a |
| 3-6 m | 326 (132) | 0.15 | 0 | n/a |
| 6-10 m | 438 (147) | 0.14 | 0 | n/a |
| 10-20 m | 466 (136) | 0.11 | 0 | n/a |
| >20 m | 0 (0) | n/a | 1353 | 0.00 |

### Ground-truth check: metadata vs segmentation, 'obstacle in window and closer than 15 m'

| Obstacle | both | metadata only | segmentation only | neither | no metadata |
|---|---|---|---|---|---|
| ball_20cm_red | 103 | 2 | 7 | 211 | 0 |
| box_1m_black | 130 | 2 | 7 | 216 | 0 |
| box_1m_grey | 137 | 0 | 8 | 210 | 0 |
| building_block | 102 | 78 | 29 | 8 | 0 |
| glass_pane_alpha20 | 0 | 106 | 0 | 121 | 0 |
| pole_10cm_axis_z | 226 | 2 | 7 | 74 | 0 |
| wall_white_featureless | 134 | 0 | 3 | 81 | 0 |
| wire_1cm_axis_z | 184 | 0 | 3 | 43 | 0 |
| wire_3cm_axis_z | 266 | 2 | 4 | 85 | 0 |

Glass uses the metadata label (its segmentation is see-through). A large 'metadata only' count on another type means its segmentation is see-through too, or its half-width in `HALF_WIDTH_M` is wrong.
