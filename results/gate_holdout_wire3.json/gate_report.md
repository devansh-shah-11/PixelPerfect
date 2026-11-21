L1 labels at d_safe 8 m. Train 8167 frames (1121 positive), val 1830 frames (281 positive, 19 flights). Glass labelled from metadata: 668 rows. Held out of training: wire_3cm_axis_z.

| Model / features | val AUROC [95% CI over flights] | AUPRC (base rate) | ECE | recall / FPR at train 10%-FPR cut |
|---|---|---|---|---|
| baseline_var | 0.425 [0.312 to 0.544] | 0.123 (0.154) | n/a | 0.014 / 0.094 |
| best_single | 0.574 [0.438 to 0.694] | 0.205 (0.154) | n/a | 0.110 / 0.081 |
| logreg/depth | 0.602 [0.498 to 0.706] | 0.207 (0.154) | 0.301 | 0.256 / 0.158 |
| gbt/depth | 0.617 [0.530 to 0.711] | 0.256 (0.154) | 0.108 | 0.377 / 0.224 |
| mlp/depth | 0.624 [0.511 to 0.727] | 0.263 (0.154) | 0.090 | 0.306 / 0.156 |
| logreg/depth+temporal | 0.606 [0.510 to 0.696] | 0.210 (0.154) | 0.296 | 0.231 / 0.156 |
| gbt/depth+temporal | 0.650 [0.567 to 0.739] | 0.295 (0.154) | 0.091 | 0.431 / 0.224 |
| mlp/depth+temporal | 0.611 [0.515 to 0.707] | 0.229 (0.154) | 0.062 | 0.228 / 0.138 |
| logreg/depth+temporal+image | 0.705 [0.595 to 0.819] | 0.296 (0.154) | 0.261 | 0.395 / 0.146 |
| gbt/depth+temporal+image | 0.791 [0.710 to 0.876] | 0.524 (0.154) | 0.070 | 0.580 / 0.196 |
| mlp/depth+temporal+image | 0.811 [0.726 to 0.892] | 0.552 (0.154) | 0.020 | 0.541 / 0.104 |

Best learned gate: **mlp/depth+temporal+image**, val AUROC 0.811 against the depth-variance baseline 0.425 (M3 gate: beats the baseline).

### mlp/depth+temporal+image: by obstacle family (positives vs all val negatives)

| Family | positives (flights) | AUROC | recall at cut |
|---|---|---|---|
| ball_20cm_red | 21 (2) | 0.637 | 0.095 |
| box_1m_black | 0 (0) | n/a | n/a |
| box_1m_grey | 23 (2) | 0.962 | 0.826 |
| building_block | 0 (0) | n/a | n/a |
| glass_pane_alpha20 | 36 (2) | 0.997 | 1.000 |
| none | 2 (1) | 0.186 | 0.000 |
| pole_10cm_axis_z | 69 (5) | 0.848 | 0.652 |
| wall_white_featureless | 0 (0) | n/a | n/a |
| wire_1cm_axis_z | 55 (3) | 0.808 | 0.418 |
| wire_3cm_axis_z | 75 (4) | 0.712 | 0.360 |
