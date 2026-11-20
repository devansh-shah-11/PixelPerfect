L1 labels at d_safe 8 m. Train 8279 frames (1169 positive), val 1830 frames (281 positive, 19 flights). Glass labelled from metadata: 668 rows. Held out of training: pole_10cm_axis_z.

| Model / features | val AUROC [95% CI over flights] | AUPRC (base rate) | ECE | recall / FPR at train 10%-FPR cut |
|---|---|---|---|---|
| baseline_var | 0.425 [0.312 to 0.544] | 0.123 (0.154) | n/a | 0.018 / 0.101 |
| best_single | 0.574 [0.438 to 0.694] | 0.205 (0.154) | n/a | 0.110 / 0.079 |
| logreg/depth | 0.615 [0.514 to 0.711] | 0.207 (0.154) | 0.328 | 0.253 / 0.170 |
| gbt/depth | 0.603 [0.498 to 0.711] | 0.247 (0.154) | 0.113 | 0.349 / 0.250 |
| mlp/depth | 0.619 [0.504 to 0.732] | 0.221 (0.154) | 0.056 | 0.256 / 0.132 |
| logreg/depth+temporal | 0.610 [0.523 to 0.693] | 0.210 (0.154) | 0.323 | 0.231 / 0.161 |
| gbt/depth+temporal | 0.628 [0.502 to 0.739] | 0.282 (0.154) | 0.095 | 0.423 / 0.250 |
| mlp/depth+temporal | 0.640 [0.527 to 0.746] | 0.269 (0.154) | 0.058 | 0.302 / 0.139 |
| logreg/depth+temporal+image | 0.677 [0.570 to 0.789] | 0.256 (0.154) | 0.305 | 0.335 / 0.163 |
| gbt/depth+temporal+image | 0.779 [0.665 to 0.881] | 0.524 (0.154) | 0.052 | 0.626 / 0.234 |
| mlp/depth+temporal+image | 0.785 [0.695 to 0.877] | 0.497 (0.154) | 0.036 | 0.534 / 0.138 |

Best learned gate: **mlp/depth+temporal+image**, val AUROC 0.785 against the depth-variance baseline 0.425 (M3 gate: beats the baseline).

### mlp/depth+temporal+image: by obstacle family (positives vs all val negatives)

| Family | positives (flights) | AUROC | recall at cut |
|---|---|---|---|
| ball_20cm_red | 21 (2) | 0.647 | 0.143 |
| box_1m_black | 0 (0) | n/a | n/a |
| box_1m_grey | 23 (2) | 0.983 | 1.000 |
| building_block | 0 (0) | n/a | n/a |
| glass_pane_alpha20 | 36 (2) | 0.997 | 1.000 |
| none | 2 (1) | 0.132 | 0.000 |
| pole_10cm_axis_z | 69 (5) | 0.805 | 0.580 |
| wall_white_featureless | 0 (0) | n/a | n/a |
| wire_1cm_axis_z | 55 (3) | 0.783 | 0.436 |
| wire_3cm_axis_z | 75 (4) | 0.663 | 0.320 |
