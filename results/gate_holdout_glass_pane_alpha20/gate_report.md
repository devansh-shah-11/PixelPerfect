L1 labels at d_safe 8 m. Train 8415 frames (1370 positive), val 1830 frames (281 positive, 19 flights). Glass labelled from metadata: 668 rows. Held out of training: glass_pane_alpha20.

| Model / features | val AUROC [95% CI over flights] | AUPRC (base rate) | ECE | recall / FPR at train 10%-FPR cut |
|---|---|---|---|---|
| baseline_var | 0.425 [0.312 to 0.544] | 0.123 (0.154) | n/a | 0.014 / 0.088 |
| best_single | 0.695 [0.559 to 0.801] | 0.356 (0.154) | n/a | 0.306 / 0.086 |
| logreg/depth | 0.614 [0.516 to 0.714] | 0.211 (0.154) | 0.313 | 0.270 / 0.174 |
| gbt/depth | 0.614 [0.525 to 0.712] | 0.256 (0.154) | 0.102 | 0.395 / 0.233 |
| mlp/depth | 0.623 [0.517 to 0.725] | 0.264 (0.154) | 0.075 | 0.295 / 0.154 |
| logreg/depth+temporal | 0.612 [0.523 to 0.699] | 0.212 (0.154) | 0.306 | 0.256 / 0.167 |
| gbt/depth+temporal | 0.634 [0.542 to 0.729] | 0.282 (0.154) | 0.090 | 0.427 / 0.231 |
| mlp/depth+temporal | 0.653 [0.549 to 0.750] | 0.279 (0.154) | 0.061 | 0.324 / 0.147 |
| logreg/depth+temporal+image | 0.624 [0.485 to 0.748] | 0.232 (0.154) | 0.306 | 0.281 / 0.164 |
| gbt/depth+temporal+image | 0.714 [0.598 to 0.810] | 0.403 (0.154) | 0.080 | 0.520 / 0.229 |
| mlp/depth+temporal+image | 0.706 [0.557 to 0.825] | 0.431 (0.154) | 0.032 | 0.441 / 0.110 |

Best learned gate: **gbt/depth+temporal+image**, val AUROC 0.714 against the depth-variance baseline 0.425 (M3 gate: beats the baseline).

### gbt/depth+temporal+image: by obstacle family (positives vs all val negatives)

| Family | positives (flights) | AUROC | recall at cut |
|---|---|---|---|
| ball_20cm_red | 21 (2) | 0.699 | 0.333 |
| box_1m_black | 0 (0) | n/a | n/a |
| box_1m_grey | 23 (2) | 0.972 | 1.000 |
| building_block | 0 (0) | n/a | n/a |
| glass_pane_alpha20 | 36 (2) | 0.215 | 0.000 |
| none | 2 (1) | 0.786 | 0.500 |
| pole_10cm_axis_z | 69 (5) | 0.842 | 0.725 |
| wall_white_featureless | 0 (0) | n/a | n/a |
| wire_1cm_axis_z | 55 (3) | 0.768 | 0.564 |
| wire_3cm_axis_z | 75 (4) | 0.717 | 0.453 |
