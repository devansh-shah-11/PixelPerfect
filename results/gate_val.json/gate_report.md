L1 labels at d_safe 8 m. Train 9023 frames (1489 positive), val 1830 frames (281 positive, 19 flights). Glass labelled from metadata: 668 rows.

| Model / features | val AUROC [95% CI over flights] | AUPRC (base rate) | ECE | recall / FPR at train 10%-FPR cut |
|---|---|---|---|---|
| baseline_var | 0.425 [0.312 to 0.544] | 0.123 (0.154) | n/a | 0.014 / 0.093 |
| best_single | 0.695 [0.559 to 0.801] | 0.356 (0.154) | n/a | 0.310 / 0.088 |
| logreg/depth | 0.612 [0.510 to 0.712] | 0.208 (0.154) | 0.315 | 0.256 / 0.168 |
| gbt/depth | 0.616 [0.526 to 0.715] | 0.250 (0.154) | 0.105 | 0.388 / 0.230 |
| mlp/depth | 0.623 [0.511 to 0.726] | 0.258 (0.154) | 0.070 | 0.278 / 0.130 |
| logreg/depth+temporal | 0.611 [0.519 to 0.701] | 0.211 (0.154) | 0.309 | 0.246 / 0.159 |
| gbt/depth+temporal | 0.640 [0.548 to 0.737] | 0.291 (0.154) | 0.086 | 0.423 / 0.224 |
| mlp/depth+temporal | 0.635 [0.530 to 0.735] | 0.249 (0.154) | 0.037 | 0.263 / 0.138 |
| logreg/depth+temporal+image | 0.695 [0.593 to 0.806] | 0.275 (0.154) | 0.285 | 0.356 / 0.164 |
| gbt/depth+temporal+image | 0.808 [0.739 to 0.880] | 0.520 (0.154) | 0.058 | 0.630 / 0.214 |
| mlp/depth+temporal+image | 0.829 [0.753 to 0.906] | 0.566 (0.154) | 0.008 | 0.552 / 0.109 |

Best learned gate: **mlp/depth+temporal+image**, val AUROC 0.829 against the depth-variance baseline 0.425 (M3 gate: beats the baseline).

### mlp/depth+temporal+image: by obstacle family (positives vs all val negatives)

| Family | positives (flights) | AUROC | recall at cut |
|---|---|---|---|
| ball_20cm_red | 21 (2) | 0.697 | 0.238 |
| box_1m_black | 0 (0) | n/a | n/a |
| box_1m_grey | 23 (2) | 0.972 | 0.870 |
| building_block | 0 (0) | n/a | n/a |
| glass_pane_alpha20 | 36 (2) | 0.993 | 1.000 |
| none | 2 (1) | 0.188 | 0.000 |
| pole_10cm_axis_z | 69 (5) | 0.858 | 0.638 |
| wall_white_featureless | 0 (0) | n/a | n/a |
| wire_1cm_axis_z | 55 (3) | 0.844 | 0.436 |
| wire_3cm_axis_z | 75 (4) | 0.723 | 0.347 |
