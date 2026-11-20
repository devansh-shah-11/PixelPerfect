## Part A: corridor-minimum relative error, train frames, GT <= 40 m

| Frames | n | mean | median | trimmed mean (drop worst 5%) | share of total error from worst 5% | median est/gt |
|---|---|---|---|---|---|---|
| all | 8151 | 2.566 | 0.325 | 1.484 | 0.450 | 1.266 |
| nearest = scene | 3987 | 0.709 | 0.104 | 0.119 | 0.840 | 1.048 |
| nearest = obstacle | 4164 | 4.344 | 2.193 | 3.152 | 0.311 | 3.193 |

Frames where the nearest corridor surface is the scene, by episode obstacle type:

| Obstacle type | n | mean | median | median est/gt |
|---|---|---|---|---|
| ball_20cm_red | 353 | 0.145 | 0.135 | 1.102 |
| box_1m_black | 305 | 0.356 | 0.044 | 0.997 |
| box_1m_grey | 291 | 0.612 | 0.097 | 1.016 |
| building_block | 167 | 1.419 | 0.095 | 1.094 |
| glass_pane_alpha20 | 512 | 1.055 | 0.100 | 1.067 |
| none | 2184 | 0.656 | 0.109 | 1.045 |
| pole_10cm_axis_z | 62 | 0.146 | 0.144 | 1.144 |
| wall_white_featureless | 29 | 0.070 | 0.062 | 0.938 |
| wire_1cm_axis_z | 21 | 0.123 | 0.152 | 1.151 |
| wire_3cm_axis_z | 63 | 4.173 | 0.153 | 1.131 |
