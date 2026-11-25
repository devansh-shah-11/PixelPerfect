backend airsim, fast stack da2, delay 1.0 s, split val

Judge: qwen3-vl-8b

| Policy | flights | crash rate [95% CI] | crashes/km | success | calls/flight | query rate | mission time (s) | safety retained [95% CI] |
|---|---|---|---|---|---|---|---|---|
| gate_l1_q10 | 5 | 0.00 [0.00 to 0.43] | 0.00 | 1.00 | 1.2 | 0.011 | 12.0 | n/a |
| gate_l3_q10 | 5 | 0.00 [0.00 to 0.43] | 0.00 | 1.00 | 0.2 | 0.002 | 10.3 | n/a |
| never | 5 | 0.20 [0.04 to 0.62] | 4.95 | 0.80 | 0.0 | 0.000 | 9.2 | n/a |
| oracle | 5 | 0.00 [0.00 to 0.43] | 0.00 | 1.00 | 5.6 | 0.051 | 16.7 | n/a |
| random_p02 | 5 | 0.20 [0.04 to 0.62] | 4.43 | 0.80 | 1.6 | 0.014 | 13.2 | n/a |
| threshold_far_q10 | 5 | 0.20 [0.04 to 0.62] | 4.49 | 0.80 | 2.0 | 0.018 | 13.3 | n/a |
