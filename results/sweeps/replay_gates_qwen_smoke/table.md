backend airsim, fast stack da2, delay 1.0 s, split val

Judge: qwen3-vl-8b

| Policy | flights | crash rate [95% CI] | crashes/km | success | calls/flight | query rate | mission time (s) | safety retained [95% CI] |
|---|---|---|---|---|---|---|---|---|
| gate_l1_q10 | 2 | 0.00 [0.00 to 0.66] | 0.00 | 1.00 | 1.0 | 0.010 | 10.6 | n/a |
| gate_l3_q10 | 2 | 0.00 [0.00 to 0.66] | 0.00 | 1.00 | 1.0 | 0.010 | 10.6 | n/a |
| never | 2 | 0.50 [0.09 to 0.91] | 15.50 | 0.50 | 0.0 | 0.000 | 6.6 | n/a |
| oracle | 2 | 0.00 [0.00 to 0.66] | 0.00 | 1.00 | 9.5 | 0.091 | 19.9 | n/a |
| random_p02 | 2 | 0.50 [0.09 to 0.91] | 14.10 | 0.50 | 1.0 | 0.012 | 9.2 | n/a |
| threshold_far_q10 | 2 | 0.50 [0.09 to 0.91] | 12.00 | 0.50 | 4.0 | 0.035 | 15.3 | n/a |
