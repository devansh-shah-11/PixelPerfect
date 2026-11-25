backend airsim, fast stack da2, delay 1.0 s, split val

Judge: surrogate VLM

| Policy | flights | crash rate [95% CI] | crashes/km | success | calls/flight | query rate | mission time (s) | safety retained [95% CI] |
|---|---|---|---|---|---|---|---|---|
| random_p01 | 60 | 0.37 [0.26 to 0.49] | 10.48 | 0.63 | 0.8 | 0.010 | 8.6 | n/a |
| random_p02 | 60 | 0.28 [0.19 to 0.41] | 7.61 | 0.72 | 1.3 | 0.015 | 9.7 | n/a |
| threshold_far_q05 | 60 | 0.38 [0.27 to 0.51] | 11.08 | 0.62 | 0.5 | 0.007 | 8.1 | n/a |
| threshold_far_q10 | 60 | 0.28 [0.19 to 0.41] | 7.36 | 0.72 | 1.0 | 0.011 | 10.0 | n/a |
