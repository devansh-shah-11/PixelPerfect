backend airsim, fast stack true-depth, delay 1.0 s, split val

Judge: surrogate VLM

| Policy | flights | crash rate [95% CI] | crashes/km | success | calls/flight | query rate | mission time (s) | safety retained, reference = 1 [95% CI] |
|---|---|---|---|---|---|---|---|---|
| always | 20 | 0.10 [0.03 to 0.30] | 1.45 | 0.75 | 20.9 | 0.099 | 42.0 | n/a |
| never | 20 | 0.00 [0.00 to 0.16] | 0.00 | 0.95 | 0.0 | 0.000 | 14.5 | n/a |
