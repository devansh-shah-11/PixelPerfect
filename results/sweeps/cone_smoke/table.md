backend airsim, fast stack da2, delay 1.0 s, split val

Judge: surrogate VLM

| Policy | flights | crash rate [95% CI] | crashes/km | success | calls/flight | query rate | mission time (s) | safety retained, reference = 1 [95% CI] |
|---|---|---|---|---|---|---|---|---|
| always | 20 | 0.05 [0.01 to 0.24] | 0.74 | 0.85 | 20.4 | 0.099 | 40.8 | n/a |
| never | 20 | 0.35 [0.18 to 0.57] | 8.08 | 0.65 | 0.0 | 0.000 | 9.4 | n/a |
