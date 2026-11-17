# GTA VI Twitch CCU Forecast — Muse variant v3

Entry for [Forecast GTA VI's launch-day Twitch viewers hour by hour](https://make.fun/bounty/gta6-twitch-ccu-forecast-01) (make.fun, $250).

Python 3.13, standard library only. `main.py` reads a case at `$MAKEFUN_INPUT_DIR` (or `/input`) and writes one forecast per case to `$MAKEFUN_OUTPUT_DIR` (or `/output`): 23 rows, hours 2–24, five quantiles each.

## Method

Pre-midnight anchor (hour 0 from `history_hourly`), per-comparable curves in log-ratio space calibrated to the case's hour-1 print, unlock-pattern weighting, Gaussian a_case similarity weighting (σ=0.12), weighted per-hour quantiles, shrinkage K=30, spread cap 1.0. Falls back to the hour-1 anchor when no pre-midnight point exists. Deterministic, no network, no model calls.

## Practice scores (pinball loss, log scale — lower is better)

| case | v3 | v2 | Hermes variant A | kit baseline |
|------|----|----|------------------|--------------|
| p00 | 0.259206 | 0.343634 | 0.499637 | 0.846602 |
| p01 | 0.135130 | 0.187549 | 0.183889 | 0.310193 |
| p02 | 0.129290 | 0.128754 | 0.181091 | 0.202271 |
| p03 | 0.182941 | 0.184475 | 0.200298 | 0.289901 |
| **mean** | **0.176642** | 0.211103 | 0.266229 | 0.412242 |

Practice is no guide to the final. Built by OmO (oh-my-openagent) on the local FreeLLMAPI, verified independently with the kit harness.
