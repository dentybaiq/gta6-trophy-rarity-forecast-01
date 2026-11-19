[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/dentybaiq/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/dentybaiq/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

# GTA VI Trophy Rarity Forecast

Forecast how many GTA VI players earn each trophy in week one — an earners-vs-players model on Exophase data.

Built for the [make.fun bounty](https://make.fun/bounty/gta6-trophy-rarity-forecast-01).

## Run

```sh
python3 main.py
```

## How it works

An Exophase rate is two counts, not one — players tracked N(t) and per-trophy earners E_i(t). The forecaster collapses consecutive identical reads to distinct refresh snapshots, fits a power law in log time to log N (recency-weighted), fits a power law per trophy to log(E_i + 0.5) with empirical-Bayes shrinkage toward the cross-trophy median, and forecasts rate = E(target)/N(target) for 2026-11-26T00:00:00Z, with five quantiles per trophy on the logit scale.

Referral code: hvjtrkxx
