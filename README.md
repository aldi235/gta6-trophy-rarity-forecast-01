# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/aldi235/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/aldi235/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A forecaster for Grand Theft Auto VI's PlayStation 5 trophy earn rates on
Exophase at 2026-11-26 00:00 UTC, one week after launch. Given a case file of
the house's recorded reads of the game's trophy page, `main.py` (Python 3.13,
standard library only) writes, for each of the 55 trophies, five quantiles
(5%, 25%, 50%, 75%, 95%) of its earn rate at the target, scored by pinball
loss on the logit scale.

## The model: count decomposition

An earn rate is `earned_count / players_tracked`, and both move fast after
a launch, so the program forecasts the two counts separately and divides,
instead of extrapolating the rate itself:

- **Players.** Least-squares line of `ln(players_tracked)` on
  `ln(1 + hours since launch)`, fitted on the recent distinct reads (the
  last 24 hours, or the whole history if shorter), read at the target.
  The forecast is floored at the last observed count.
- **Earned, per trophy.** Each read's earned count is reconstructed as
  `rate x players / 100`, then a least-squares line of `ln(earned + 0.5)`
  on the same clock is read at the target. Earned counts never fall, so the
  forecast is floored at the last observed count and capped at the player
  base. A trophy whose last rate is 95% or more is saturated — its earned
  count tracks the player base one for one — so its median holds the last
  rate instead of extrapolating slope noise.
- **Median.** `100 x earned(target) / players(target)`, blended in logit
  space with the last observed rate (persistence), weighted by the fits'
  quality: tight fits trust the model (up to 90%), noisy ones lean on the
  last rate.
- **Spread.** A random-walk term over the horizon (`0.45 x sqrt(days)`),
  both fits' prediction intervals at the target (leverage-adjusted), and a
  0.25-logit allowance for Exophase's refresh jumps, combined on the logit
  scale, floored at 0.35 and capped at 1.35.

Identical consecutive reads are dropped before fitting: Exophase refreshes
on its own schedule, so long runs of reads repeat the same page exactly.

## Assumptions and sources

- Rates and player counts come from the house recorder's reads of
  [Exophase's GTA VI PS5 trophy page](https://www.exophase.com/game/grand-theft-auto-vi-psn/trophies/),
  as given in each case file; the program runs offline and reads nothing
  else. Launch-night behaviour (page frozen at 638 players tracked, then a
  refresh to 3,868 players with every rate but the 100% one lower) is taken
  from the competition's rules.
- Launch dynamics are assumed to decay in log time: both the player influx
  and trophy earning slow as `ln(1 + hours since launch)`, which is why the
  fits use that clock. The last 24 hours of reads are assumed to describe
  the trend the target inherits better than the launch surge does.
- The answering read is the first valid read in the hour from the target,
  showing the rates as of Exophase's last refresh before it; the spread's
  refresh allowance covers the jump a refresh can cause.
- Current rates (checked 20 Nov 2026, ~30 hours after launch) show the
  dilution the model is built for: the platinum fell from 0.63% to 0.03%
  and most trophies from ~1% to ~0.05% as the tracked player base grew
  roughly 500-fold, while "Complete Gone Fishing" stayed at ~100%.

## Validation

- Practice cases `p00`/`p01`: valid output (55 rows, five non-decreasing
  quantiles each, 0–100), byte-identical across runs, ~0.08 s for both.
- A synthetic final-like case (308 reads over 60 hours, players
  638 → ~650k, target 4.5 days out) beats the house persistence baseline
  on pinball loss in three held-out scenarios: player growth decelerating
  (−3.1%), player growth continuing so rates fall (−6.7%), and earned
  counts accelerating so rates rise (−2.6%).

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` case in `$MAKEFUN_INPUT_DIR` (default `/input`) and
writes one `<id>.out` per case to `$MAKEFUN_OUTPUT_DIR` (default
`/output`). The grader runs `python3 /bot/main.py` with both unset.

---

<sub>Entered on [make.fun](https://make.fun/?ref=7awpzphz&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
