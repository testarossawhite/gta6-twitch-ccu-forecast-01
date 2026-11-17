# gta6-twitch-ccu-forecast-01

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/testarossawhite/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/testarossawhite/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts hours 2–24 of Grand Theft Auto VI's Twitch release day
(2026-11-19 UTC) as five quantiles (5/25/50/75/95%), from the first
hour's print alone.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=rjhnbr79&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>

## The idea

Hour 1's print is almost everything. Across 37 past big launches, a
launch already huge at 00:00 UTC stays nearly flat all day, while one
tiny at 00:00 UTC (rolling midnight, only New Zealand/Australia awake)
explodes 10–100x by the US evening: log(y_h / y_1) correlates about
−0.93 with log(y_1) at hour 19. So per hour we regress log(y_h / y_1)
on log(y_1) across the case file's own comparables — a hinge fit and a
quadratic fit, averaged in log space, quantiles from the empirical
residuals, blended 30% toward the 8 nearest launches by hour-1 size,
log(p1) clamped to the observed range (no extrapolation past the
biggest launch ever seen).

## Validation

Practice answers recovered from the kit's own public source
(SullyGnome hourly Twitch category averages; hour 1 matched the kit
exactly on all four cases):

| model | practice pinball loss | LOO over 37 comparables |
|---|---|---|
| this entry | **0.216** | **0.380** |
| house persistence baseline | 0.387 | 0.930 |

90% coverage on practice: 0.91–1.00; typical miss 31–61% (baseline 82.8%).

Python 3.13, standard library only, deterministic, ~0.2 s per case.
Reads every `*.in` in the input dir, writes `<id>.out` per case.
If a case carries too few usable comparables, that hour falls back to
the house persistence shape, so the output is always valid.
