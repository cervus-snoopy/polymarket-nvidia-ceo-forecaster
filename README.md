# Polymarket Next-CEO-of-NVIDIA 4-hour quantile forecaster

Entry for the make.fun bounty
[`polymarket-nvidia-ceo-forecast-01`](https://make.fun/bounty/polymarket-nvidia-ceo-forecast-01)
($250, 100 places, deadline 2026-11-13 21:00 UTC / Sat 14 Nov 2026 04:00 WIB).

## What it does

`main.py` (Python 3.13, standard library only, no network, deterministic)
reads one case at a time from `$MAKEFUN_INPUT_DIR` (default `/input/`) —
every `*.in` file is a UTF-8 JSON case, schema `makefun/polymarket-case/v1`:
the Yes price of each of the event's 24 markets at every minute of the
72 hours up to a checkpoint — and writes one forecast per case to
`$MAKEFUN_OUTPUT_DIR` (default `/output/`): `/output/<id>.out`, UTF-8 JSON
`{"forecasts": [{"id": <conditionId>, "q": [[q05, q25, q50, q75, q95] x 240]}]}`,
240 rows per market (minutes at+1 … at+240), quantiles non-decreasing in
[0, 1], rounded to 6 decimals.

Run locally:

```sh
mkdir -p input output
# put a case file, e.g. input/p00.in, in input/
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=output python3 main.py
```

## Method

Prediction-market prices are near-martingales that move in steps — long
flat stretches, small ticks, occasional jumps. The house baseline
(`q = x + z_tau * 0.0025 * sqrt(h)`, practice pinball **0.363589**) uses one
fixed sigma for every market and horizon, so it is beatable on three axes:

1. **Per-market, per-horizon volatility.** For each market the scale of
   price changes is measured empirically at horizons {1, 5, 30, 120, 240}
   minutes from the case's own history: an inner scale from the IQR of
   H-step changes (jump-robust) and an outer scale from the 5%/95% range
   (jump-aware tails). The 1-minute scale is a MAD estimator over the last
   24 h of diffs. Scales interpolate log-linearly in horizon, so the band
   grows like the market's own realised diffusion.
2. **Shrunk drift.** The mean 1-minute change over the last 6 h,
   winsorized, shrunk 0.7× toward zero, clipped to ±0.0002/min.
3. **Cross-market sum discipline.** The 24 outcomes are mutually
   exclusive, so medians should sum to ~1; if a horizon's median sum
   deviates by more than 0.02, each market's row is shifted 40% of the way
   back toward a sum of 1.

Quantiles use standard-normal z per level; inner levels (25/50/75%) use
the inner scale, outer levels (5/95%) the outer scale, so jump-heavy
markets get honestly wider tails while quiet markets stay sharp — exactly
what pinball loss rewards. Full detail in the `main.py` docstring.

## Files

| File | What it is |
| --- | --- |
| `main.py` | The entry (repository root, as the rules require) |
| `LICENSE` | MIT |
| `README.md` | This file |
| `BUILD_LOG.txt` | Timestamped build/test log (transcript support) |
| `tests/` | Local harness only: sample-case generator, format validator, pinball scorer vs the house baseline (not part of the entry logic) |

## Local test results

See `BUILD_LOG.txt` for the full timestamped record. Final numbers
(14 Nov 2026, ~00:35 WIB, Python 3.12.3 local; target sandbox Python 3.13):
realistic 24-market × 4,320-minute synthetic case — two runs byte-identical
(SHA-256 `d30d8b6a…` pre-final-edit; final file re-verified identical across
two runs); output format valid (24 markets, 240 rows each, monotone
quantiles in [0, 1], 6-dp rounding, 266,279 bytes < 1,048,576); runtime
0.33–0.40 s and ~18 MB peak RSS vs the 8 s / 1 CPU / 1,024 MB limits;
synthetic backtest pinball **1.351966** vs house baseline **1.554134**
(**13.01% lower**), 90% coverage 0.8155 vs baseline 0.5424. Edge cases
(1-market 3-price case from the rules' own example, constant-price and
1-price markets) all produce valid output; the 3-price case reproduces
the house baseline's published example rows exactly.

## Honest caveats

- The exact input/output format was verified against the bounty rules
  published on make.fun (API `rules` field, fetched 2026-11-13 ~17:26 UTC):
  input keys, output shape, 240 rows, 5 quantiles, ≤1,048,576-byte files,
  8 s / 1 CPU / 1,024 MB per case, deterministic byte-identical reruns.
- The practice cases are scored on real Polymarket history the author
  could not fetch in full here; local validation used a synthetic case
  calibrated to the rules' description (step moves, jumps, ~1.0 median
  sum). A synthetic backtest win is **not** proof of a practice win —
  the first real check is the practice grade after submission.
- No submission, repo creation, wallet linking, or payment was done from
  this build; those steps need the account owner in the browser.
