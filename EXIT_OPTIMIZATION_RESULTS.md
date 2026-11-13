# Exit-only optimization results

Entry parameters were frozen to the immutable baseline:

- Donchian lookback: 45 completed 10-minute candles
- EMA fast/slow: 8 / 89
- Next-bar entry after breakout
- 2 bps slippage per side

## Control

The immutable fixed-exit baseline produced:

- 78 trades
- +51.53% compounded return
- -11.10% maximum drawdown

## Promoted exit candidate

The best exit-only candidate that improved both return and drawdown used:

- Take profit: 8%
- Initial stop: 3.95% below actual fill cost
- Break-even protection: arm after +3%
- Red/bear exit: close below EMA89
- Green: close at/above EMA8 while EMA8 > EMA89
- Yellow: close below EMA8 but at/above EMA89 while EMA8 > EMA89

Full-period result:

- 74 trades
- +55.48% compounded return
- -9.44% maximum drawdown
- 59.46% win rate

Split checks:

- January–June: +27.70%, -9.16% drawdown
- July–November: +21.76%, -9.44% drawdown

The entry logic was not changed. This candidate is separate from the immutable baseline and must pass Moomoo Detect Error and SIMULATE Replay before any live promotion.

## Dynamic parameter discovery pass

NVIDIA was used to design a small exit-only search around the promoted candidate.
The local test froze the entries and discovered dynamic values only after the
next-bar buy fill, using the actual filled cost and completed 10-minute bars.
It tested take profit, initial stop, break-even trigger/offset, peak-based
trailing activation/distance, ATR stop candidates, and yellow/red thresholds.

The no-lookahead result did not justify a new Moomoo promotion:

- The best tested dynamic/trailing configuration was approximately +53.59%
  compounded with -10.54% max drawdown.
- A yellow exit after +1% profit reduced the result to approximately +50.35%,
  below the 51% guardrail.
- The existing promoted configuration remains the preferred candidate at
  +55.48% compounded and -9.44% drawdown from the prior exit-only evaluation.

The optimizer source and full candidate output are preserved in
`BBI_5m_Backtest/donchian_ema_10m_exit_optimizer.py` and
`BBI_5m_Backtest/US_APP_donchian_ema_exit_optimizer.json`. No baseline or
Moomoo production candidate was modified by this pass.

## NVIDIA drawdown research pass

NVIDIA ranked volatility-aware trailing, time-in-trade limits, and early
adverse-excursion controls as the most plausible drawdown levers, while warning
that tighter yellow exits could damage compounding. A focused local test used
the frozen entries and tested:

- initial stops of 3.75% and 3.95%;
- break-even triggers of +3% and +3.5%;
- time caps of 2, 4, and 6 hours;
- early adverse-excursion cuts of 2% and 2.5% during the first 2 hours.

Across 80 combinations, no qualifying configuration improved the existing
profit/drawdown frontier. The best qualifying result remained the current
EMA89-red exit configuration at approximately +53.59% compounded and -10.54%
drawdown under this conservative local model. The NVIDIA recommendations are
therefore treated as research direction, not a promotion decision.

The focused test and output are preserved in
`BBI_5m_Backtest/donchian_ema_10m_drawdown_research.py` and
`BBI_5m_Backtest/US_APP_donchian_ema_drawdown_research.json`.

## Daily-candle comparison

The same Donchian/EMA parameters were run on daily candles as a separate
comparison. With history from 2023 through November 2026, the daily version
produced 36 trades, +58.22% compounded return, 41.67% win rate, and -30.55%
maximum drawdown. On the matching 2026-only window it had just 1 completed
trade and +7.98%, so that slice is not statistically useful.

Daily candles therefore show higher raw compounded return over the longer
sample but materially worse drawdown and too few current-year trades. No Quant
code was changed; this is research only.

## Daily parameter optimization

A 900-combination daily search tested Donchian lookback, EMA fast/slow, TP,
and initial stop. The best drawdown-oriented candidate in the full sample was:

- Donchian lookback: 60 days
- EMA fast/slow: 13 / 89
- TP: 8%
- Initial stop: 5%
- 31 trades, +84.72% compounded, -20.72% max drawdown

This is better than the daily default's -30.55% drawdown, but its 2026 holdout
contains only one trade, so it is not robust enough to promote to Quant. The
highest-return search candidate reached +107.63% with -26.41% drawdown, which
is also inferior on risk and has weak holdout evidence. The daily results are
therefore research candidates only; the 10-minute baseline and Quant scripts
remain unchanged.

## Randomized daily robustness check

The daily search was reworked into a reproducible randomized sample of 300
parameter sets, evaluated over six rolling six-month validation windows. The
selection score favored median validation return, worst validation drawdown,
and then full-sample return.

The top median-validation candidate was 75-day Donchian, EMA 12/135, 11.75%
TP, and 5.95% stop: +126.30% full-sample return and -16.96% full-sample
drawdown. However, it still had negative validation folds and only four folds
with at least three trades. No sampled candidate was profitable in every
validation window. This indicates daily parameters are regime-sensitive, not
that a safe Quant configuration has been found.

The randomized robustness source and output are preserved in
`BBI_5m_Backtest/donchian_ema_daily_robustness.py` and
`BBI_5m_Backtest/US_APP_donchian_ema_daily_robustness.json`. No Quant code was
updated.

## Daily trailing-exit test

Trailing exits were tested on the high-profit daily candidate using the same
60-day Donchian and EMA 5/40 entries, 3% initial stop, and 8% TP. Across
activation levels from 1% to 8% and trailing distances from 0.5% to 5%, every
configuration matched the fixed-exit control: +131.06% compounded, -21.75%
drawdown, and 34 trades. The fixed TP or EMA40 red exit occurred first, so the
trailing stop added no protection.

Replacing the fixed TP with a trailing-only exit increased the full-sample
return to about +212.98% but worsened drawdown to -26.41% and reduced the win
rate to 13.64%. It is therefore not promoted. The daily high-profit Quant
candidate remains fixed-TP/fixed-initial-stop.

## Jason McIntosh-style daily exit test

Jason McIntosh's documented approach uses a wide ATR-based trailing stop—he
describes using a multiple of 10—and a 60-trading-day time stop for positions
that remain at a loss. He also emphasizes letting winners run rather than
capping them with a fixed target.

I tested a daily approximation on the 60-day Donchian / EMA 5/40 entries:
ATR(14) trailing stop, 6x–12x ATR search, no fixed TP, optional EMA40 red exit,
and a 60-day loss-only time stop. The best full-history variant with EMA40
confirmation reached +556.63% compounded and -15.86% drawdown over 10 trades,
but failed the recent regime: -28.91% from 2025 onward and -13.55% in 2026.

It is rejected for Quant despite the attractive historical return. No
Jason-style script was promoted.

## Full one-year parameter search

For the latest one-year window, 2025-11-12 through 2026-11-12, a randomized
search tested 3,000 combinations across daily breakout lookback, fast/slow
EMAs, fixed TP/SL, ATR period/multiple, red-exit mode, and loss-only time stop.

The best result with at least three completed trades was approximately +14.92%
with 5 trades; the earlier structured search's best was +17.30% with only 3
trades. No ATR/trailing configuration produced a better one-year result. The
data therefore does not support a genuinely high-profit daily system over this
one-year window, and no new Quant parameters were promoted.

The full search is preserved in
`BBI_5m_Backtest/donchian_ema_daily_one_year_all_optimizer.py` and
`BBI_5m_Backtest/US_APP_donchian_ema_daily_one_year_all_results.json`.

## Optimized interval comparison

Each interval was separately randomized over lookback, fast/slow EMA, TP, and
stop for the same one-year window. The best sampled results were:

| Interval | Compounded | Max DD | Trades | Best sampled parameters |
|---|---:|---:|---:|---|
| 5m | +56.46% | -14.75% | 119 | LB 41, EMA 9/193, TP 45.07%, SL 4.10% |
| 10m | **+57.78%** | -17.14% | 129 | LB 33, EMA 17/26, TP 2.74%, SL 13.24% |
| 15m | +30.59% | -13.08% | 94 | LB 22, EMA 2/92, TP 16.80%, SL 17.10% |
| 30m | +20.65% | -6.21% | 27 | LB 60, EMA 21/150, TP 45.54%, SL 15.10% |
| Daily | +9.72% | -4.25% | 6 | LB 28, EMA 4/10, TP 9.42%, SL 2.13% |

The best interval is therefore **10 minutes**, narrowly ahead of 5 minutes.
These are randomized one-year fits and require a separate holdout before any
parameter promotion. No immutable baseline or Quant script was changed.

## Exit-only optimization by interval

With each interval's sampled entry parameters frozen, an exit-only search tested
fixed TP/SL, EMA red exits, break-even, trailing, ATR trailing, and loss-only
time stops. Best sampled exit configurations were:

| Interval | Exit configuration | Compounded | Max DD | Trades |
|---|---|---:|---:|---:|
| 5m | Fixed TP 45.07%, SL 4.10% | **+54.53%** | -16.30% | 120 |
| 10m | ATR(24) × 4.60, no fixed TP/red exit | +46.29% | -14.28% | 111 |
| 15m | ATR(7) × 5.11, EMA red exit, 40-bar loss time cap | +38.69% | -11.04% | 94 |
| 30m | Fixed TP 6.11%, SL 8.56%, break-even/trail candidate | +21.68% | -6.21% | 27 |
| Daily | Fixed TP 9.42%, SL 2.13% | +9.72% | -4.25% | 6 |

The 5-minute fixed exit had the highest sampled profit; the 10-minute ATR exit
had lower drawdown but also lower profit. These are research candidates only;
the 10-minute baseline remains immutable and no Quant script was changed.

The optimizer and complete output are preserved in
`BBI_5m_Backtest/donchian_ema_interval_exit_optimizer.py` and
`BBI_5m_Backtest/US_APP_donchian_ema_interval_exit_results.json`.

## Advanced 10-minute drawdown search

A broader 2,000-sample exit search tested interactions that the earlier simple
grid did not cover: break-even offsets, two-stage profit locks, percentage
trailing after a favorable excursion, confirmed red exits, yellow-state exits,
and loss-only time stops. Entries remained the immutable Donchian 45 / EMA 8/89
signal.

The best drawdown-oriented candidate was:

- TP 10.69%; initial stop 3.54%
- Break-even at +2%, locking +0.20%
- At +8% peak, lock +1%
- After +6% peak, trail 2% below peak
- Exit on yellow or one confirmed red bar

Local result over the 2026 research window: 65 trades, 78.46% win rate,
+71.31% compounded return, and -5.29% maximum drawdown. Split checks were
+44.23% in the first half (-5.29% drawdown) and +18.78% in the second half
(-4.90% drawdown). This is the first exit-only candidate in the research that
materially improves drawdown while remaining well above the 51% profit floor.

It is still a research candidate until Moomoo Detect Error and SIMULATE Replay
confirm fills and P/L. The candidate source is
`Moomoo_Quant_Donchian_EMA_10m_ProfitLockYellow.py`; the search output is
`BBI_5m_Backtest/US_APP_donchian_ema_advanced_exit_results.json`.
