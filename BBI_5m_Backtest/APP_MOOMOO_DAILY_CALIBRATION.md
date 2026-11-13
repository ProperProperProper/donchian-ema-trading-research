# APP daily calibration reference

This comparison does not modify the ASX strategy. `APP` is a separate US
comparison instrument in the local portal.

Window: 1 January 2026 through 12 November 2026, daily candles, 2 bps
slippage, next-bar-open execution, 95% allocation, AUD 100,000 notional.

| Configuration | Trades | Compounded return | Max drawdown |
|---|---:|---:|---:|
| Current ASX profit-lock/yellow tester | 1 | +1.682% | -6.629% |
| Moomoo-compatible daily settings: Donchian 45, EMA 8/89, 8% target, 3.95% stop, no dynamic exits | 1 | +8.871% | 0.000% |
| Saved Moomoo daily reference | 1 | +7.978% | 0.000% |

The local Moomoo-compatible result is 0.893 percentage points above the saved
Moomoo result. The remaining difference is attributable to the underlying
Yahoo versus Moomoo daily candles and fill data. The ASX profit-lock strategy
and its defaults remain unchanged.
