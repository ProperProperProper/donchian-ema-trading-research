# Trading System Project Plan

## Current benchmark

The frozen 10-minute US.APP Donchian/EMA baseline remains the promotion benchmark:

- Donchian lookback: 45 completed candles
- EMA: 8 / 89
- Take profit: 6.38%
- Fixed stop: 3.95% below actual filled average cost
- Slippage assumption: 2 bps per side
- Result: +51.53% compounded return
- Maximum drawdown: -11.10%

Promotion target: compounded return above 51% and maximum drawdown better than -11.10% on the same period and execution assumptions.

## Phase 1 — Research integrity

- Keep the baseline immutable and versioned.
- Use completed candles only and enter at the next candle open.
- Separate in-sample optimization, holdout testing, and walk-forward testing.
- Report compounded return, sum of returns, maximum drawdown, win rate, profit factor, trade count, and costs.

## Phase 2 — Profit/drawdown search

Test small, interpretable grids around the baseline:

- Donchian lookback: 30, 45, 60
- EMA fast: 8, 13
- EMA slow: 89, 120
- Take profit: 6.38%, 8%, 10%
- Fixed stop: 3%, 3.95%, 5%
- Optional OHLC-only filters: EMA slope, breakout-bar close location, and volatility/channel-width thresholds

Reject any candidate that does not beat both promotion gates. Do not promote a high-return candidate with materially worse drawdown.

## Phase 3 — Dynamic stop after fill

Verify this in Moomoo Replay before using it:

1. Submit the market buy.
2. Wait until `position_holding_qty(symbol)` is positive.
3. Read `position_cost(symbol, cost_price_model=CostPriceModel.AVG)`.
4. Only then calculate the stop from the actual filled average cost.
5. Confirm the stop is evaluated on every subsequent 10-minute event.
6. Confirm a delayed or rejected buy cannot create a phantom stop or duplicate entry.

Candidate dynamic stops to test locally:

- Initial fixed stop: 3.95% below fill.
- Break-even stop after price reaches +2%.
- Trailing stop after price reaches +3%, with trail widths of 2.5%, 3%, and 3.5%.

The first trailing test reduced return to 45.78%, so it is not promoted. It remains a research candidate only.

## Phase 4 — Moomoo execution validation

- Run Detect Error.
- Run SIMULATE Replay with one symbol and one position.
- Verify actual filled buy price, position cost, quantity, stop trigger, take-profit trigger, trend exit, and close quantity in `QuantRunLogs`.
- Confirm no duplicate buys while an order is pending.
- Confirm the strategy is flat before the session close.

## Phase 5 — Promotion and release

- Promote only a candidate that passes the return and drawdown gates.
- Re-run the full-period backtest and a contiguous recent holdout.
- Save the candidate as a new Moomoo strategy version; preserve the baseline.
- Record parameters, data period, slippage, fills, and metrics in the README and research JSON.
- Keep credentials in the macOS Keychain; never commit API keys.

## Current decision

Keep the fixed-stop baseline preserved. The exit-only Green/Yellow/Red candidate is now the preferred SIMULATE/Replay candidate: it keeps the baseline entries and produced +55.48% compounded return with -9.44% maximum drawdown. Do not move it to live trading until the Moomoo filled-cost, break-even protection, red exit, and close quantity are verified in Replay.

The baseline is immutable and tagged `baseline-2bps-6.38tp`. Green/yellow/red exit experiments must be separate candidate scripts and must never modify the baseline files.
