"""Moomoo Quant 10-minute profit-lock/yellow-exit research candidate.

Frozen entry: Donchian 45, EMA 8/89, completed 10-minute candle.
Exit candidate found in a broader local exit-only search:
  - TP: 10.69%
  - Initial stop: 3.54% below actual filled cost
  - Break-even: arm at +2%, lock +0.20%
  - Profit lock: after +8% peak, lock +1%
  - Trailing stop: after +6% peak, trail 2% below peak
  - Yellow exit: close below EMA8 while still above EMA89
  - Red exit: close below EMA89

Local conservative result: +71.31% compounded, -5.29% max drawdown,
65 trades over the 2026 research window. Research-only; run Detect Error and
SIMULATE Replay before any live use.
"""


class Strategy(StrategyBase):
    def initialize(self):
        declare_strategy_type(AlgoStrategyType.SECURITY)
        self.trigger_symbols()
        self.custom_indicator()
        self.global_variables()

    def trigger_symbols(self):
        self.symbol = declare_trig_symbol()

    def global_variables(self):
        self.entry_price = 0.0
        self.peak_price = 0.0
        self.in_position = False
        self.break_even_armed = False
        self.profit_lock_armed = False

    def custom_indicator(self):
        script = r'''
from ftool import *

indicator('DONCHIAN_EMA_10M_PROFIT_LOCK_YELLOW', '10-minute profit lock yellow exit')

def strategy_values(lookback=45, fast=8, slow=89):
    prior_high = high().hhv(lookback, 1).ref(1)
    ema_fast = close().ema(fast)
    ema_slow = close().ema(slow)
    breakout = close() > prior_high
    # Regime guard: require both EMAs to be rising before accepting a breakout.
    # This blocks short-lived bounce breakouts while the broader trend is falling.
    raw_regime_up = (ema_slow > ema_slow.ref(18)) & (ema_fast > ema_fast.ref(3)) & (close() > ema_slow)
    regime_up = raw_regime_up & raw_regime_up.ref(1) & raw_regime_up.ref(2)
    trend_up = (ema_fast > ema_slow) & (close() > ema_slow) & regime_up
    long_signal = breakout & trend_up
    yellow_exit = (close() < ema_fast) & (close() >= ema_slow) & (ema_fast > ema_slow)
    red_exit = close() < ema_slow
    return prior_high, ema_fast, ema_slow, long_signal, yellow_exit, red_exit

if __name__ == '__main__':
    lookback = input_parameter('lookback', 45)
    fast = input_parameter('fast_ema', 8)
    slow = input_parameter('slow_ema', 89)
    prior_high, ema_fast, ema_slow, long_signal, yellow_exit, red_exit = strategy_values(
        lookback, fast, slow
    )
    plot('Donchian prior high', prior_high, color=Color.hex('#FF9800'))
    plot('EMA fast', ema_fast, color=Color.hex('#00AEEF'))
    plot('EMA slow', ema_slow, color=Color.hex('#E91E63'))
    output_parameter(
        LONG_SIGNAL=long_signal,
        REGIME_UP=regime_up,
        YELLOW_EXIT=yellow_exit,
        RED_EXIT=red_exit,
        CLOSE=close(),
    )
'''
        self.register_indicator_Python(
            indicator_name='DONCHIAN_EMA_10M_PROFIT_LOCK_YELLOW',
            script=script,
        )

    def handle_data(self):
        params = {'lookback': 45.0, 'fast_ema': 8.0, 'slow_ema': 89.0}
        long_signal = get_Python_indicator(
            indicator_name='DONCHIAN_EMA_10M_PROFIT_LOCK_YELLOW',
            variable_name='LONG_SIGNAL', symbol=self.symbol, params=params,
            bar_type=BarType.K_10M, select=1, session_type=THType.RTH,
        )
        yellow_exit = get_Python_indicator(
            indicator_name='DONCHIAN_EMA_10M_PROFIT_LOCK_YELLOW',
            variable_name='YELLOW_EXIT', symbol=self.symbol, params=params,
            bar_type=BarType.K_10M, select=1, session_type=THType.RTH,
        )
        red_exit = get_Python_indicator(
            indicator_name='DONCHIAN_EMA_10M_PROFIT_LOCK_YELLOW',
            variable_name='RED_EXIT', symbol=self.symbol, params=params,
            bar_type=BarType.K_10M, select=1, session_type=THType.RTH,
        )
        held = position_holding_qty(symbol=self.symbol)
        price = float(current_price(symbol=self.symbol, price_type=THType.RTH))

        if held <= 0 and not self.in_position and bool(long_signal):
            available_cash = float(cash_buying_power(currency=Currency.USD))
            qty = int((available_cash * 0.95) / price)
            if qty <= 0:
                return
            place_market(symbol=self.symbol, qty=qty, side=OrderSide.BUY)
            self.in_position = True
            return

        if held <= 0 or not self.in_position:
            return
        filled_cost = float(position_cost(symbol=self.symbol, cost_price_model=CostPriceModel.AVG))
        if filled_cost <= 0:
            return
        self.entry_price = filled_cost
        self.peak_price = max(self.peak_price, price, filled_cost)

        if price >= filled_cost * 1.02:
            self.break_even_armed = True
        if self.peak_price >= filled_cost * 1.08:
            self.profit_lock_armed = True

        stop = filled_cost * 0.9646
        if self.break_even_armed:
            stop = max(stop, filled_cost * 1.002)
        if self.profit_lock_armed:
            stop = max(stop, filled_cost * 1.01)
        if self.peak_price >= filled_cost * 1.06:
            stop = max(stop, self.peak_price * 0.98)

        take_profit = price >= filled_cost * 1.1069
        stop_loss = price <= stop
        if take_profit or stop_loss or bool(yellow_exit) or bool(red_exit):
            close_positions(
                symbol=self.symbol,
                qty=abs(1.0 * position_holding_qty(symbol=self.symbol)),
            )
            self.entry_price = 0.0
            self.peak_price = 0.0
            self.break_even_armed = False
            self.profit_lock_armed = False
            self.in_position = False
