"""Moomoo Quant daily high-profit research candidate.

Daily-specific candidate selected for highest compounded profit with a better
risk profile than the other top full-sample result:
  - Donchian lookback: 60 completed daily candles
  - EMA fast/slow: 5 / 40
  - Take profit: 8.00%
  - Initial stop: 3.00% below actual filled average cost
  - Red exit: completed close below EMA40
  - Long only; one position at a time

This is a research candidate, separate from the immutable 10-minute baseline
and the earlier daily candidate. Run Detect Error and SIMULATE Replay first.
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
        self.in_position = False

    def custom_indicator(self):
        script = r'''
from ftool import *

indicator('DONCHIAN_EMA_1D_HIGH_PROFIT', 'Daily high-profit Donchian EMA')

def strategy_values(lookback=60, fast=5, slow=40):
    prior_high = high().hhv(lookback, 1).ref(1)
    ema_fast = close().ema(fast)
    ema_slow = close().ema(slow)
    breakout = close() > prior_high
    trend_up = (ema_fast > ema_slow) & (close() > ema_slow)
    long_signal = breakout & trend_up
    red_exit = close() < ema_slow
    return prior_high, ema_fast, ema_slow, long_signal, red_exit

if __name__ == '__main__':
    lookback = input_parameter('lookback', 60)
    fast = input_parameter('fast_ema', 5)
    slow = input_parameter('slow_ema', 40)
    prior_high, ema_fast, ema_slow, long_signal, red_exit = strategy_values(
        lookback, fast, slow
    )
    plot('Daily Donchian prior high', prior_high, color=Color.hex('#FF9800'))
    plot('Daily EMA fast', ema_fast, color=Color.hex('#00AEEF'))
    plot('Daily EMA slow', ema_slow, color=Color.hex('#E91E63'))
    output_parameter(
        LONG_SIGNAL=long_signal,
        RED_EXIT=red_exit,
        CLOSE=close(),
    )
'''
        self.register_indicator_Python(
            indicator_name='DONCHIAN_EMA_1D_HIGH_PROFIT',
            script=script,
        )

    def handle_data(self):
        params = {'lookback': 60.0, 'fast_ema': 5.0, 'slow_ema': 40.0}
        long_signal = get_Python_indicator(
            indicator_name='DONCHIAN_EMA_1D_HIGH_PROFIT',
            variable_name='LONG_SIGNAL',
            symbol=self.symbol,
            params=params,
            bar_type=BarType.K_DAY,
            select=2,
            session_type=THType.RTH,
        )
        red_exit = get_Python_indicator(
            indicator_name='DONCHIAN_EMA_1D_HIGH_PROFIT',
            variable_name='RED_EXIT',
            symbol=self.symbol,
            params=params,
            bar_type=BarType.K_DAY,
            select=2,
            session_type=THType.RTH,
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

        # Set risk levels only after the market buy has actually filled.
        filled_cost = float(
            position_cost(symbol=self.symbol, cost_price_model=CostPriceModel.AVG)
        )
        if filled_cost <= 0:
            return
        self.entry_price = filled_cost
        target = filled_cost * 1.08
        stop = filled_cost * 0.97

        if price >= target or price <= stop or bool(red_exit):
            close_positions(
                symbol=self.symbol,
                qty=abs(1.0 * position_holding_qty(symbol=self.symbol)),
            )
            self.entry_price = 0.0
            self.in_position = False
