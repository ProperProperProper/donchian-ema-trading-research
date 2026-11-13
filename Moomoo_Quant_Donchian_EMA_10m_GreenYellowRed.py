"""Moomoo Algo/Quant copy-paste strategy.

Green/Yellow/Red 10-minute Donchian/EMA exit candidate:
  - Donchian lookback: 45 completed candles
  - EMA fast/slow: 8 / 89
  - Take profit: 8.00% (exit parameter set)
  - Stop loss: 3.95%
  - Break-even protection activates after +3.00%
  - Red state exits when close falls below EMA89
  - Long only; one position at a time

Paste this entire file into Moomoo Quant's Python strategy editor.
Run Detect Error and Replay in SIMULATE mode before enabling live trading.
The trigger symbol is selected by Quant; position size is calculated from
available USD cash, using 95% of cash per entry.
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
        # Internal state only. These must not appear as user-editable parameters.
        self.entry_price = 0.0
        self.in_position = False
        self.break_even_armed = False

    def custom_indicator(self):
        script = r'''
from ftool import *

indicator('DONCHIAN_EMA_10M_GYR', '10-minute Green Yellow Red exits')

def strategy_values(lookback=45, fast=8, slow=89):
    # Every value is based on completed candles. The one-bar shift on the
    # Donchian high prevents the current candle from defining its own breakout.
    prior_high = high().hhv(lookback, 1).ref(1)
    ema_fast = close().ema(fast)
    ema_slow = close().ema(slow)
    breakout = close() > prior_high
    trend_up = (ema_fast > ema_slow) & (close() > ema_slow)
    long_signal = breakout & trend_up
    green_state = (close() >= ema_fast) & (ema_fast > ema_slow)
    yellow_state = (close() < ema_fast) & (close() >= ema_slow) & (ema_fast > ema_slow)
    red_exit = close() < ema_slow
    return prior_high, ema_fast, ema_slow, long_signal, green_state, yellow_state, red_exit

if __name__ == '__main__':
    lookback = input_parameter('lookback', 45)
    fast = input_parameter('fast_ema', 8)
    slow = input_parameter('slow_ema', 89)
    prior_high, ema_fast, ema_slow, long_signal, green_state, yellow_state, red_exit = strategy_values(
        lookback, fast, slow
    )
    plot('Donchian prior high', prior_high, color=Color.hex('#FF9800'))
    plot('EMA fast', ema_fast, color=Color.hex('#00AEEF'))
    plot('EMA slow', ema_slow, color=Color.hex('#E91E63'))
    output_parameter(
        LONG_SIGNAL=long_signal,
        GREEN_STATE=green_state,
        YELLOW_STATE=yellow_state,
        RED_EXIT=red_exit,
        CLOSE=close(),
    )
'''
        self.register_indicator_Python(
            indicator_name='DONCHIAN_EMA_10M_GYR',
            script=script,
        )

    def handle_data(self):
        params = {'lookback': 45.0, 'fast_ema': 8.0, 'slow_ema': 89.0}
        long_signal = get_Python_indicator(
            indicator_name='DONCHIAN_EMA_10M_GYR',
            variable_name='LONG_SIGNAL',
            symbol=self.symbol,
            params=params,
            bar_type=BarType.K_10M,
            select=1,
            session_type=THType.RTH,
        )
        red_exit = get_Python_indicator(
            indicator_name='DONCHIAN_EMA_10M_GYR',
            variable_name='RED_EXIT',
            symbol=self.symbol,
            params=params,
            bar_type=BarType.K_10M,
            select=1,
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

        # Use the actual filled average cost, not the signal candle price.
        filled_cost = float(
            position_cost(symbol=self.symbol, cost_price_model=CostPriceModel.AVG)
        )
        if filled_cost <= 0:
            return
        self.entry_price = filled_cost

        target = filled_cost * 1.08
        if price >= filled_cost * 1.03:
            self.break_even_armed = True
        stop = filled_cost if self.break_even_armed else filled_cost * 0.9605
        take_profit = price >= target
        stop_loss = price <= stop

        if take_profit or stop_loss or bool(red_exit):
            close_positions(
                symbol=self.symbol,
                qty=abs(1.0 * position_holding_qty(symbol=self.symbol)),
            )
            self.entry_price = 0.0
            self.break_even_armed = False
            self.in_position = False

