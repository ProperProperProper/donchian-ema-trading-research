"""Moomoo Quant daily Jason McIntosh-style exit research candidate.

Frozen daily entry: Donchian 60, EMA 5/40, completed daily candle.
Exit approximation based on Jason McIntosh's publicly described approach:
  - no fixed take-profit;
  - wide ATR(14) trailing stop at 10x ATR;
  - red exit when completed close falls below EMA40;
  - loss-only time stop after 60 completed daily bars.

Research-only. Run Detect Error and SIMULATE Replay before any live use.
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
        self.bars_held = 0
        self.last_completed_close = 0.0

    def custom_indicator(self):
        script = r'''
from ftool import *

indicator('DONCHIAN_EMA_1D_JASON_ATR', 'Daily Jason ATR exit research')

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
            indicator_name='DONCHIAN_EMA_1D_JASON_ATR',
            script=script,
        )

    def handle_data(self):
        params = {'lookback': 60.0, 'fast_ema': 5.0, 'slow_ema': 40.0}
        long_signal = get_Python_indicator(
            indicator_name='DONCHIAN_EMA_1D_JASON_ATR',
            variable_name='LONG_SIGNAL',
            symbol=self.symbol,
            params=params,
            bar_type=BarType.K_DAY,
            select=2,
            session_type=THType.RTH,
        )
        red_exit = get_Python_indicator(
            indicator_name='DONCHIAN_EMA_1D_JASON_ATR',
            variable_name='RED_EXIT',
            symbol=self.symbol,
            params=params,
            bar_type=BarType.K_DAY,
            select=2,
            session_type=THType.RTH,
        )
        completed_close = float(get_Python_indicator(
            indicator_name='DONCHIAN_EMA_1D_JASON_ATR',
            variable_name='CLOSE',
            symbol=self.symbol,
            params=params,
            bar_type=BarType.K_DAY,
            select=2,
            session_type=THType.RTH,
        ))
        atr_value = float(atr_atr(
            symbol=self.symbol,
            period=14,
            bar_type=BarType.K_DAY,
            select=2,
            session_type=THType.RTH,
        ))
        held = position_holding_qty(symbol=self.symbol)
        price = float(current_price(symbol=self.symbol, price_type=THType.RTH))

        if held <= 0 and not self.in_position and bool(long_signal):
            available_cash = float(cash_buying_power(currency=Currency.USD))
            qty = int((available_cash * 0.95) / price)
            if qty <= 0:
                return
            place_market(symbol=self.symbol, qty=qty, side=OrderSide.BUY)
            self.in_position = True
            self.bars_held = 0
            self.last_completed_close = completed_close
            return

        if held <= 0 or not self.in_position:
            return

        # Discover the actual fill first, then calculate all risk levels.
        filled_cost = float(
            position_cost(symbol=self.symbol, cost_price_model=CostPriceModel.AVG)
        )
        if filled_cost <= 0 or atr_value <= 0:
            return
        self.entry_price = filled_cost
        if self.peak_price <= 0:
            self.peak_price = max(filled_cost, price)
        self.peak_price = max(self.peak_price, price)

        # Count completed daily candles once, using the completed indicator
        # close returned from select=2.
        if completed_close != self.last_completed_close:
            self.bars_held += 1
            self.last_completed_close = completed_close

        trailing_stop = self.peak_price - (10.0 * atr_value)
        loss_time_stop = self.bars_held >= 60 and price < filled_cost
        stop_hit = price <= trailing_stop

        if stop_hit or loss_time_stop or bool(red_exit):
            close_positions(
                symbol=self.symbol,
                qty=abs(1.0 * position_holding_qty(symbol=self.symbol)),
            )
            self.entry_price = 0.0
            self.peak_price = 0.0
            self.bars_held = 0
            self.last_completed_close = 0.0
            self.in_position = False
