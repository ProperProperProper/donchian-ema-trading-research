"""Moomoo Quant version of the GitHub 57% APP 10-minute candidate.

Parameters: Donchian 33, EMA 17/26, TP 2.74%, stop 13.24%, RTH.
Load this file as a new Moomoo strategy and run Detect Error before Replay.
"""


class Strategy(StrategyBase):
    def initialize(self):
        declare_strategy_type(AlgoStrategyType.SECURITY)
        self.symbol = declare_trig_symbol()
        self.register_indicator_Python(indicator_name='NEW_STRATEGY24_10M', script=r'''
from ftool import *
indicator('NEW_STRATEGY24_10M', 'GitHub 57 percent APP 10-minute candidate')
def values(lookback=33, fast=17, slow=26):
    prior_high = high().hhv(lookback, 1).ref(1)
    ema_fast = close().ema(fast)
    ema_slow = close().ema(slow)
    regime_ema = close().ema(89)
    regime_drift = (regime_ema / regime_ema.ref(12)) - 1
    downtrend_blocked = (regime_drift <= -0.001) & (close() < regime_ema)
    long_signal = ((close() > prior_high) & (ema_fast > ema_slow) & (close() > ema_slow)
                   & ~downtrend_blocked)
    trend_exit = (close() < ema_slow) & (ema_fast < ema_slow)
    return prior_high, ema_fast, ema_slow, long_signal, trend_exit, downtrend_blocked
if __name__ == '__main__':
    p = values(input_parameter('lookback', 33), input_parameter('fast_ema', 17), input_parameter('slow_ema', 26))
    plot('Donchian prior high', p[0], color=Color.hex('#FF9800'))
    plot('EMA fast', p[1], color=Color.hex('#00AEEF'))
    plot('EMA slow', p[2], color=Color.hex('#E91E63'))
    output_parameter(LONG_SIGNAL=p[3], TREND_EXIT=p[4], DOWN_TREND_BLOCKED=p[5], CLOSE=close())
''')

    def handle_data(self):
        params = {'lookback': 33.0, 'fast_ema': 17.0, 'slow_ema': 26.0}
        long_signal = get_Python_indicator(indicator_name='NEW_STRATEGY24_10M', variable_name='LONG_SIGNAL', symbol=self.symbol, params=params, bar_type=BarType.K_10M, select=1, session_type=THType.RTH)
        trend_exit = get_Python_indicator(indicator_name='NEW_STRATEGY24_10M', variable_name='TREND_EXIT', symbol=self.symbol, params=params, bar_type=BarType.K_10M, select=1, session_type=THType.RTH)
        held = position_holding_qty(symbol=self.symbol)
        price = float(current_price(symbol=self.symbol, price_type=THType.RTH))
        if held <= 0 and bool(long_signal):
            qty = int(float(cash_buying_power(currency=Currency.USD)) * 0.95 / price)
            if qty > 0: place_market(symbol=self.symbol, qty=qty, side=OrderSide.BUY)
            return
        if held > 0:
            avg = float(position_cost(symbol=self.symbol, cost_price_model=CostPriceModel.AVG))
            if price >= avg * 1.0274 or price <= avg * (1 - 0.1324) or bool(trend_exit):
                close_positions(symbol=self.symbol, qty=abs(1.0 * position_holding_qty(symbol=self.symbol)))
