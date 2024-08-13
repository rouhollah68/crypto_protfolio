from strategies.strategy import Strategy
from market_data import Trade

class AlternatingStrategy(Strategy):
    def __init__(self, assets):
        self.assets = assets
        self.position = {asset: 0 for asset in self.assets}
        self.last_trade_time = None
        self.trade_interval = 60
        # seconds
    def on_event(self, params, market_data,cash,assets_quantity):
        current_time = market_data.timestamp

        if self.last_trade_time is None:
            self.last_trade_time = current_time
            return None

        time_diff = (current_time - self.last_trade_time).total_seconds()

        if time_diff >= self.trade_interval:
            self.last_trade_time = current_time
            current_price = (market_data.best_ask + market_data.best_bid) / 2

            if self.position[market_data.asset] == 0:
                trade = Trade(current_time, market_data.asset, 'buy', market_data.best_ask, cash/market_data.best_ask)
                self.position[market_data.asset] = 1
                return trade
            elif self.position[market_data.asset] == 1:
                trade = Trade(current_time, market_data.asset, 'sell', market_data.best_bid, assets_quantity)
                self.position[market_data.asset] = 0
                return trade

        return None