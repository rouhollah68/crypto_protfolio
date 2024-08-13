import pandas as pd
from typing import Any, List, Optional

class MarketData:
    def __init__(self, timestamp, best_bid, best_bid_volume, best_ask, best_ask_volume, asset):
        self.timestamp = timestamp
        self.best_bid = best_bid
        self.best_bid_volume = best_bid_volume
        self.best_ask = best_ask
        self.best_ask_volume = best_ask_volume
        self.asset = asset

    def __repr__(self):
        return (f"MarketData(timestamp={self.timestamp}, best_bid={self.best_bid}, "
                f"best_bid_volume={self.best_bid_volume}, best_ask={self.best_ask}, "
                f"best_ask_volume={self.best_ask_volume}, asset={self.asset})")
    

class Trade:
    def __init__(self, timestamp: Any, asset: str, trade_type: str, price: float, volume: int):
        self.timestamp = timestamp
        self.asset = asset
        self.trade_type = trade_type  # 'buy' or 'sell'
        self.price = price
        self.volume = volume

    def __repr__(self):
        return (f"Trade(timestamp={self.timestamp}, asset={self.asset}, trade_type={self.trade_type}, "
                f"price={self.price}, volume={self.volume})")

