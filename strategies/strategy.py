from market_data import MarketData, Trade
from typing import Any, Optional
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

class Strategy:
    def on_event(self, params: Any, market_data: MarketData) -> Optional[Any]:
        raise NotImplementedError("Subclass must implement this method")