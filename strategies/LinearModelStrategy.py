import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from strategies.strategy import Strategy
from market_data import Trade

class LinearModelStrategy(Strategy):
    def __init__(self, assets, start_date, end_date):
        self.assets = assets
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.price_history = {asset: [] for asset in self.assets}
        self.models = {asset: None for asset in self.assets}
        self.position = {asset: 0 for asset in self.assets}
        self.data_collected = {asset: pd.DataFrame(columns=['timestamp', 'price']) for asset in self.assets}
        self.data_threshold = 0.4  # 40% of the data
        self.total_seconds = int((self.end_date - self.start_date).total_seconds())

    def on_event(self, params, market_data,cash,assets_quantity):
        current_price = (market_data.best_ask + market_data.best_bid) / 2
        self.price_history[market_data.asset].append(current_price)
        self.data_collected[market_data.asset] = pd.concat([self.data_collected[market_data.asset], 
            pd.DataFrame({'timestamp': [market_data.timestamp], 'price': [current_price]})], ignore_index=True)

        # Check if we have collected enough data to build the model
        if len(self.price_history[market_data.asset]) >= self.total_seconds * self.data_threshold:
            if self.models[market_data.asset] is None:
                # Build the linear model
                df = self.data_collected[market_data.asset]
                df['log_return'] = np.log(df['price'] / df['price'].shift(1))
                df['x'] = df['log_return'].shift(1)
                df['y'] = df['log_return']
                df = df.dropna()    
                X = df['x'].values.reshape(-1, 1)
                y = df['y'].values
                model = LinearRegression(fit_intercept=False)
                model.fit(X, y)
                self.models[market_data.asset] = model

        # If the model is built, make predictions and trade
        if self.models[market_data.asset] is not None:
            current_price = (market_data.best_ask + market_data.best_bid) / 2
            previous_price = self.price_history[market_data.asset][-2] if len(self.price_history[market_data.asset]) > 1 else current_price
            log_return = np.log(current_price / previous_price)
            
            predicted_log_return = self.models[market_data.asset].predict([[log_return]])[0]

            if predicted_log_return > 0:
                if self.position[market_data.asset] in [0, -1]:
                    trade = Trade(market_data.timestamp, market_data.asset, 'buy', market_data.best_ask, cash/market_data.best_ask)
                    self.position[market_data.asset] += 1
                    return trade
            elif predicted_log_return < 0:
                if self.position[market_data.asset] in [0, 1]:
                    trade = Trade(market_data.timestamp, market_data.asset, 'sell', market_data.best_bid, assets_quantity)
                    self.position[market_data.asset] -= 1
                    return trade

        return None
