from strategies.strategy import Strategy
from market_data import Trade
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import itertools 

    
class MeanReversionStrategy(Strategy):
    def __init__(self, assets, lookback_period=600, sleep_period=600):
        self.assets = assets
        self.lookback_period = lookback_period
        self.sleep_period = sleep_period
        self.price_history = {asset: [] for asset in self.assets}
        self.moving_average = {asset: 0 for asset in self.assets}
        self.std_dev = {asset: 0 for asset in self.assets}
        self.upper_band = {asset: 0 for asset in self.assets}
        self.lower_band = {asset: 0 for asset in self.assets}
        self.position = {asset: 0 for asset in self.assets}
        self.num_std_dev = 3
        self.start_time = None

    def on_event(self, params, market_data,cash,assets_quantity):
        if self.start_time is None:
            self.start_time = market_data.timestamp

        # Calculate the time difference in minutes
        time_diff = (market_data.timestamp - self.start_time).total_seconds() / 60

        # Skip trading during the sleep period
        if time_diff < self.sleep_period:
            return None
        if len(self.price_history[market_data.asset])<2:
            self.price_history[market_data.asset].append((market_data.best_ask+market_data.best_bid)/2)
            return None
        self.price_history[market_data.asset].append((market_data.best_ask+market_data.best_bid)/2)
        self.moving_average[market_data.asset] = sum(self.price_history[market_data.asset]) / len(self.price_history[market_data.asset])
        self.std_dev[market_data.asset] = (sum((x - self.moving_average[market_data.asset])**2 for x in self.price_history[market_data.asset]) / len(self.price_history[market_data.asset]))**0.5
        self.upper_band[market_data.asset] = self.moving_average[market_data.asset] + self.num_std_dev * self.std_dev[market_data.asset]
        self.lower_band[market_data.asset] = self.moving_average[market_data.asset] - self.num_std_dev * self.std_dev[market_data.asset]
        
        if len(self.price_history[market_data.asset]) > self.lookback_period:
            self.price_history[market_data.asset].pop(0)
        
        if len(self.price_history[market_data.asset]) == self.lookback_period:
            current_price = (market_data.best_ask+market_data.best_bid)/2
            
            if current_price < self.lower_band[market_data.asset]: 
                if self.position[market_data.asset] in [0, -1]:
                    trade = Trade(market_data.timestamp, market_data.asset, 'buy', market_data.best_ask, cash/market_data.best_ask)
                    self.position[market_data.asset] += 1
                    return trade
            elif current_price > self.upper_band[market_data.asset]:
                if self.position[market_data.asset] in [0, 1]:
                    trade = Trade(market_data.timestamp, market_data.asset, 'sell', market_data.best_bid, assets_quantity)
                    self.position[market_data.asset] -= 1
                    return trade
        
        return None
   

class AskBidMeanReversionStrategy(Strategy):
    def __init__(self, assets, lookback_period=600, sleep_period=600):
        self.assets = assets
        self.lookback_period = lookback_period
        self.sleep_period = sleep_period
        self.price_history = {asset: [] for asset in self.assets}
        self.bid_history = {asset: [] for asset in self.assets}
        self.ask_history = {asset: [] for asset in self.assets}
        self.moving_price_average = {asset: 0 for asset in self.assets}
        self.moving_bid_average = {asset: 0 for asset in self.assets}
        self.moving_ask_average = {asset: 0 for asset in self.assets}
        self.std_dev = {asset: 0 for asset in self.assets}
        self.upper_band = {asset: 0 for asset in self.assets}
        self.lower_band = {asset: 0 for asset in self.assets}
        self.position = {asset: 0 for asset in self.assets}
        self.num_std_dev = 2.5
        self.start_time = None

    def on_event(self, params, market_data, pnl, assets_quantity):
        if self.start_time is None:
            self.start_time = market_data.timestamp

        # Calculate the time difference in minutes
        time_diff = (market_data.timestamp - self.start_time).total_seconds() / 60

        # Skip trading during the sleep period
        if time_diff < self.sleep_period:
            return None

        self.price_history[market_data.asset].append((market_data.best_ask+market_data.best_bid)/2)
        self.bid_history[market_data.asset].append(market_data.best_bid)
        self.ask_history[market_data.asset].append(market_data.best_ask)

        if len(self.price_history[market_data.asset]) > self.lookback_period:
            self.price_history[market_data.asset].pop(0)
            self.bid_history[market_data.asset].pop(0)
            self.ask_history[market_data.asset].pop(0)

        self.moving_price_average[market_data.asset] = sum(self.price_history[market_data.asset]) / len(self.price_history[market_data.asset])
        self.moving_bid_average[market_data.asset] = sum(self.bid_history[market_data.asset]) / len(self.bid_history[market_data.asset])
        self.moving_ask_average[market_data.asset] = sum(self.ask_history[market_data.asset]) / len(self.ask_history[market_data.asset])

        if len(self.price_history[market_data.asset]) > 1:
            self.std_dev[market_data.asset] = (sum((x - self.moving_price_average[market_data.asset])**2 for x in self.price_history[market_data.asset]) / (len(self.price_history[market_data.asset]) - 1))**0.5
        self.upper_band[market_data.asset] = self.moving_ask_average[market_data.asset] + self.num_std_dev * self.std_dev[market_data.asset]
        self.lower_band[market_data.asset] = self.moving_bid_average[market_data.asset] - self.num_std_dev * self.std_dev[market_data.asset]
        
        
        
        if len(self.price_history[market_data.asset]) == self.lookback_period:
            if self.ask_history[market_data.asset][-1] < self.lower_band[market_data.asset]:
                if self.position[market_data.asset] in [0, -1]:
                    trade = Trade(market_data.timestamp, market_data.asset, 'buy', market_data.best_ask, pnl/market_data.best_ask - assets_quantity[market_data.asset])
                    self.position[market_data.asset] = 1
                    return {market_data.asset: trade}
            elif self.bid_history[market_data.asset][-1] > self.moving_price_average[market_data.asset]:
                if self.position[market_data.asset] == 1:
                    trade = Trade(market_data.timestamp, market_data.asset, 'sell', market_data.best_bid, assets_quantity[market_data.asset])
                    self.position[market_data.asset] = 0
                    return {market_data.asset: trade}
            elif self.bid_history[market_data.asset][-1] > self.upper_band[market_data.asset]:
                if self.position[market_data.asset] in [0, 1]:
                    trade = Trade(market_data.timestamp, market_data.asset, 'sell', market_data.best_bid, assets_quantity[market_data.asset] + pnl/market_data.best_bid)
                    self.position[market_data.asset] = -1
                    return {market_data.asset: trade}
            elif self.ask_history[market_data.asset][-1] < self.moving_price_average[market_data.asset]:
                if self.position[market_data.asset] == -1:
                    trade = Trade(market_data.timestamp, market_data.asset, 'buy', market_data.best_ask, -assets_quantity[market_data.asset])
                    self.position[market_data.asset] = 0
                    return {market_data.asset: trade}
        
        return None
    
        
class PortfolioMeanReversionStrategy(MeanReversionStrategy):
    def __init__(self, assets, lookback_period=60, train_period=180):
        super().__init__(assets, lookback_period)
        self.weights = {asset: 0 for asset in assets}
        self.portfolio_value = np.zeros(self.lookback_period)
        self.portfolio_moving_average = 0
        self.portfolio_std_dev = 0
        self.portfolio_upper_band = 0
        self.portfolio_lower_band = 0
        self.counter = 0
        self.ask_history = {asset: [] for asset in self.assets}
        self.bid_history = {asset: [] for asset in self.assets}
        self.train_period = train_period
        self.portfolio_position = 0
        self.num_std_dev = 3
   
    def on_event(self, params, market_data, pnl, assets_quantity):
       # print(market_data.timestamp)
        if self.start_time is None:
            self.start_time = market_data.timestamp

        self.price_history[market_data.asset].append((market_data.best_ask + market_data.best_bid) / 2)
        self.ask_history[market_data.asset].append(market_data.best_ask)
        self.bid_history[market_data.asset].append(market_data.best_bid)
        
        if len(self.price_history[market_data.asset]) > self.train_period:
            self.price_history[market_data.asset].pop(0)
            self.ask_history[market_data.asset].pop(0)
            self.bid_history[market_data.asset].pop(0)
        self.counter+=1
        
        if len(self.price_history[market_data.asset]) >= self.train_period and self.counter % len(self.assets) == 0:
            if all(len(self.price_history[asset]) >= self.lookback_period for asset in self.assets):
                self.weights = self.calculate_portfolio_weights()

                self.portfolio_value = [sum(self.weights[asset] * self.price_history[asset][-i] for asset in self.assets) for i in range(self.lookback_period, 0, -1)]
                self.portfolio_value_ask = [sum(self.weights[asset] * (self.ask_history[asset][-i] if self.weights[asset] > 0 else self.bid_history[asset][-i]) for asset in self.assets) for i in range(self.lookback_period, 0, -1)]
                self.portfolio_value_bid = [sum(self.weights[asset] * (self.bid_history[asset][-i] if self.weights[asset] > 0 else self.ask_history[asset][-i]) for asset in self.assets) for i in range(self.lookback_period, 0, -1)]
                
                self.portfolio_moving_average = sum(self.portfolio_value) / len(self.portfolio_value)
                if len(self.portfolio_value) > 1:
                    self.portfolio_std_dev = (sum((x - self.portfolio_moving_average)**2 for x in self.portfolio_value) / (len(self.portfolio_value)-1))**0.5
                self.portfolio_upper_band = self.portfolio_moving_average + self.num_std_dev * self.portfolio_std_dev
                self.portfolio_lower_band = self.portfolio_moving_average - self.num_std_dev * self.portfolio_std_dev

                current_price = self.portfolio_value[-1]
                trades = {i: None for i in self.assets}
                # print(f'current_price: {current_price}')
                # print(f'portfolio_lower_band: {self.portfolio_lower_band}')
                # print(f'portfolio_upper_band: {self.portfolio_upper_band}')

                if current_price < self.portfolio_lower_band:
                    print('buy', self.portfolio_value_ask[-1])
                    for i in self.assets:
                        if self.weights[i] * pnl / self.portfolio_value_ask[-1] > assets_quantity[i]:
                            trades[i] = Trade(market_data.timestamp, i, 'buy', self.ask_history[i][-1], self.weights[i] * pnl / self.portfolio_value_ask[-1] - assets_quantity[i])
                        elif self.weights[i] * pnl / self.portfolio_value_ask[-1] < assets_quantity[i]:
                            trades[i] = Trade(market_data.timestamp, i, 'sell', self.bid_history[i][-1], -self.weights[i] * pnl / self.portfolio_value_ask[-1] + assets_quantity[i])
                elif current_price > self.portfolio_upper_band:
                    print('sell', self.portfolio_value_bid[-1])
                    for i in self.assets:
                        if -self.weights[i] * pnl / self.portfolio_value_bid[-1] < assets_quantity[i]:
                            trades[i] = Trade(market_data.timestamp, i, 'sell', self.bid_history[i][-1], self.weights[i] * pnl / self.portfolio_value_bid[-1] + assets_quantity[i])
                        elif -self.weights[i] * pnl / self.portfolio_value_bid[-1] > assets_quantity[i]:
                            trades[i] = Trade(market_data.timestamp, i, 'buy', self.ask_history[i][-1], -self.weights[i] * pnl / self.portfolio_value_bid[-1] - assets_quantity[i])
                return trades
        return None
    
    def calculate_portfolio_weights(self, method=4):
        price_df = pd.DataFrame(self.price_history)
        
        if method == 1:
            rolling_mean_price = price_df.rolling(window=self.lookback_period).mean()
            rolling_std_price = price_df.rolling(window=self.lookback_period).std()
            rolling_normalized_price = (price_df - rolling_mean_price) / rolling_std_price
            rolling_normalized_price = rolling_normalized_price.dropna()
            cov_matrix = rolling_normalized_price.cov().values
            
        elif method == 2:
            rolling_covs = np.zeros((self.train_period-self.lookback_period, len(self.assets), len(self.assets)))
            for i in range(self.lookback_period, self.train_period):
                window_data = price_df[i-self.lookback_period:i]
                cov_matrix = window_data.cov().values
                rolling_covs[i-self.lookback_period] = cov_matrix

            average_covariance = np.mean(rolling_covs, axis=0)
            cov_matrix = average_covariance / np.sum(average_covariance)
        
        elif method == 3:
            cov_matrix = price_df.cov().values
            cov_matrix = cov_matrix / np.sum(cov_matrix)

        elif method == 4:
            rolling_mean_price = price_df.rolling(window=self.lookback_period).mean()
            rolling_normalized_price = (price_df - rolling_mean_price)
            rolling_normalized_price = rolling_normalized_price.dropna()
            rolling_covs = np.zeros((rolling_normalized_price.shape[0], len(self.assets), len(self.assets)))
            for i in range(rolling_normalized_price.shape[0]):
                row = rolling_normalized_price.iloc[i].values
                rolling_covs[i] = np.outer(row, row)
            cov_matrix = np.mean(rolling_covs, axis=0)
            
        initial_price = [self.price_history[asset][-1] for asset in self.assets]
        w = self.optimize_weights(cov_matrix, np.array(initial_price))
        weights = {self.assets[i]: w[i] for i in range(len(self.assets))}
        return weights
    
    def optimize_weights(self,C, a):
        n = C.shape[0]
        min_value = float('inf')
        min_row = None
        C_pinv = np.linalg.pinv(C)
        for i in range(1, n+1):
            for indices in itertools.combinations(range(n), i):
                b = a.copy()
                for index in indices:
                    b[index] = -a[index]

                w = (C_pinv @ b) / (b.T @ C_pinv @ b)
                for index in indices:
                    w[index] = -w[index]
                value = w.T @ C @ w
                if value < min_value:
                    min_value = value
                    min_row = w

        w = (C_pinv @ a) / (a.T @ C_pinv @ a)
        value = w.T @ C @ w
        if value < min_value:
            min_value = value
            min_row = w
            
        return min_row 

class PortfolioAskBidMeanReversionStrategy(PortfolioMeanReversionStrategy):
    def __init__(self, assets, lookback_period=600, train_period=18000):
        super().__init__(assets, lookback_period)
        self.weights = {asset: 0 for asset in assets}
        self.portfolio_value = np.zeros(self.lookback_period)
        self.portfolio_value_ask = np.zeros(self.lookback_period)
        self.portfolio_value_bid = np.zeros(self.lookback_period)
        self.portfolio_moving_average = 0
        self.portfolio_std_dev = 0
        self.portfolio_upper_band = 0
        self.portfolio_lower_band = 0
        self.counter = 0
        self.ask_history = {asset: [] for asset in self.assets}
        self.bid_history = {asset: [] for asset in self.assets}
        self.train_period = train_period
        self.num_std_dev = 2.5
   
    def on_event(self, params, market_data, pnl, assets_quantity):
        if self.start_time is None:
            self.start_time = market_data.timestamp

        self.price_history[market_data.asset].append((market_data.best_ask + market_data.best_bid) / 2)
        self.ask_history[market_data.asset].append(market_data.best_ask)
        self.bid_history[market_data.asset].append(market_data.best_bid)
        
        if len(self.price_history[market_data.asset]) > self.train_period:
            self.price_history[market_data.asset].pop(0)
            self.ask_history[market_data.asset].pop(0)
            self.bid_history[market_data.asset].pop(0)
        self.counter+=1

        if len(self.price_history[market_data.asset]) == self.train_period and self.counter%len(self.assets)==0 and market_data.timestamp.time() > pd.Timestamp('09:10').time():
            self.weights = self.calculate_portfolio_weights(method=3)
            # if market_data.timestamp.date() != self.start_time.date():
            #     self.weights = self.calculate_portfolio_weights(method=3)
            #     self.start_time = market_data.timestamp
            
            self.portfolio_value = [sum(self.weights[asset] * self.price_history[asset][-i] for asset in self.assets) for i in range(self.lookback_period, 0, -1)]
            self.portfolio_value_ask = [sum(self.weights[asset] * (self.ask_history[asset][-i] if self.weights[asset] > 0 else self.bid_history[asset][-i]) for asset in self.assets) for i in range(self.lookback_period, 0, -1)]
            self.portfolio_value_bid = [sum(self.weights[asset] * (self.bid_history[asset][-i] if self.weights[asset] > 0 else self.ask_history[asset][-i]) for asset in self.assets) for i in range(self.lookback_period, 0, -1)]
            
            self.portfolio_moving_average = sum(self.portfolio_value) / len(self.portfolio_value)
            self.portfolio_moving_average_ask = sum(self.portfolio_value_ask) / len(self.portfolio_value_ask)
            self.portfolio_moving_average_bid = sum(self.portfolio_value_bid) / len(self.portfolio_value_bid)

            if len(self.portfolio_value) > 1:
                self.portfolio_std_dev = (sum((x - self.portfolio_moving_average)**2 for x in self.portfolio_value) / (len(self.portfolio_value)-1))**0.5
            self.portfolio_upper_band = self.portfolio_moving_average_ask + self.num_std_dev * self.portfolio_std_dev
            self.portfolio_lower_band = self.portfolio_moving_average_bid - self.num_std_dev * self.portfolio_std_dev
            
            # current_price = self.portfolio_value[-1]
            trades={i:None for i in self.assets}
            for i in self.assets:
                if self.portfolio_value_ask[-1] < self.portfolio_lower_band:
                    if self.weights[i] * pnl/self.portfolio_value_ask[-1] > assets_quantity[i]:
                        trades[i] = Trade(market_data.timestamp, i, 'buy', self.ask_history[i][-1], self.weights[i] * pnl/self.portfolio_value_ask[-1] - assets_quantity[i])
                    elif self.weights[i] * pnl/self.portfolio_value_ask[-1] < assets_quantity[i]:
                        trades[i] = Trade(market_data.timestamp, i, 'sell', self.bid_history[i][-1], -self.weights[i] * pnl/self.portfolio_value_ask[-1] + assets_quantity[i])
                elif self.portfolio_value_bid[-1] > self.portfolio_upper_band:
                    if -self.weights[i] * pnl/self.portfolio_value_bid[-1]  < assets_quantity[i]:
                        trades[i] = Trade(market_data.timestamp, i, 'sell', self.bid_history[i][-1], self.weights[i] * pnl/self.portfolio_value_bid[-1] + assets_quantity[i])
                    elif -self.weights[i] * pnl/self.portfolio_value_bid[-1] > assets_quantity[i]:
                        trades[i] = Trade(market_data.timestamp, i, 'buy', self.ask_history[i][-1], -self.weights[i] * pnl/self.portfolio_value_bid[-1] - assets_quantity[i])
            return trades
        return None
    
    
