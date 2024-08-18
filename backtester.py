import pandas as pd
import os
os.chdir('C:/Users/rouho/Desktop/data/crypto_protfolio/class')
from typing import Any, List, Optional
from market_data import MarketData, Trade
from marketagg import MarketAggregator
from strategies.strategy import Strategy
from strategies.meanreversion import AskBidMeanReversionStrategy, MeanReversionStrategy, PortfolioMeanReversionStrategy, PortfolioAskBidMeanReversionStrategy
import matplotlib.pyplot as plt
import time
from datetime import datetime
import socket
from client_test import Client
import threading

class Backtester:
    def __init__(self, assets: List[str], strategy: 'Strategy', cash: float, fee: float, market_aggregator):
        self.assets = assets
        self.cash = cash
        self.fee = fee
        self.strategy = strategy
        self.assets_quantity = {asset: 0 for asset in assets}
        self.trades = []
        self.pnls = {'timestamp': [], 'value': []}
        self.current_asset_ask_prices = {asset: 0 for asset in assets}
        self.current_asset_bid_prices = {asset: 0 for asset in assets}
        self.market_aggregator = market_aggregator
        time.sleep(2)
        self.assets_counter = 0


    def run(self):
        print('backtesting started')
        while True:
            try:
                # market_data = self.market_aggregator.emit_market_data(self.assets[self.assets_counter])
                time.sleep(1)
                market_data = self.market_aggregator.emit_market_data(self.assets[self.assets_counter])
                # print(market_data)
                self.assets_counter += 1
                if self.assets_counter == len(self.assets):
                    self.assets_counter = 0
            except:
                print("-----------------")
                continue
            if market_data is None:
                    break

            total_portfolio_value = self.cash
            for asset in self.assets:
                if self.assets_quantity[asset] > 0:
                    total_portfolio_value += self.assets_quantity[asset] * self.current_asset_bid_prices[asset]
                elif self.assets_quantity[asset] < 0:
                    total_portfolio_value += self.assets_quantity[asset] * self.current_asset_ask_prices[asset]

            self.pnls['value'].append(total_portfolio_value)
            self.pnls['timestamp'].append(datetime.now())
            trades = self.strategy.on_event(self.assets, market_data, self.pnls['value'][-1], self.assets_quantity)
            self.current_asset_ask_prices[market_data.asset] = market_data.best_ask
            self.current_asset_bid_prices[market_data.asset] = market_data.best_bid
            if trades and all(trade is not None for trade in trades.values()):
                for trade in trades.values():
                    trade_df = pd.DataFrame({
                        'timestamp': [trade.timestamp],
                        'asset': [trade.asset],
                        'trade_type': [trade.trade_type],
                        'price': [trade.price],
                        'volume': [trade.volume]
                    })
                    self.trades = pd.concat([self.trades, trade_df], ignore_index=True) if isinstance(self.trades, pd.DataFrame) else trade_df
                    print(trade)
                    if trade.trade_type == 'buy':
                        self.assets_quantity[trade.asset] += trade.volume
                        self.cash -= trade.price * trade.volume
                        self.cash -= self.cash*self.fee
                    elif trade.trade_type == 'sell':
                        self.assets_quantity[trade.asset] -= trade.volume
                        self.cash += trade.price * trade.volume
                        self.cash -= self.cash*self.fee
                    print(f"pnl: {self.pnls['value'][-1]}")

        self.pnls = pd.DataFrame(self.pnls)
        print(self.pnls)
        self.plot_pnls()

    def plot_pnls(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.pnls['timestamp'], self.pnls['value'])
        plt.title('Portfolio Value Over Time')
        plt.xlabel('Timestamp')
        plt.ylabel('Portfolio Value')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('portfolio_value_over_time.png')
        plt.close()

if __name__ == "__main__":
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT']
    market_aggregator = Client(symbols)
    time.sleep(20)
    # market_aggregator.start_client()
    client_thread = threading.Thread(target=market_aggregator.start_client)
    client_thread.start()

    strategy = PortfolioMeanReversionStrategy(assets=symbols, train_period=100)
    backtester = Backtester(symbols, strategy, 1000000, 0.0002, market_aggregator)
    backtester.run()
    print('backtesting finished')
    pnls = backtester.pnls