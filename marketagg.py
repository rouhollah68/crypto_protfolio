import asyncio
import os
os.chdir('C:/Users/rouho/Desktop/data/crypto_protfolio/class')
import websockets
import json
import threading
import pandas as pd
from datetime import datetime
from typing import List, Optional
from market_data import MarketData
import time
import websocket

class MarketAggregator:
    def __init__(self, symbols):
        self.symbols = [symbol.lower() for symbol in symbols]
        self.base_url = "wss://fstream.binance.com"
        self.trades_data = {symbol: pd.DataFrame() for symbol in self.symbols}
        self.orderbook_data = {symbol: pd.DataFrame() for symbol in self.symbols}
        self.quotes_data = {symbol: pd.DataFrame() for symbol in self.symbols}
        self.loop = None
        self.thread = None

    async def listen_trades(self, symbol):
        trade_url = f"{self.base_url}/{symbol}@trade"
        async with websockets.connect(trade_url) as websocket:
            while True:
                trade_data = await websocket.recv()
                trade = json.loads(trade_data)
                timestamp = datetime.now().replace(microsecond=0)
                new_trade = pd.DataFrame({'P': [float(trade['p'])], 'V': [float(trade['q'])]}, index=[timestamp])
                
                if not self.trades_data[symbol].empty and self.trades_data[symbol].index[-1] == timestamp:
                    self.trades_data[symbol].iloc[-1] = new_trade.iloc[0]
                else:
                    self.trades_data[symbol] = pd.concat([self.trades_data[symbol], new_trade])
                
    async def listen_orderbook(self, symbol):
        orderbook_url = f"{self.base_url}/{symbol}@depth"
        async with websockets.connect(orderbook_url) as websocket:
            while True:
                orderbook_data = await websocket.recv()
                orderbook = json.loads(orderbook_data)
                bids = orderbook['b'][:5]  # Take first 5 bids
                asks = orderbook['a'][:5]  # Take first 5 asks
                data = {
                    'BP_1': [float(bids[0][0])], 'BV_1': [float(bids[0][1])],
                    'BP_2': [float(bids[1][0])], 'BV_2': [float(bids[1][1])],
                    'BP_3': [float(bids[2][0])], 'BV_3': [float(bids[2][1])],
                    'BP_4': [float(bids[3][0])], 'BV_4': [float(bids[3][1])],
                    'BP_5': [float(bids[4][0])], 'BV_5': [float(bids[4][1])],
                    'AP_1': [float(asks[0][0])], 'AV_1': [float(asks[0][1])],
                    'AP_2': [float(asks[1][0])], 'AV_2': [float(asks[1][1])],
                    'AP_3': [float(asks[2][0])], 'AV_3': [float(asks[2][1])],
                    'AP_4': [float(asks[3][0])], 'AV_4': [float(asks[3][1])],
                    'AP_5': [float(asks[4][0])], 'AV_5': [float(asks[4][1])]
                }
                timestamp = datetime.now().replace(microsecond=0)
                df = pd.DataFrame(data, index=[timestamp])
                if not self.orderbook_data[symbol].empty and self.orderbook_data[symbol].index[-1] == timestamp:
                    self.orderbook_data[symbol].iloc[-1] = df.iloc[0]
                else:
                    self.orderbook_data[symbol] = pd.concat([self.orderbook_data[symbol], df])

    def listen_to_quotes(self):
        streams = "/".join([f"{symbol.lower()}@bookTicker" for symbol in self.symbols])
        quote_url = f"{self.base_url}/stream?streams={streams}"
        print(quote_url)
        time.sleep(1)
        def on_message(ws, message):
            quote_data = json.loads(message)['data']
            symbol = quote_data['s'].lower()
            bid = float(quote_data['b'])
            ask = float(quote_data['a'])
            event_time = datetime.fromtimestamp(quote_data['E'] / 1000)
            self.quotes_data[symbol] = pd.concat([self.quotes_data[symbol], pd.DataFrame({'B': [bid], 'A': [ask]}, index=[event_time])])
            # print(self.quotes_data[symbol])
            
        def on_open(ws):
            for symbol in self.symbols:
                print(f"Connection opened for {symbol.upper()}")

        def on_close(ws):
            for symbol in self.symbols:
                print(f"Connection closed for {symbol.upper()}")

        def on_error(ws, error):
            for symbol in self.symbols:
                print(f"Error for {symbol.upper()}: {error}")

        def on_ping(ws, message):
            for symbol in self.symbols:
                print(f"Ping received for {symbol.upper()}")
                ws.send(message, websocket.ABNF.OPCODE_PONG)
                print(f"Sent pong: {message}")

        def on_pong(ws, message):
            for symbol in self.symbols:
                print(f"Pong received for {symbol.upper()}")

        wsapp = websocket.WebSocketApp(
            quote_url,
            on_message=on_message,
            on_open=on_open,
            on_close=on_close,
            on_error=on_error,
            on_ping=on_ping,
            on_pong=on_pong
        )
        wsapp.run_forever(ping_interval=30, ping_timeout=10)

    def clean_data(self, df):
        df = df.resample('1s').agg(lambda x: x.sum() if 'V' in x.name else x.mean())
        if len(df) > 1000000:
            start_time = df.index[0]
            end_time = df.index[-1]
            df.iloc[-999900:].to_csv(f'{start_time}_{end_time}.csv')
            df = df.iloc[-100:]
        return df

    def start(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        tasks = []
        # for symbol in self.symbols:
        #     tasks.append(self.listen_trades(symbol))
        #     tasks.append(self.listen_orderbook(symbol))
        threading.Thread(target=self.listen_to_quotes).start()  # Run listen_to_quotes in a separate thread
        self.loop.run_until_complete(asyncio.wait(tasks))

    def start_in_thread(self):
        self.thread = threading.Thread(target=self.start)
        self.thread.start()
        return self.thread

    def stop(self):
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.loop.run_until_complete(self.loop.shutdown_asyncgens())
            self.loop.close()
        if self.thread:
            self.thread.join()

    def emit_market_data(self, symbol) -> Optional[MarketData]:
        row = self.quotes_data[symbol.lower()].iloc[-1]
        return MarketData(
            timestamp=row.name,
            best_bid=float(row['B']),
            best_bid_volume=0,
            best_ask=float(row['A']),
            best_ask_volume=0,
            asset=symbol.upper()
        )

if __name__ == '__main__':
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT']

    market_aggregator = MarketAggregator(symbols)
    # market_aggregator.listen_to_quotes()
    client_thread = market_aggregator.start_in_thread()
    time.sleep(5)
    def print_market_data(self):
        while True:
            time.sleep(1)
            for symbol in self.symbols:
                try:
                    print(self.emit_market_data(symbol))
                except Exception as e:
                    print(e)


    # Start the print_market_data method in a separate thread
    threading.Thread(target=print_market_data, args=(market_aggregator,), daemon=True).start()
