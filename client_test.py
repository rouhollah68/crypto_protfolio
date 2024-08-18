import socket
import pandas as pd
import json
from datetime import datetime
import time
from market_data import MarketData, Trade
class Client:
    def __init__(self, symbols):
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect(('localhost', 54321))
        print("Connected to server")
        self.symbols = symbols
        self.quote_data = {symbol: pd.DataFrame(columns=['timestamp', 'bid', 'ask']) for symbol in symbols}

    def start_client(self):
        while True:
            data = self.client_socket.recv(1024).decode()
            # print(data)
            if not data:
                break
            for json_data in data.split('\n'):
                if json_data.strip():
                    try:
                        data = json.loads(json_data)
                        # time.sleep(0.3)
                        symbol = data['symbol']
                        timestamp = datetime.fromisoformat(data['timestamp'])
                        new_data = pd.DataFrame({
                            'timestamp': [timestamp],
                            'bid': [data['bid']],
                            'ask': [data['ask']]
                        })                        
                        self.quote_data[symbol] = pd.concat([self.quote_data[symbol], new_data], ignore_index=True)
                    except json.JSONDecodeError as e:
                        continue

        self.client_socket.close()
    def emit_market_data(self,symbol):
        try:
            return MarketData(
                timestamp=self.quote_data[symbol].iloc[-1]['timestamp'],
            best_bid=float(self.quote_data[symbol].iloc[-1]['bid']),
            best_bid_volume=0,
            best_ask=float(self.quote_data[symbol].iloc[-1]['ask']),
            best_ask_volume=0,
                asset=symbol.upper()
            )
        except:
            return None

if __name__ == "__main__":
    # symbols = ['btcusdt', 'ethusdt']  # Add your symbols here
    symbols = ['btcusdt', 'ethusdt', 'bnbusdt', 'solusdt']
    client = Client(symbols)
    import threading
    client_thread = threading.Thread(target=client.start_client)
    client_thread.start()
    while True:
        time.sleep(1)
        # print(client.quote_data)
        for i in client.quote_data.keys():
            try:    
                print(client.emit_market_data(i))
            except:
                continue

            

