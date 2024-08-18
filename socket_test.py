import json
import time
import socket
import websocket
from datetime import datetime
class MarketAggregator:
    def __init__(self, symbols):
        self.symbols = [symbol.lower() for symbol in symbols]
        self.base_url = "wss://fstream.binance.com"
        self.server_socket = None
        self.conn = None

    def start_server(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind(('localhost', 54321))  # Use a higher port number
        self.server_socket.listen(1)
        print("Server started and listening on port 54321")

        self.conn, addr = self.server_socket.accept()
        print(f"Connection from {addr}")

    def send_data_to_client(self, data):
        if self.conn:
            self.conn.send(data.encode())

    def listen_to_quotes(self):
        self.start_server()
        
        streams = "/".join([f"{symbol.lower()}@bookTicker" for symbol in self.symbols])
        quote_url = f"{self.base_url}/stream?streams={streams}"
        print(quote_url)
        time.sleep(1)

        def on_message(ws, message):
            quote_data = json.loads(message)['data']
            symbol = quote_data['s'].upper()
            bid = float(quote_data['b'])
            ask = float(quote_data['a'])
            event_time = datetime.fromtimestamp(quote_data['E'] / 1000)
            data = {'symbol': symbol, 'bid': bid, 'ask': ask, 'timestamp': event_time.isoformat()}
            self.send_data_to_client(json.dumps(data))

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

        self.conn.close()
        self.server_socket.close()

if __name__ == "__main__":
    symbols = ['btcusdt', 'ethusdt', 'bnbusdt', 'solusdt']
    aggregator = MarketAggregator(symbols)
    aggregator.listen_to_quotes()