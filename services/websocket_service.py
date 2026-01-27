"""
Binance WebSocket Service
Provides real-time cryptocurrency price streaming via WebSocket
"""

import asyncio
import json
import websockets
from typing import Dict, List, Set
from datetime import datetime
from collections import deque
import logging

logger = logging.getLogger(__name__)


class BinanceWebSocketService:
    """
    Manages WebSocket connections to Binance for real-time price data
    Broadcasts updates to connected frontend clients
    """
    
    def __init__(self, symbols: List[str] = None):
        """
        Initialize the WebSocket service
        
        Args:
            symbols: List of trading pairs to subscribe to (e.g., ['BTCUSDT', 'ETHUSDT'])
        """
        self.symbols = symbols or ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'DOGEUSDT', 'DOTUSDT']
        self.price_data: Dict[str, Dict] = {}
        self.price_history: Dict[str, deque] = {symbol: deque(maxlen=100) for symbol in self.symbols}
        self.connected_clients: Set = set()
        self.kline_subscribers: Dict = {}  # Track kline subscribers
        self.binance_ws = None
        self.running = False
        
        # Initialize price data structure
        for symbol in self.symbols:
            self.price_data[symbol] = {
                'symbol': symbol,
                'price': 0,
                'percent_change': 0,
                'high_24h': 0,
                'low_24h': 0,
                'volume_24h': 0,
                'last_update': None
            }
    
    async def start(self):
        """Start the WebSocket service"""
        self.running = True
        # Start the main connection task
        asyncio.create_task(self._management_loop())
    
    async def stop(self):
        """Stop the service"""
        self.running = False
        if self.binance_ws:
            await self.binance_ws.close()

    async def _management_loop(self):
        """Main loop that manages connection attempts and fallbacks"""
        ws_endpoints = [
            "wss://stream.binance.com:9443",
            "wss://stream.binance.us:9443"
        ]
        
        while self.running:
            connected = False
            
            # Try WebSockets first
            for endpoint in ws_endpoints:
                if not self.running: break
                
                streams = '/'.join([f"{symbol.lower()}@ticker" for symbol in self.symbols])
                url = f"{endpoint}/stream?streams={streams}"
                
                logger.info(f"Attempting Binance WebSocket: {endpoint}")
                try:
                    async with websockets.connect(url, timeout=10) as websocket:
                        self.binance_ws = websocket
                        logger.info(f"✅ Connected to Binance WebSocket: {endpoint}")
                        connected = True
                        
                        async for message in websocket:
                            if not self.running: break
                            await self._process_binance_message(message)
                            
                except (websockets.exceptions.ConnectionClosed, Exception) as e:
                    errMsg = str(e)
                    if "451" in errMsg:
                        logger.warning(f"Region blocked (451) for {endpoint}. Trying next fallback...")
                        continue # Try next WS endpoint
                    else:
                        logger.error(f"Binance WS error ({endpoint}): {e}")
                        break # Try next WS endpoint or fallback
            
            # If WebSockets failed/blocked, use Polling Fallback
            if self.running and not connected:
                logger.info("⚠️ All WebSockets failed or blocked. Entering Polling Fallback mode...")
                await self._run_polling_fallback()
                
            await asyncio.sleep(5)

    async def _run_polling_fallback(self):
        """Poll price data via REST API when WebSockets are blocked"""
        import requests
        
        # We'll use the REST endpoints that usually have different geographic rules
        poll_endpoints = [
            "https://api.binance.us/api/v3/ticker/24hr",
            "https://api.binance.com/api/v3/ticker/24hr"
        ]
        
        while self.running:
            updated = False
            for endpoint in poll_endpoints:
                try:
                    # Filter for our symbols to save bandwidth
                    response = requests.get(endpoint, timeout=5)
                    if response.status_code == 200:
                        data = response.json()
                        # Binance returns a list of tickers
                        ticker_map = {item['s'].upper(): item for item in data if item['s'].upper() in self.symbols}
                        
                        for symbol, ticker in ticker_map.items():
                            current_price = float(ticker['lastPrice']) if 'lastPrice' in ticker else float(ticker['c'])
                            percent_change = float(ticker['priceChangePercent']) if 'priceChangePercent' in ticker else float(ticker['P'])
                            
                            self.price_data[symbol] = {
                                'symbol': symbol,
                                'price': current_price,
                                'percent_change': percent_change,
                                'high_24h': float(ticker.get('highPrice', ticker.get('h', 0))),
                                'low_24h': float(ticker.get('lowPrice', ticker.get('l', 0))),
                                'volume_24h': float(ticker.get('volume', ticker.get('v', 0))),
                                'last_update': datetime.now().isoformat(),
                                'source': 'polling'
                            }
                            
                            self.price_history[symbol].append({
                                'time': datetime.now().isoformat(),
                                'price': current_price
                            })
                            
                            await self._broadcast_update(symbol)
                        
                        updated = True
                        break # Success with this endpoint
                    elif response.status_code == 451:
                        continue # Try next REST endpoint
                except Exception as e:
                    logger.error(f"Polling error for {endpoint}: {e}")
                    continue
            
            if updated:
                # Poll every 10 seconds in fallback mode
                # We also want to check if WebSockets might work again occasionally
                for _ in range(3): # Wait 30 seconds before trying WS again
                    if not self.running: return
                    await asyncio.sleep(10)
                break # Exit polling to try WebSockets in management loop
            else:
                await asyncio.sleep(10)
    
    async def _process_binance_message(self, message: str):
        """
        Process incoming message from Binance WebSocket
        
        Message format:
        {
            "stream": "btcusdt@ticker",
            "data": {
                "s": "BTCUSDT",
                "c": "50000.00",  // Current price
                "P": "2.5",       // Price change percent
                "h": "51000.00",  // High price
                "l": "49000.00",  // Low price
                "v": "1000.00"    // Volume
            }
        }
        """
        try:
            msg = json.loads(message)
            
            if 'data' not in msg:
                return
            
            data = msg['data']
            symbol = data['s']
            
            if symbol not in self.symbols:
                return
            
            # Update price data
            current_price = float(data['c'])
            percent_change = float(data['P'])
            
            self.price_data[symbol] = {
                'symbol': symbol,
                'price': current_price,
                'percent_change': percent_change,
                'high_24h': float(data['h']),
                'low_24h': float(data['l']),
                'volume_24h': float(data['v']),
                'last_update': datetime.now().isoformat()
            }
            
            # Add to price history for mini charts
            self.price_history[symbol].append({
                'time': datetime.now().isoformat(),
                'price': current_price
            })
            
            # Broadcast to all connected clients
            await self._broadcast_update(symbol)
            
        except Exception as e:
            logger.error(f"Error processing Binance message: {e}")
    
    async def _broadcast_update(self, symbol: str):
        """Broadcast price update to all connected clients"""
        if not self.connected_clients:
            return
        
        update_message = {
            'type': 'price_update',
            'symbol': symbol,
            'data': self.price_data[symbol],
            'history': list(self.price_history[symbol])[-20:]  # Last 20 points for mini chart
        }
        
        message_json = json.dumps(update_message)
        
        # Send to all connected clients
        disconnected_clients = set()
        for client in self.connected_clients:
            try:
                await client.send_text(message_json)
            except Exception as e:
                logger.error(f"Error sending to client: {e}")
                disconnected_clients.add(client)
        
        # Remove disconnected clients
        self.connected_clients -= disconnected_clients
    
    async def add_client(self, websocket):
        """Add a new client connection"""
        self.connected_clients.add(websocket)
        logger.info(f"Client connected. Total clients: {len(self.connected_clients)}")
        
        # Send initial data to new client
        for symbol in self.symbols:
            if self.price_data[symbol]['price'] > 0:
                initial_message = {
                    'type': 'price_update',
                    'symbol': symbol,
                    'data': self.price_data[symbol],
                    'history': list(self.price_history[symbol])[-20:]
                }
                try:
                    await websocket.send_text(json.dumps(initial_message))
                except Exception as e:
                    logger.error(f"Error sending initial data: {e}")
    
    async def remove_client(self, websocket):
        """Remove a client connection"""
        self.connected_clients.discard(websocket)
        logger.info(f"Client disconnected. Total clients: {len(self.connected_clients)}")
    
    def get_current_prices(self) -> Dict:
        """Get current price data for all symbols"""
        return self.price_data
    
    def get_price_history(self, symbol: str, limit: int = 100) -> List[Dict]:
        """Get price history for a specific symbol"""
        if symbol not in self.price_history:
            return []
        return list(self.price_history[symbol])[-limit:]
    
    async def add_kline_client(self, websocket, symbol: str, interval: str):
        """Add a client specifically for kline data"""
        key = f"{symbol}_{interval}"
        if key not in self.kline_subscribers:
            self.kline_subscribers[key] = set()
        self.kline_subscribers[key].add(websocket)
        logger.info(f"Added kline subscriber for {symbol} {interval}. Total: {len(self.kline_subscribers[key])}")
        
        # Start kline stream if not already active
        await self._start_kline_stream(symbol, interval)
    
    async def remove_kline_client(self, websocket, symbol: str, interval: str):
        """Remove a kline client"""
        key = f"{symbol}_{interval}"
        if key in self.kline_subscribers:
            self.kline_subscribers[key].discard(websocket)
            if len(self.kline_subscribers[key]) == 0:
                del self.kline_subscribers[key]
                logger.info(f"Removed kline stream for {symbol} {interval}")
    
    async def _start_kline_stream(self, symbol: str, interval: str):
        """Start a kline stream for a specific symbol and interval"""
        # Check if we already have a kline stream for this symbol/interval
        key = f"{symbol}_{interval}"
        if hasattr(self, f"_kline_ws_{key}"):
            return  # Already running
        
        # Start kline WebSocket connection
        await self._connect_to_binance_kline(symbol, interval)
    
    async def _connect_to_binance_kline(self, symbol: str, interval: str):
        """Connect to Binance WebSocket for kline data"""
        stream_name = f"{symbol.lower()}@kline_{interval}"
        url = f"wss://stream.binance.com:9443/ws/{stream_name}"
        
        logger.info(f"Connecting to Binance Kline WebSocket: {url}")
        
        key = f"{symbol}_{interval}"
        setattr(self, f"_kline_ws_{key}", True)  # Mark as running
        
        while self.running and key in self.kline_subscribers and len(self.kline_subscribers[key]) > 0:
            try:
                async with websockets.connect(url) as websocket:
                    logger.info(f"✅ Connected to Binance Kline WebSocket for {symbol} {interval}")
                    
                    async for message in websocket:
                        if not self.running or key not in self.kline_subscribers or len(self.kline_subscribers[key]) == 0:
                            break
                        
                        await self._process_kline_message(message, symbol, interval)
                        
            except websockets.exceptions.ConnectionClosed:
                logger.warning(f"Binance Kline WebSocket for {symbol} {interval} closed, reconnecting in 5s...")
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Binance Kline WebSocket error for {symbol} {interval}: {e}, reconnecting in 5s...")
                await asyncio.sleep(5)
        
        # Clean up when done
        if hasattr(self, f"_kline_ws_{key}"):
            delattr(self, f"_kline_ws_{key}")
    
    async def _process_kline_message(self, message: str, symbol: str, interval: str):
        """Process incoming kline message from Binance"""
        try:
            msg = json.loads(message)
            
            if 'k' not in msg:
                return
            
            kline = msg['k']
            
            # Format kline data to match frontend expectations
            kline_data = {
                't': kline['t'],  # Open time
                'T': kline['T'],  # Close time
                's': kline['s'],  # Symbol
                'i': kline['i'],  # Interval
                'f': kline['f'],  # First trade ID
                'L': kline['L'],  # Last trade ID
                'o': kline['o'],  # Open price
                'c': kline['c'],  # Close price
                'h': kline['h'],  # High price
                'l': kline['l'],  # Low price
                'v': kline['v'],  # Base asset volume
                'n': kline['n'],  # Number of trades
                'x': kline['x'],  # Is this kline closed?
                'q': kline['q'],  # Quote asset volume
                'V': kline['V'],  # Taker buy base asset volume
                'Q': kline['Q'],  # Taker buy quote asset volume
                'B': kline['B']   # Ignore
            }
            
            # Broadcast kline update to all subscribers of this symbol/interval
            await self._broadcast_kline_update(symbol, interval, kline_data)
            
        except Exception as e:
            logger.error(f"Error processing kline message: {e}")
    
    async def _broadcast_kline_update(self, symbol: str, interval: str, kline_data):
        """Broadcast kline update to all subscribed clients"""
        key = f"{symbol}_{interval}"
        if key not in self.kline_subscribers or not self.kline_subscribers[key]:
            return
        
        update_message = {
            'type': 'kline_update',
            'symbol': symbol,
            'interval': interval,
            'kline': kline_data
        }
        
        message_json = json.dumps(update_message)
        
        # Send to all subscribed clients
        disconnected_clients = set()
        for client in self.kline_subscribers[key]:
            try:
                await client.send_text(message_json)
            except Exception as e:
                logger.error(f"Error sending kline update to client: {e}")
                disconnected_clients.add(client)
        
        # Remove disconnected clients
        self.kline_subscribers[key] -= disconnected_clients