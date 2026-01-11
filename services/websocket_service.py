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
        """Start the WebSocket connection to Binance"""
        self.running = True
        await self._connect_to_binance()
    
    async def stop(self):
        """Stop the WebSocket connection"""
        self.running = False
        if self.binance_ws:
            await self.binance_ws.close()
    
    async def _connect_to_binance(self):
        """Connect to Binance WebSocket and subscribe to ticker streams"""
        # Build stream URL for multiple symbols
        # Format: wss://stream.binance.com:9443/stream?streams=btcusdt@ticker/ethusdt@ticker
        streams = '/'.join([f"{symbol.lower()}@ticker" for symbol in self.symbols])
        url = f"wss://stream.binance.com:9443/stream?streams={streams}"
        
        logger.info(f"Connecting to Binance WebSocket: {url}")
        
        while self.running:
            try:
                async with websockets.connect(url) as websocket:
                    self.binance_ws = websocket
                    logger.info("✅ Connected to Binance WebSocket")
                    
                    async for message in websocket:
                        if not self.running:
                            break
                        
                        await self._process_binance_message(message)
                        
            except websockets.exceptions.ConnectionClosed:
                logger.warning("Binance WebSocket connection closed, reconnecting in 5s...")
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Binance WebSocket error: {e}, reconnecting in 5s...")
                await asyncio.sleep(5)
    
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
