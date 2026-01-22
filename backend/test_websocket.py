"""
Test script to verify WebSocket service functionality
"""
import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.websocket_service import BinanceWebSocketService


async def test_websocket_service():
    """Test the WebSocket service"""
    print("Testing Binance WebSocket Service...")
    
    # Initialize service
    service = BinanceWebSocketService(['BTCUSDT', 'ETHUSDT'])
    
    # Start service
    print("Starting WebSocket service...")
    asyncio.create_task(service.start())
    
    # Wait for connection and data
    print("Waiting for price data...")
    await asyncio.sleep(10)
    
    # Check if we received data
    prices = service.get_current_prices()
    
    print("\nCurrent Prices:")
    for symbol, data in prices.items():
        if data['price'] > 0:
            print(f"  {symbol}: ${data['price']:,.2f} ({data['percent_change']:+.2f}%)")
        else:
            print(f"  {symbol}: Waiting for data...")
    
    # Check history
    btc_history = service.get_price_history('BTCUSDT', limit=5)
    print(f"\nBTC Price History (last 5 points): {len(btc_history)} points")
    
    # Stop service
    print("\nStopping service...")
    await service.stop()
    
    print("Test complete!")


if __name__ == "__main__":
    asyncio.run(test_websocket_service())
