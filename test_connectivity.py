import asyncio
import websockets
import requests
import json
import time

async def test_websocket(name, url):
    print(f"Testing {name} WebSocket: {url}")
    try:
        async with websockets.connect(url, timeout=10) as ws:
            print(f"✅ {name} WebSocket: Success!")
            return True
    except Exception as e:
        print(f"❌ {name} WebSocket: Failed - {e}")
        return False

def test_rest(name, url):
    print(f"Testing {name} REST: {url}")
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            print(f"✅ {name} REST: Success!")
            return True
        else:
            print(f"❌ {name} REST: Failed - HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ {name} REST: Failed - {e}")
        return False

async def main():
    print("--- Connectivity Test ---")
    
    # Test Binance.com
    await test_websocket("Binance COM", "wss://stream.binance.com:9443/ws/btcusdt@ticker")
    test_rest("Binance COM", "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT")
    
    # Test Binance.us
    await test_websocket("Binance US", "wss://stream.binance.us:9443/ws/btcusdt@ticker")
    test_rest("Binance US", "https://api.binance.us/api/v3/ticker/price?symbol=BTCUSDT")
    
    # Test CoinGecko
    test_rest("CoinGecko", "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd")
    
    print("--- End Test ---")

if __name__ == "__main__":
    asyncio.run(main())
