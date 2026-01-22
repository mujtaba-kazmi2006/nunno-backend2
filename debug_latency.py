
import asyncio
import time
import os
import sys
from dotenv import load_dotenv

# Load env variables
load_dotenv()

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), 'backend'))

from backend.services.chat_service import ChatService

async def test_latency():
    print("--- Starting Latency Test ---")
    service = ChatService()
    
    # Test Query 1: Simple Chat (No Tools)
    print("\n[Test 1] Simple Greeting: 'Hello'")
    start = time.time()
    async for chunk in service.stream_message("Hello"):
        if "data: " in chunk:
            print(f"Received chunk at {time.time() - start:.3f}s: {chunk[:50]}...")
            break # Just need first token time
    
    # Test Query 2: Technical Analysis (Binance)
    print("\n[Test 2] Tech Analysis: 'Analyze BTC price'")
    start = time.time()
    first_token_received = False
    gathering_start = False
    
    async for chunk in service.stream_message("Analyze BTC price"):
        elapsed = time.time() - start
        if "status" in chunk and not gathering_start:
            print(f"Status update at {elapsed:.3f}s")
            gathering_start = True
        if "type': 'text'" in chunk and not first_token_received:
            print(f"FIRST TEXT TOKEN at {elapsed:.3f}s")
            first_token_received = True
            break # Stop after first token
            
    # Test Query 3: News/Web (Potential Bottleneck)
    print("\n[Test 3] News: 'What is the news on Bitcoin?'")
    start = time.time()
    first_token_received = False
    
    async for chunk in service.stream_message("What is the news on Bitcoin?"):
        elapsed = time.time() - start
        if "type': 'text'" in chunk and not first_token_received:
            print(f"FIRST TEXT TOKEN at {elapsed:.3f}s")
            first_token_received = True
            break

if __name__ == "__main__":
    asyncio.run(test_latency())
