import asyncio
import os
import sys
from dotenv import load_dotenv

# Add project root to path
root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_path)

# Mock some things if needed, but let's try real first
from backend.services.chat_service import ChatService

async def debug_chat():
    load_dotenv('backend/.env')
    print("Initializing ChatService...")
    chat = ChatService()
    
    print("\nTesting message: 'Analyze Bitcoin'")
    try:
        # We'll use a timeout to see if it hangs
        response = await asyncio.wait_for(
            chat.process_message("Analyze Bitcoin"),
            timeout=30
        )
        print("\nResponse received:")
        print(response.get("response")[:200] + "...")
        print(f"Tools used: {response.get('tool_calls')}")
    except asyncio.TimeoutError:
        print("\n❌ ERROR: ChatService.process_message timed out after 30 seconds!")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")

if __name__ == "__main__":
    asyncio.run(debug_chat())
