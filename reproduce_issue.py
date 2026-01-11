import asyncio
import os
import sys

# Ensure backend directory is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.chat_service import ChatService

# Mock environment variables if needed
os.environ["OPENROUTER_API_KEY"] = "mock_key" 

async def test_intents():
    # Mocking dependencies to avoid import errors if services are missing
    # But verify technical analysis service imports might print errors, we can ignore them.
    print("Initializing ChatService...")
    try:
        chat_service = ChatService()
    except Exception as e:
        print(f"Error initializing ChatService: {e}")
        return
    
    test_cases = [
        "Hello",
        "How are you?",
        "What is Bitcoin?",
        "What is your name?",
        "Tell me about yourself",
        "Is the market good?",
        "Price of BTC",
        "Analysis of ETH",
        "What can you do?",
        "Help",
        "Explain jargon"
    ]
    
    print(f"\n{'Message':<30} | {'Fallback Result':<50} | {'Needs Intent LLM?'}")
    print("-" * 100)
    
    for msg in test_cases:
        # We access the internal fallback method directly
        try:
            tools = chat_service._detect_tools_needed_fallback(msg)
            needs_llm = any(t[0] == "_needs_llm" for t in tools)
            print(f"{msg:<30} | {str(tools)[:50]:<50} | {needs_llm}")
        except Exception as e:
             print(f"{msg:<30} | Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_intents())
