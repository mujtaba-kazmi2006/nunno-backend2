"""
Test OpenRouter API connection
"""
import os
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI

async def test_openrouter():
    load_dotenv(override=True)
    
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    model = os.getenv("AI_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
    
    print(f"Testing OpenRouter API...")
    print(f"API Key: {'✓ Found' if api_key else '✗ Missing'}")
    print(f"Model: {model}")
    print(f"API Key prefix: {api_key[:15]}..." if api_key else "No key")
    print()
    
    if not api_key:
        print("ERROR: No API key found in .env file")
        return
    
    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    
    try:
        print("Sending test request...")
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "Say 'Hello!' in one word"}
            ],
            max_tokens=10
        )
        
        print("✓ SUCCESS!")
        print(f"Response: {response.choices[0].message.content}")
        
    except Exception as e:
        print(f"✗ ERROR: {e}")
        print(f"Error type: {type(e).__name__}")
        
        # Try to get more details
        if hasattr(e, 'response'):
            print(f"Response status: {e.response.status_code if hasattr(e.response, 'status_code') else 'N/A'}")
            print(f"Response body: {e.response.text if hasattr(e.response, 'text') else 'N/A'}")

if __name__ == "__main__":
    asyncio.run(test_openrouter())
