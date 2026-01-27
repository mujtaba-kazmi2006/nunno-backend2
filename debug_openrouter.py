
import os
import requests
import json
from dotenv import load_dotenv

# Load .env
load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
model = os.getenv("AI_MODEL", "openai/gpt-4o-mini")

print(f"DEBUG: Checking OpenRouter key...")
print(f"DEBUG: Model selected: {model}")

if not api_key:
    print("❌ ERROR: OPENROUTER_API_KEY not found in environment variables.")
    exit(1)

# Mask key for display
masked_key = f"{api_key[:6]}...{api_key[-4:]}" if len(api_key) > 10 else "Invalid Key"
print(f"DEBUG: Using key: {masked_key}")

try:
    # 1. Check Key Status
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    response = requests.get("https://openrouter.ai/api/v1/auth/key", headers=headers)
    
    if response.status_code == 200:
        data = response.json().get("data", {})
        print(f"✅ Key Status: Valid")
        print(f"   Label: {data.get('label')}")
        # Note: OpenRouter API might not show exact credit balance here, but success means key is good.
        # It usually shows 'usage' (spent) and 'limit' (allowance), not remaining balance.
    else:
        print(f"❌ Key Check Failed: {response.status_code} - {response.text}")

    # 2. Test Model (Dry Run with 1 token)
    print(f"\nDEBUG: Testing simple chat request with {model}...")
    
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1
    }
    
    chat_response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload
    )
    
    if chat_response.status_code == 200:
        print(f"✅ Chat Request: Success!")
    else:
        print(f"❌ Chat Request Failed: {chat_response.status_code}")
        print(f"   Response: {chat_response.text}")
        if chat_response.status_code == 402:
            print("\n⚠️  Error 402 confirm: Insufficient credits or payment required.")

except Exception as e:
    print(f"❌ Exception: {str(e)}")
