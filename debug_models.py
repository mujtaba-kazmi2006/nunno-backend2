
import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
headers = {
    "Authorization": f"Bearer {api_key}",
    "HTTP-Referer": "https://nunno.finance",
    "X-Title": "Nunno Finance",
    "Content-Type": "application/json"
}

models_to_test = [
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "deepseek/deepseek-r1-0528:free"
]

print(f"Testing OpenRouter models with key: {api_key[:5]}...")

for model in models_to_test:
    print(f"\nTesting model: {model}")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Hello, are you working?"}],
        "max_tokens": 50
    }
    
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=15
        )
        
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print("Response:", response.json()['choices'][0]['message']['content'])
        else:
            print("Error Response:", response.text)
            
    except Exception as e:
        print(f"Exception: {e}")
