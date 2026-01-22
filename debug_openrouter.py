import os
import requests
import json
from dotenv import load_dotenv

# Load .env
load_dotenv('backend/.env')

api_key = os.getenv("OPENROUTER_API_KEY")
model = "openai/gpt-4o-mini"

print(f"DEBUG: Testing OpenRouter Connection")
print(f"DEBUG: API Key present: {'Yes' if api_key else 'No'}")
print(f"DEBUG: Model: {model}")

url = "https://openrouter.ai/api/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://nunno.finance",
    "X-Title": "Nunno Finance Debug"
}
data = {
    "model": model,
    "max_tokens": 100,
    "messages": [
        {"role": "user", "content": "Hi"}
    ]
}

print(f"DEBUG: Sending request to {url}...")

try:
    response = requests.post(url, headers=headers, json=data)
    print(f"DEBUG: Status Code: {response.status_code}")
    print(f"DEBUG: Response Text: {response.text}")
except Exception as e:
    print(f"DEBUG: Exception: {e}")
