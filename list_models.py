
import requests
import json

response = requests.get("https://openrouter.ai/api/v1/models")

if response.status_code == 200:
    data = response.json().get("data", [])
    print(f"Found {len(data)} models.")
    
    # Filter for free models
    free_models = []
    for m in data:
        id = m.get("id", "")
        pricing = m.get("pricing", {})
        # Check if pricing is zero or if id contains free
        is_free_price = (
            float(pricing.get("prompt", 0)) == 0 and 
            float(pricing.get("completion", 0)) == 0
        )
        
        if ":free" in id or is_free_price:
            free_models.append(id)
            
    print("Available Free Models:")
    for m in free_models:
        print(f"- {m}")
else:
    print(f"Failed to list models: {response.status_code}")
