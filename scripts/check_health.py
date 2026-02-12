import os
import requests
import json
import time
from datetime import datetime

def check_health():
    API_URL = os.getenv("API_URL", "http://localhost:8000")
    print(f"--- NUNNO PRODUCTION HEALTH CHECK ({datetime.now().isoformat()}) ---")
    
    # 1. API Availability
    try:
        start = time.time()
        resp = requests.get(f"{API_URL}/api/v1/market/highlights", timeout=10)
        latency = (time.time() - start) * 1000
        if resp.status_code == 200:
            print(f"[OK] Backend API is ONLINE (Latency: {latency:.2f}ms)")
        else:
            print(f"[FAIL] Backend API returned {resp.status_code}")
    except Exception as e:
        print(f"[CRITICAL] Backend API is OFFLINE: {e}")

    # 2. Redis Check (via internal endpoint or generic check)
    # Since we can't easily check Redis from outside without an endpoint, 
    # we'll assume the internal logs will catch it.

    # 3. Model Fallback Test
    # This usually requires a real query, but we'll skip for automated health check
    
    print("-" * 50)

if __name__ == "__main__":
    check_health()
