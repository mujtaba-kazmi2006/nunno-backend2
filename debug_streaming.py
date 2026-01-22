import requests
import json
import time

def test_streaming():
    url = "http://localhost:8000/api/v1/chat/stream"
    payload = {
        "message": "Analyze Bitcoin",
        "user_name": "TestUser",
        "user_age": 25,
        "conversation_history": []
    }
    
    print(f"Connecting to {url}...")
    try:
        # We'll use a timeout of 60 seconds
        response = requests.post(url, json=payload, stream=True, timeout=60)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code != 200:
            print(f"Error: {response.text}")
            return

        print("Waiting for chunks...")
        start_time = time.time()
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                print(f"[{time.time() - start_time:.2f}s] Received: {decoded_line[:100]}...")
                if "error" in decoded_line.lower():
                    print("⚠️ Found error in stream.")
                if "text" in decoded_line:
                    # Once we get text, we know it's working
                    # print("✅ Got text chunk, streaming works!")
                    pass
    except requests.exceptions.Timeout:
        print("❌ Timeout reached!")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    # Note: This assumes the backend is running
    test_streaming()
