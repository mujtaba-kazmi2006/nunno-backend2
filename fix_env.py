
import os

content = """# OpenRouter API Key
OPENROUTER_API_KEY=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

# AI Model Configuration
AI_MODEL=meta-llama/llama-3.3-70b-instruct:free

# Server Configuration
HOST=0.0.0.0
PORT=8000
"""

with open(".env", "w", encoding="utf-8") as f:
    f.write(content)

print("✅ Successfully wrote .env file with correct API key")
