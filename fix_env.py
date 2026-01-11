
import os

content = """# OpenRouter API Key
OPENROUTER_API_KEY=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

# AI Model Configuration
AI_MODEL=meta-llama/llama-3.2-3b-instruct:free

# Server Configuration
HOST=0.0.0.0
PORT=8000
"""

with open(".env", "w", encoding="utf-8") as f:
    f.write(content)

print("✅ Successfully wrote .env file")
