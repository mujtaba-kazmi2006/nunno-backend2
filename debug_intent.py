
import asyncio
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.chat_service import ChatService

async def verify_intent():
    print("Initializing ChatService...")
    # Mocking API key to avoid errors in init if missing
    if not os.getenv("OPENROUTER_API_KEY"):
        os.environ["OPENROUTER_API_KEY"] = "mock_key"
        
    service = ChatService()
    
    test_cases = [
        # Expected: No tools (Fast Path)
        ("Hi", False),
        ("Who are you?", False),
        ("What is your name?", False),
        ("Tell me a joke", False),
        ("What is love?", False), # Should fail strong context check
        
        # Expected: Tokenomics (Direct or Intent)
        ("What is Bitcoin?", True),
        ("Tell me about Solana", True),
        ("Tokenomics of PEPE", True),
        ("What is Grok coin?", True), # Has "coin" context
        
        # Expected: Technical
        ("Predict BTC", True),
        ("Should I buy ETH?", True),
        ("Analysis for SOL", True),
        
        # Expected: News
        ("Why is the market crashing?", True),
    ]
    
    print("\n--- Running Intent Verification ---\n")
    
    passed = 0
    total = len(test_cases)
    
    for message, expect_tools in test_cases:
        print(f"Testing: '{message}'")
        
        # We only test the fallback logic first to see what it "thinks"
        # Since _classify_intent calls fallback internally, we can use that logic
        # But we want to see if it determines _needs_llm or finds tools directly
        
        fallback_tools = service._detect_tools_needed_fallback(message)
        
        has_tools = any(t[0] != "_needs_llm" for t in fallback_tools)
        needs_llm = any(t[0] == "_needs_llm" for t in fallback_tools)
        
        result_boolean = has_tools or needs_llm
        
        status = "✅ PASS" if result_boolean == expect_tools else "❌ FAIL"
        if result_boolean == expect_tools:
            passed += 1
            
        print(f"  -> Result: {'Tools/Intent Found' if result_boolean else 'No Tools Needed'}")
        print(f"  -> {status}")
        print("-" * 30)
        
    print(f"\nVerification Complete: {passed}/{total} Passed")

if __name__ == "__main__":
    asyncio.run(verify_intent())
