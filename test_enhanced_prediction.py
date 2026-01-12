"""
Test Enhanced Prediction Module
Verifies new features: market regime, trend strength, volume profile, reasoning chain
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'services'))

from services.technical_analysis import TechnicalAnalysisService

def test_enhanced_prediction():
    print("=" * 80)
    print("Testing Enhanced Prediction Module")
    print("=" * 80)
    
    service = TechnicalAnalysisService()
    
    # Test with Bitcoin
    print("\n🔍 Testing with BTCUSDT...")
    result = service.analyze("BTCUSDT", "15m")
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return False
    
    # Verify new fields exist
    required_fields = ["market_regime", "trend_strength", "volume_profile", "reasoning_chain"]
    
    print("\n✅ Checking for new fields...")
    for field in required_fields:
        if field in result:
            print(f"  ✓ {field}: Present")
        else:
            print(f"  ✗ {field}: MISSING")
            return False
    
    # Display market regime
    print("\n📊 Market Regime:")
    regime = result["market_regime"]
    print(f"  Regime: {regime['regime']}")
    print(f"  Confidence: {regime['confidence']}%")
    print(f"  Description: {regime['description']}")
    
    # Display trend strength
    print("\n💪 Trend Strength:")
    strength = result["trend_strength"]
    print(f"  Score: {strength['score']}/100")
    print(f"  Level: {strength['level']}")
    print(f"  Components: {strength['components']}")
    
    # Display volume profile
    print("\n📈 Volume Profile:")
    volume = result["volume_profile"]
    print(f"  Profile: {volume['profile']}")
    print(f"  Strength: {volume['strength']}")
    print(f"  Description: {volume['description']}")
    
    # Display reasoning chain
    print("\n🔗 Reasoning Chain:")
    chain = result["reasoning_chain"]
    for step in chain:
        print(f"  Step {step['step']}: {step['category']}")
        print(f"    Finding: {step['finding']}")
        print(f"    Impact: {step['impact'][:100]}...")
    
    print("\n" + "=" * 80)
    print("✅ All tests passed! Enhanced prediction module is working correctly.")
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    success = test_enhanced_prediction()
    sys.exit(0 if success else 1)
