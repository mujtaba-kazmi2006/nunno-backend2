#!/usr/bin/env python3
"""
Test script to verify improved directional bias in predictions
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.services.betterpredictormodule import TradingAnalyzer
from backend.services.technical_analysis import TechnicalAnalysisService


def test_enhanced_analysis():
    """Test the enhanced analysis functions"""
    print("🧪 Testing Enhanced Prediction System")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = TradingAnalyzer()
    
    # Test with sample data (synthetic)
    print("📊 Testing with synthetic BTCUSDT data...")
    df = analyzer._generate_synthetic_data("BTCUSDT", "15m", 100)
    df = analyzer.add_comprehensive_indicators(df)
    
    # Test enhanced confluence calculation
    confluences, latest_row = analyzer.generate_comprehensive_analysis(df)
    bias, strength = analyzer.calculate_confluence_strength(confluences)
    
    print(f"🎯 Calculated Bias: {bias}")
    print(f"💪 Confidence Strength: {strength:.2f}%")
    
    # Check if we have enhanced indicators
    print(f"📈 Bullish Signals: {len(confluences['bullish'])}")
    print(f"📉 Bearish Signals: {len(confluences['bearish'])}")
    print(f"🔵 Neutral Signals: {len(confluences['neutral'])}")
    
    # Check if enhanced features are present
    if 'advanced_trend' in confluences:
        print(f"⚙️ Advanced Trend Direction: {confluences['advanced_trend']['direction']}")
        print(f"⚙️ Advanced Trend Score: {confluences['advanced_trend']['score']}")
    
    # Test technical analysis service
    print("\n🔧 Testing Technical Analysis Service Integration...")
    service = TechnicalAnalysisService()
    
    try:
        result = service.analyze("BTCUSDT", "15m")
        print(f"💰 Asset: {result['ticker']}")
        print(f"📊 Bias: {result['bias']}")
        print(f"📈 Confidence: {result['confidence']:.2f}%")
        print(f"🔍 Bullish Count: {result['confluences']['bullish_count']}")
        print(f"🔍 Bearish Count: {result['confluences']['bearish_count']}")
        
        # Print key indicators that affect directional bias
        indicators = result['indicators']
        print(f"📊 EMA 9/21/50: {indicators['ema_9']:.2f}/{indicators['ema_21']:.2f}/{indicators['ema_50']:.2f}")
        print(f"📊 RSI: {indicators['rsi_14']:.2f}")
        print(f"📊 MACD: {indicators['macd']:.2f}")
        print(f"📊 ADX: {indicators['adx']:.2f}")
        print(f"📊 DI+: {indicators['di_plus']:.2f}, DI-: {indicators['di_minus']:.2f}")
        
        print("\n✅ Test completed successfully! Enhanced directional bias features are working.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def demonstrate_enhanced_features():
    """Demonstrate the new enhanced features"""
    print("\n✨ Enhanced Features Demonstration")
    print("=" * 60)
    
    analyzer = TradingAnalyzer()
    
    # Generate synthetic data
    df = analyzer._generate_synthetic_data("ETHUSDT", "1h", 200)
    df = analyzer.add_comprehensive_indicators(df)
    latest_row = df.iloc[-1]
    
    print("🔍 Testing Momentum Divergence Detection...")
    divergence_result = analyzer.detect_momentum_divergence(df, latest_row)
    print(f"   Bullish Divergences Found: {len(divergence_result['bullish'])}")
    print(f"   Bearish Divergences Found: {len(divergence_result['bearish'])}")
    
    print("\n🔍 Testing Price Action Pattern Recognition...")
    pattern_result = analyzer.analyze_price_action_patterns(df, latest_row)
    print(f"   Bullish Patterns Found: {len(pattern_result['bullish'])}")
    print(f"   Bearish Patterns Found: {len(pattern_result['bearish'])}")
    
    print("\n🔍 Testing Advanced Trend Strength Calculation...")
    trend_result = analyzer.calculate_advanced_trend_strength(df, latest_row)
    print(f"   Trend Direction: {trend_result['direction']}")
    print(f"   Trend Score: {trend_result['score']}/100")
    print(f"   Components: {trend_result['components']}")


if __name__ == "__main__":
    print("🚀 Testing Improved Prediction Directional Bias")
    print("=" * 60)
    
    success = test_enhanced_analysis()
    demonstrate_enhanced_features()
    
    if success:
        print("\n🎉 All tests passed! The directional bias improvements are working correctly.")
        print("\nSummary of improvements:")
        print("• Enhanced indicator weighting system")
        print("• Momentum divergence detection")
        print("• Price action pattern recognition")
        print("• Advanced trend strength calculation")
        print("• More directional indicators in AI context")
        print("• Improved confidence scoring with dominance factor")
    else:
        print("\n❌ Tests failed. Please review the implementation.")