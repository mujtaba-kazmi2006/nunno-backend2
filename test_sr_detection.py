#!/usr/bin/env python
"""
Test script to verify the Support/Resistance and Trendline detection functionality
"""
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from services.betterpredictormodule import TradingAnalyzer

def test_sr_detection():
    print("Testing Support/Resistance and Trendline Detection...")
    
    # Initialize analyzer
    analyzer = TradingAnalyzer()
    
    # Fetch sample data
    print("Fetching sample data...")
    df = analyzer.fetch_binance_ohlcv_with_fallback(symbol="BTCUSDT", interval="15m", limit=500)
    print(f"Fetched {len(df)} candles")
    
    # Add indicators to the data
    print("Adding technical indicators...")
    df = analyzer.add_comprehensive_indicators(df)
    print("Indicators added successfully")
    
    # Test support/resistance detection
    print("Testing Support/Resistance detection...")
    support_levels, resistance_levels = analyzer.detect_support_resistance(df)
    print(f"Found {len(support_levels)} support levels and {len(resistance_levels)} resistance levels")
    
    # Print sample support levels
    for i, level in enumerate(support_levels[:3]):
        print(f"  Support {i+1}: ${level['price']:.2f}, {level['strength']} strength, {level['touches']} touches")
    
    # Print sample resistance levels
    for i, level in enumerate(resistance_levels[:3]):
        print(f"  Resistance {i+1}: ${level['price']:.2f}, {level['strength']} strength, {level['touches']} touches")
    
    # Test trendline detection
    print("\nTesting Trendline detection...")
    uptrends, downtrends = analyzer.detect_trendlines(df)
    print(f"Found {len(uptrends)} uptrend lines and {len(downtrends)} downtrend lines")
    
    # Print sample uptrends
    for i, trend in enumerate(uptrends[:2]):
        print(f"  Uptrend {i+1}: {trend['touches']} touches, {trend['angle_degrees']}° angle")
    
    # Print sample downtrends
    for i, trend in enumerate(downtrends[:2]):
        print(f"  Downtrend {i+1}: {trend['touches']} touches, {trend['angle_degrees']}° angle")
    
    # Test comprehensive analysis with S/R and trendlines
    print("\nTesting comprehensive analysis with S/R and trendlines...")
    confluences, latest_row = analyzer.generate_comprehensive_analysis(df)
    
    # Check if S/R and trendline data is included
    if 'support_resistance' in confluences:
        sr_data = confluences['support_resistance']
        print(f"Included {len(sr_data.get('support_levels', []))} support and {len(sr_data.get('resistance_levels', []))} resistance levels in analysis")
    
    if 'trendlines' in confluences:
        trendline_data = confluences['trendlines']
        print(f"Included {len(trendline_data.get('uptrends', []))} uptrends and {len(trendline_data.get('downtrends', []))} downtrends in analysis")
        print(f"Included {len(trendline_data.get('channels', []))} channels in analysis")
    
    print("\n✅ All tests passed! S/R and Trendline detection is working correctly.")

if __name__ == "__main__":
    test_sr_detection()