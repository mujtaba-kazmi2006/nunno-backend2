#!/usr/bin/env python3
"""
Debug script to understand the confidence calculation
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.services.betterpredictormodule import TradingAnalyzer


def debug_confidence_calculation():
    """Debug the confidence calculation step by step"""
    print("🔍 Debugging Confidence Calculation")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = TradingAnalyzer()
    
    # Generate synthetic data
    df = analyzer._generate_synthetic_data("BTCUSDT", "15m", 100)
    df = analyzer.add_comprehensive_indicators(df)
    
    # Get the original confluence calculation before our changes
    # First, let's manually calculate using the old method
    latest_row = df.iloc[-1]
    
    # Run the enhanced analysis
    confluences, latest_row = analyzer.generate_comprehensive_analysis(df)
    
    print(f"Bullish signals: {len(confluences['bullish'])}")
    print(f"Bearish signals: {len(confluences['bearish'])}")
    print(f"Neutral signals: {len(confluences['neutral'])}")
    
    # Print the specific confluence signals to see what's in each category
    print("\nBullish Signals:")
    for i, sig in enumerate(confluences['bullish']):
        print(f"  {i+1}. {sig['indicator']} - {sig['strength']} - {sig['condition']}")
    
    print("\nBearish Signals:")
    for i, sig in enumerate(confluences['bearish']):
        print(f"  {i+1}. {sig['indicator']} - {sig['strength']} - {sig['condition']}")
    
    # Calculate the weighted scores manually to see what's happening
    indicator_weights = {
        'MACD': 1.5,
        'RSI (14)': 1.3,
        'Stochastic': 1.2,
        'EMA Alignment': 1.4,
        'Price vs EMA 21': 1.1,
        'ADX Trend Strength': 1.3,
        'Bollinger Bands': 1.0,
        'Volume': 0.8,
        'Price Action': 1.2,
        'Williams %R': 1.1,
        'ML Support Zone': 1.6,
        'ML Resistance Zone': 1.6,
    }
    
    print("\nCalculating weighted scores:")
    
    bullish_score = 0
    for conf in confluences['bullish']:
        base_weight = {'Strong': 3, 'Medium': 2, 'Low': 1}[conf['strength']]
        indicator_weight = indicator_weights.get(conf['indicator'], 1.0)
        weighted_score = base_weight * indicator_weight
        bullish_score += weighted_score
        print(f"  Bullish: {conf['indicator']} ({conf['strength']}) * {indicator_weight} = {weighted_score}")
    
    bearish_score = 0
    for conf in confluences['bearish']:
        base_weight = {'Strong': 3, 'Medium': 2, 'Low': 1}[conf['strength']]
        indicator_weight = indicator_weights.get(conf['indicator'], 1.0)
        weighted_score = base_weight * indicator_weight
        bearish_score += weighted_score
        print(f"  Bearish: {conf['indicator']} ({conf['strength']}) * {indicator_weight} = {weighted_score}")
    
    neutral_score = 0
    for conf in confluences['neutral']:
        base_weight = {'Strong': 3, 'Medium': 2, 'Low': 1}[conf['strength']]
        indicator_weight = indicator_weights.get(conf['indicator'], 1.0)
        weighted_score = base_weight * indicator_weight
        neutral_score += weighted_score
    
    total_score = bullish_score + bearish_score + neutral_score
    enhanced_threshold = analyzer.confluence_threshold * 1.2  # 3.6
    
    print(f"\nRaw counts - Bullish: {len(confluences['bullish'])}, Bearish: {len(confluences['bearish'])}")
    print(f"Weighted scores - Bullish: {bullish_score:.2f}, Bearish: {bearish_score:.2f}, Neutral: {neutral_score:.2f}")
    print(f"Total score: {total_score:.2f}")
    print(f"Enhanced threshold: {enhanced_threshold}")
    print(f"Difference: {abs(bullish_score - bearish_score):.2f}")
    print(f"Min difference for bias: {(bullish_score + bearish_score) * 0.15:.2f}")
    
    # Now calculate the final result
    bias, strength = analyzer.calculate_confluence_strength(confluences)
    print(f"\nFinal result: {bias} with {strength:.2f}% confidence")


if __name__ == "__main__":
    debug_confidence_calculation()