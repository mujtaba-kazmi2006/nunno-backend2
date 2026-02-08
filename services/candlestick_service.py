import pandas as pd
import numpy as np
from typing import List, Dict, Optional

class CandlestickService:
    """Service for recognizing candlestick patterns and providing educational insights"""
    
    def __init__(self):
        self.patterns_metadata = {
            'pinbar_bullish': {
                'name': 'Bullish Pinbar (Hammer)',
                'sentiment': 'bullish',
                'description': 'A small body at the top with a long lower wick. Rejection of lower prices.',
                'trading_tip': 'Look for this at long-term support levels.'
            },
            'pinbar_bearish': {
                'name': 'Bearish Pinbar (Shooting Star)',
                'sentiment': 'bearish',
                'description': 'A small body at the bottom with a long upper wick. Rejection of higher prices.',
                'trading_tip': 'Look for this at key resistance zones.'
            },
            'engulfing_bullish': {
                'name': 'Bullish Engulfing',
                'sentiment': 'bullish',
                'description': 'A large green candle engulfing the previous red candle.',
                'trading_tip': 'Signifies a strong momentum shift.'
            },
            'engulfing_bearish': {
                'name': 'Bearish Engulfing',
                'sentiment': 'bearish',
                'description': 'A large red candle engulfing the previous green candle.',
                'trading_tip': 'Signifies sellers taking control.'
            },
            'doji': {
                'name': 'Doji',
                'sentiment': 'neutral',
                'description': 'Identical open/close prices. Market indecision.',
                'trading_tip': 'Wait for the next candle for direction.'
            },
            'morning_star': {
                'name': 'Morning Star',
                'sentiment': 'bullish',
                'description': '3-candle bottom reversal.',
                'trading_tip': 'High reliability reversal pattern.'
            },
            'evening_star': {
                'name': 'Evening Star',
                'sentiment': 'bearish',
                'description': '3-candle top reversal.',
                'trading_tip': 'High reliability reversal pattern.'
            },
            'marubozu_bullish': {
                'name': 'Bullish Marubozu',
                'sentiment': 'bullish',
                'description': 'Long green candle, no wicks.',
                'trading_tip': 'Strong trend continuation signal.'
            },
            'marubozu_bearish': {
                'name': 'Bearish Marubozu',
                'sentiment': 'bearish',
                'description': 'Long red candle, no wicks.',
                'trading_tip': 'Strong downward momentum.'
            },
            'harami_bullish': {
                'name': 'Bullish Harami',
                'sentiment': 'bullish',
                'description': 'Small green inside large red.',
                'trading_tip': 'Potential reversal starting.'
            },
            'harami_bearish': {
                'name': 'Bearish Harami',
                'sentiment': 'bearish',
                'description': 'Small red inside large green.',
                'trading_tip': 'Exhaustion signal.'
            },
            'tweezer_bottom': {
                'name': 'Tweezer Bottom',
                'sentiment': 'bullish',
                'description': 'Identical lows.',
                'trading_tip': 'Strong floor at support.'
            },
            'tweezer_top': {
                'name': 'Tweezer Top',
                'sentiment': 'bearish',
                'description': 'Identical highs.',
                'trading_tip': 'Strong ceiling at resistance.'
            },
            'piercing_line': {
                'name': 'Piercing Line',
                'sentiment': 'bullish',
                'description': 'Bullish reversal after downtrend.',
                'trading_tip': 'Second candle closes > 50% of first.'
            },
            'dark_cloud_cover': {
                'name': 'Dark Cloud Cover',
                'sentiment': 'bearish',
                'description': 'Bearish reversal pattern.',
                'trading_tip': 'Second candle closes < 50% of first.'
            },
            'three_soldiers': {
                'name': 'Three White Soldiers',
                'sentiment': 'bullish',
                'description': '3 long green candles.',
                'trading_tip': 'Uptrend confirmation.'
            },
            'three_crows': {
                'name': 'Three Black Crows',
                'sentiment': 'bearish',
                'description': '3 long red candles.',
                'trading_tip': 'Downtrend confirmation.'
            }
        }

    def detect_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detect patterns with LOOSE SR filtering to ensure markers are visible"""
        markers = []
        if len(df) < 10: return markers

        # Identify SR levels dynamically with multiple windows
        sr_levels = self._find_sr_levels(df)
        
        # INCREASE PROXIMITY THRESHOLD: Be much more "forgiving" to show markers
        # We use a 1.5x ATR window now instead of 0.25x
        if 'ATR' in df.columns:
            proximity_values = df['ATR'] * 1.5
        else:
            # Fallback to 0.5% of price
            proximity_values = df['Close'] * 0.005

        def is_near_sr(price, idx_pos):
            if not sr_levels: return True # If no levels found, show all (fallback)
            thresh = proximity_values.iloc[idx_pos]
            return any(abs(price - level) <= thresh for level in sr_levels)

        for i in range(2, len(df)):
            curr = df.iloc[i]
            prev = df.iloc[i-1]
            prev2 = df.iloc[i-2]
            
            # SR Proximity Checks
            is_at_support = is_near_sr(curr['Low'], i)
            is_at_resistance = is_near_sr(curr['High'], i)
            
            # Helper metrics
            body_size = abs(curr['Close'] - curr['Open'])
            prev_body = abs(prev['Close'] - prev['Open'])
            total_range = curr['High'] - curr['Low']
            if total_range == 0: continue
            
            upper_wick = curr['High'] - max(curr['Open'], curr['Close'])
            lower_wick = min(curr['Open'], curr['Close']) - curr['Low']
            
            # Use a slightly longer rolling window for avg_body context
            window_slice = df.iloc[max(0, i-20):i]
            avg_body = abs(window_slice['Close'] - window_slice['Open']).mean() or 0
            
            # --- Bullish Reversals (At Support) ---
            if is_at_support:
                # Hammer
                if lower_wick > (body_size * 1.8) and upper_wick < (body_size * 0.8) and total_range > (avg_body * 0.5):
                    markers.append({'time': idx_to_timestamp(df.index[i]), 'position': 'belowBar', 'color': '#22c55e', 'shape': 'arrowUp', 'text': 'Hammer', 'pattern': 'pinbar_bullish'})
                # Engulfing
                elif curr['Close'] > curr['Open'] and prev['Close'] < prev['Open'] and curr['Close'] >= prev['Open'] and curr['Open'] <= prev['Close'] and body_size > prev_body:
                    markers.append({'time': idx_to_timestamp(df.index[i]), 'position': 'belowBar', 'color': '#22c55e', 'shape': 'circle', 'text': 'B-Engulf', 'pattern': 'engulfing_bullish'})
                # Tweezer
                elif abs(curr['Low'] - prev['Low']) < (proximity_values.iloc[i] * 0.1):
                    markers.append({'time': idx_to_timestamp(df.index[i]), 'position': 'belowBar', 'color': '#22c55e', 'shape': 'arrowUp', 'text': 'T-Bottom', 'pattern': 'tweezer_bottom'})

            # --- Bearish Reversals (At Resistance) ---
            if is_at_resistance:
                # Shooting Star
                if upper_wick > (body_size * 1.8) and lower_wick < (body_size * 0.8) and total_range > (avg_body * 0.5):
                    markers.append({'time': idx_to_timestamp(df.index[i]), 'position': 'aboveBar', 'color': '#ef4444', 'shape': 'arrowDown', 'text': 'Star', 'pattern': 'pinbar_bearish'})
                # Engulfing
                elif curr['Close'] < curr['Open'] and prev['Close'] > prev['Open'] and curr['Close'] <= prev['Open'] and curr['Open'] >= prev['Close'] and body_size > prev_body:
                    markers.append({'time': idx_to_timestamp(df.index[i]), 'position': 'aboveBar', 'color': '#ef4444', 'shape': 'circle', 'text': 'S-Engulf', 'pattern': 'engulfing_bearish'})
                # Tweezer
                elif abs(curr['High'] - prev['High']) < (proximity_values.iloc[i] * 0.1):
                    markers.append({'time': idx_to_timestamp(df.index[i]), 'position': 'aboveBar', 'color': '#ef4444', 'shape': 'arrowDown', 'text': 'T-Top', 'pattern': 'tweezer_top'})

            # --- Universal Patterns (Always Show) ---
            # Marubozu (Institutional Momentum)
            if body_size > (avg_body * 2.0) and upper_wick < (body_size * 0.2) and lower_wick < (body_size * 0.2):
                if curr['Close'] > curr['Open']:
                    markers.append({'time': idx_to_timestamp(df.index[i]), 'position': 'belowBar', 'color': '#10b981', 'shape': 'square', 'text': 'MARU', 'pattern': 'marubozu_bullish'})
                else:
                    markers.append({'time': idx_to_timestamp(df.index[i]), 'position': 'aboveBar', 'color': '#f43f5e', 'shape': 'square', 'text': 'MARU', 'pattern': 'marubozu_bearish'})

            # Doji (Indecision)
            if body_size < (total_range * 0.05):
                markers.append({'time': idx_to_timestamp(df.index[i]), 'position': 'inBar', 'color': '#94a3b8', 'shape': 'circle', 'text': 'Doji', 'pattern': 'doji'})

        return markers

    def _find_sr_levels(self, df: pd.DataFrame) -> List[float]:
        """Detect local peaks/valleys and pivot levels"""
        levels = []
        # Local peaks (window 10 for more levels)
        window = 10
        for i in range(window, len(df) - window):
            h = df['High'].iloc[i]
            l = df['Low'].iloc[i]
            if h == df['High'].iloc[i-window:i+window+1].max(): levels.append(float(h))
            if l == df['Low'].iloc[i-window:i+window+1].min(): levels.append(float(l))
            
        # Add Fibonacci-like Pivots from the whole DF
        if len(df) > 0:
            h, l, c = df['High'].max(), df['Low'].min(), df['Close'].iloc[-1]
            pivot = (h + l + c) / 3
            levels.extend([float(pivot), float(h), float(l)])
            
        # S1/R1 from indicators
        if 'R1' in df.columns: levels.append(float(df['R1'].iloc[-1]))
        if 'S1' in df.columns: levels.append(float(df['S1'].iloc[-1]))
        
        return list(set(levels))

def idx_to_timestamp(idx):
    if hasattr(idx, 'timestamp'): return int(idx.timestamp())
    try: return int(idx)
    except: return 0
