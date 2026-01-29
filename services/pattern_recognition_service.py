"""
Pattern Recognition Service - Enhanced Edition
Recognizes chart patterns from user queries and generates structured data with trendlines and annotations
"""
import re
from typing import Dict, List, Optional, Tuple
import random
import math


class PatternRecognitionService:
    """Service for recognizing and generating chart patterns with enhanced metadata"""
    
    # Expanded pattern definitions with keywords and metadata
    PATTERNS = {
        # === REVERSAL PATTERNS ===
        'head_and_shoulders': {
            'keywords': ['head and shoulders', 'head & shoulders', 'h&s', 'hns'],
            'type': 'reversal',
            'direction': 'bearish',
            'success_rate': 0.83,
            'category': 'reversal'
        },
        'inverse_head_and_shoulders': {
            'keywords': ['inverse head and shoulders', 'inverse h&s', 'reverse head and shoulders'],
            'type': 'reversal',
            'direction': 'bullish',
            'success_rate': 0.85,
            'category': 'reversal'
        },
        'double_top': {
            'keywords': ['double top', 'twin peaks', 'double peak', 'm top'],
            'type': 'reversal',
            'direction': 'bearish',
            'success_rate': 0.78,
            'category': 'reversal'
        },
        'double_bottom': {
            'keywords': ['double bottom', 'twin valleys', 'w pattern', 'w bottom'],
            'type': 'reversal',
            'direction': 'bullish',
            'success_rate': 0.79,
            'category': 'reversal'
        },
        'triple_top': {
            'keywords': ['triple top', 'three peaks'],
            'type': 'reversal',
            'direction': 'bearish',
            'success_rate': 0.81,
            'category': 'reversal'
        },
        'triple_bottom': {
            'keywords': ['triple bottom', 'three valleys'],
            'type': 'reversal',
            'direction': 'bullish',
            'success_rate': 0.82,
            'category': 'reversal'
        },
        'rounding_bottom': {
            'keywords': ['rounding bottom', 'saucer bottom', 'u shape', 'bowl'],
            'type': 'reversal',
            'direction': 'bullish',
            'success_rate': 0.76,
            'category': 'reversal'
        },
        'rounding_top': {
            'keywords': ['rounding top', 'dome', 'inverted u', 'inverse bowl'],
            'type': 'reversal',
            'direction': 'bearish',
            'success_rate': 0.74,
            'category': 'reversal'
        },
        'falling_wedge': {
            'keywords': ['falling wedge', 'descending wedge'],
            'type': 'reversal',
            'direction': 'bullish',
            'success_rate': 0.72,
            'category': 'reversal'
        },
        'rising_wedge': {
            'keywords': ['rising wedge', 'ascending wedge'],
            'type': 'reversal',
            'direction': 'bearish',
            'success_rate': 0.73,
            'category': 'reversal'
        },
        
        # === CONTINUATION PATTERNS ===
        'ascending_triangle': {
            'keywords': ['ascending triangle', 'bullish triangle'],
            'type': 'continuation',
            'direction': 'bullish',
            'success_rate': 0.75,
            'category': 'continuation'
        },
        'descending_triangle': {
            'keywords': ['descending triangle', 'bearish triangle'],
            'type': 'continuation',
            'direction': 'bearish',
            'success_rate': 0.76,
            'category': 'continuation'
        },
        'symmetrical_triangle': {
            'keywords': ['symmetrical triangle', 'symmetric triangle', 'coil'],
            'type': 'continuation',
            'direction': 'neutral',
            'success_rate': 0.70,
            'category': 'continuation'
        },
        'cup_and_handle': {
            'keywords': ['cup and handle', 'cup & handle', 'cup with handle'],
            'type': 'continuation',
            'direction': 'bullish',
            'success_rate': 0.89,
            'category': 'continuation'
        },
        'inverse_cup_and_handle': {
            'keywords': ['inverse cup and handle', 'inverted cup', 'reverse cup'],
            'type': 'continuation',
            'direction': 'bearish',
            'success_rate': 0.87,
            'category': 'continuation'
        },
        'bull_flag': {
            'keywords': ['bull flag', 'bullish flag', 'flag up'],
            'type': 'continuation',
            'direction': 'bullish',
            'success_rate': 0.68,
            'category': 'continuation'
        },
        'bear_flag': {
            'keywords': ['bear flag', 'bearish flag', 'flag down'],
            'type': 'continuation',
            'direction': 'bearish',
            'success_rate': 0.67,
            'category': 'continuation'
        },
        'pennant': {
            'keywords': ['pennant', 'triangle pennant'],
            'type': 'continuation',
            'direction': 'neutral',
            'success_rate': 0.65,
            'category': 'continuation'
        },
        'rectangle': {
            'keywords': ['rectangle', 'trading range', 'consolidation box'],
            'type': 'continuation',
            'direction': 'neutral',
            'success_rate': 0.71,
            'category': 'continuation'
        },
        
        # === BILATERAL PATTERNS ===
        'broadening_top': {
            'keywords': ['broadening top', 'megaphone top', 'expanding triangle'],
            'type': 'bilateral',
            'direction': 'bearish',
            'success_rate': 0.64,
            'category': 'bilateral'
        },
        'broadening_bottom': {
            'keywords': ['broadening bottom', 'megaphone bottom', 'expanding bottom'],
            'type': 'bilateral',
            'direction': 'bullish',
            'success_rate': 0.66,
            'category': 'bilateral'
        },
        'diamond_top': {
            'keywords': ['diamond top', 'diamond reversal'],
            'type': 'reversal',
            'direction': 'bearish',
            'success_rate': 0.77,
            'category': 'bilateral'
        },
        'diamond_bottom': {
            'keywords': ['diamond bottom', 'diamond base'],
            'type': 'reversal',
            'direction': 'bullish',
            'success_rate': 0.78,
            'category': 'bilateral'
        },
    }
    
    def __init__(self):
        pass
    
    def recognize_pattern(self, query: str) -> Optional[str]:
        """Recognize a chart pattern from user query"""
        query_lower = query.lower()
        
        for pattern_name, pattern_info in self.PATTERNS.items():
            for keyword in pattern_info['keywords']:
                if keyword in query_lower:
                    return pattern_name
        
        return None
    
    def _get_scale_factor(self, interval: str) -> float:
        """Get vertical scale factor based on chart interval"""
        factors = {
            '1m': 0.05,
            '5m': 0.1,
            '15m': 0.2,
            '1h': 0.4,
            '4h': 0.7,
            '1d': 1.0,
            '1w': 2.0
        }
        return factors.get(interval, 1.0)
    
    def generate_pattern_data(
        self, 
        pattern_name: str, 
        base_price: float = 50000,
        num_points: int = 50,
        volatility: float = 0.01,
        interval: str = '1d'
    ) -> Dict:
        """
        Generate structured data for a chart pattern with trendlines and annotations
        """
        if pattern_name not in self.PATTERNS:
            return None
        
        pattern_info = self.PATTERNS[pattern_name]
        
        # Generate pattern-specific data
        generator_map = {
            'head_and_shoulders': self._generate_head_and_shoulders,
            'inverse_head_and_shoulders': self._generate_inverse_head_and_shoulders,
            'double_top': self._generate_double_top,
            'double_bottom': self._generate_double_bottom,
            'triple_top': self._generate_triple_top,
            'triple_bottom': self._generate_triple_bottom,
            'ascending_triangle': self._generate_ascending_triangle,
            'descending_triangle': self._generate_descending_triangle,
            'symmetrical_triangle': self._generate_symmetrical_triangle,
            'cup_and_handle': self._generate_cup_and_handle,
            'inverse_cup_and_handle': self._generate_inverse_cup_and_handle,
            'falling_wedge': self._generate_falling_wedge,
            'rising_wedge': self._generate_rising_wedge,
            'bull_flag': self._generate_bull_flag,
            'bear_flag': self._generate_bear_flag,
            'pennant': self._generate_pennant,
            'rounding_bottom': self._generate_rounding_bottom,
            'rounding_top': self._generate_rounding_top,
            'rectangle': self._generate_rectangle,
            'broadening_top': self._generate_broadening_top,
            'broadening_bottom': self._generate_broadening_bottom,
            'diamond_top': self._generate_diamond_top,
            'diamond_bottom': self._generate_diamond_bottom,
        }
        
        generator = generator_map.get(pattern_name)
        if not generator:
            data = []
            trendlines = []
            annotations = []
        else:
            scale = self._get_scale_factor(interval)
            data, trendlines, annotations = generator(base_price, num_points, volatility, scale)
        
        return {
            'pattern_name': pattern_name,
            'pattern_type': pattern_info['type'],
            'direction': pattern_info['direction'],
            'success_rate': pattern_info.get('success_rate', 0.70),
            'category': pattern_info.get('category', 'other'),
            'data': data,
            'trendlines': trendlines,
            'annotations': annotations,
            'description': self._get_pattern_description(pattern_name),
            'key_levels': self._extract_key_levels(data, pattern_name),
            'trading_tips': self._get_trading_tips(pattern_name)
        }
    
    # Pattern generators return (data, trendlines, annotations)
    
    def _generate_head_and_shoulders(self, base_price: float, num_points: int, volatility: float, scale: float = 1.0) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Generate head and shoulders with neckline"""
        data = []
        points_per_section = num_points // 7
        
        # Left shoulder
        for i in range(points_per_section):
            progress = i / points_per_section
            price = base_price + (base_price * 0.05 * scale * progress) + random.uniform(-volatility * scale, volatility * scale) * base_price
            data.append({'x': len(data), 'y': price})
        
        left_shoulder_peak = data[-1]['y']
        
        # Dip after left shoulder
        for i in range(points_per_section):
            progress = i / points_per_section
            price = data[-1]['y'] - (base_price * 0.03 * scale * progress) + random.uniform(-volatility * scale, volatility * scale) * base_price
            data.append({'x': len(data), 'y': price})
        
        neckline_left = data[-1]['y']
        
        # Head
        for i in range(points_per_section):
            progress = i / points_per_section
            price = data[-1]['y'] + (base_price * 0.08 * scale * progress) + random.uniform(-volatility * scale, volatility * scale) * base_price
            data.append({'x': len(data), 'y': price})
        
        head_peak = data[-1]['y']
        
        # Dip after head
        for i in range(points_per_section):
            progress = i / points_per_section
            price = data[-1]['y'] - (base_price * 0.06 * scale * progress) + random.uniform(-volatility * scale, volatility * scale) * base_price
            data.append({'x': len(data), 'y': price})
        
        neckline_right = data[-1]['y']
        
        # Right shoulder
        for i in range(points_per_section):
            progress = i / points_per_section
            price = data[-1]['y'] + (base_price * 0.04 * scale * progress) + random.uniform(-volatility * scale, volatility * scale) * base_price
            data.append({'x': len(data), 'y': price})
        
        right_shoulder_peak = data[-1]['y']
        
        # Breakdown
        for i in range(points_per_section):
            progress = i / points_per_section
            price = data[-1]['y'] - (base_price * 0.07 * scale * progress) + random.uniform(-volatility * scale, volatility * scale) * base_price
            data.append({'x': len(data), 'y': price})
        
        # Neckline trendline
        neckline_y = (neckline_left + neckline_right) / 2
        trendlines = [{
            'type': 'neckline',
            'x1': points_per_section,
            'y1': neckline_left,
            'x2': points_per_section * 4,
            'y2': neckline_right,
            'color': '#ef4444',
            'label': 'Neckline'
        }]
        
        # Annotations
        annotations = [
            {'x': points_per_section // 2, 'y': left_shoulder_peak, 'label': 'Left Shoulder', 'type': 'peak'},
            {'x': points_per_section * 3, 'y': head_peak, 'label': 'Head', 'type': 'peak'},
            {'x': points_per_section * 5, 'y': right_shoulder_peak, 'label': 'Right Shoulder', 'type': 'peak'},
        ]
        
        return data, trendlines, annotations
    
    def _generate_inverse_head_and_shoulders(self, base_price: float, num_points: int, volatility: float, scale: float = 1.0) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Generate inverse H&S with neckline"""
        data = []
        points_per_section = num_points // 7
        
        # Left shoulder (down)
        for i in range(points_per_section):
            progress = i / points_per_section
            price = base_price - (base_price * 0.05 * progress) + random.uniform(-volatility, volatility) * base_price
            data.append({'x': len(data), 'y': price})
        
        # Rally after left shoulder
        for i in range(points_per_section):
            progress = i / points_per_section
            price = data[-1]['y'] + (base_price * 0.03 * progress) + random.uniform(-volatility, volatility) * base_price
            data.append({'x': len(data), 'y': price})
        
        neckline_left = data[-1]['y']
        
        # Head (deeper low)
        for i in range(points_per_section):
            progress = i / points_per_section
            price = data[-1]['y'] - (base_price * 0.08 * progress) + random.uniform(-volatility, volatility) * base_price
            data.append({'x': len(data), 'y': price})
        
        # Rally after head
        for i in range(points_per_section):
            progress = i / points_per_section
            price = data[-1]['y'] + (base_price * 0.06 * progress) + random.uniform(-volatility, volatility) * base_price
            data.append({'x': len(data), 'y': price})
        
        neckline_right = data[-1]['y']
        
        # Right shoulder
        for i in range(points_per_section):
            progress = i / points_per_section
            price = data[-1]['y'] - (base_price * 0.04 * progress) + random.uniform(-volatility, volatility) * base_price
            data.append({'x': len(data), 'y': price})
        
        # Breakout
        for i in range(points_per_section):
            progress = i / points_per_section
            price = data[-1]['y'] + (base_price * 0.07 * progress) + random.uniform(-volatility, volatility) * base_price
            data.append({'x': len(data), 'y': price})
        
        trendlines = [{
            'type': 'neckline',
            'x1': points_per_section,
            'y1': neckline_left,
            'x2': points_per_section * 4,
            'y2': neckline_right,
            'color': '#22c55e',
            'label': 'Neckline'
        }]
        
        annotations = [
            {'x': points_per_section // 2, 'y': data[points_per_section // 2]['y'], 'label': 'Left Shoulder', 'type': 'valley'},
            {'x': points_per_section * 3, 'y': data[points_per_section * 3]['y'], 'label': 'Head', 'type': 'valley'},
            {'x': points_per_section * 5, 'y': data[points_per_section * 5]['y'], 'label': 'Right Shoulder', 'type': 'valley'},
        ]
        
        return data, trendlines, annotations
    
    def _generate_double_top(self, base_price: float, num_points: int, volatility: float, scale: float = 1.0) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Generate double top"""
        data = []
        points_per_section = num_points // 5
        
        # First peak
        for i in range(points_per_section):
            progress = i / points_per_section
            price = base_price + (base_price * 0.06 * scale * progress) + random.uniform(-volatility * scale, volatility * scale) * base_price
            data.append({'x': len(data), 'y': price})
        
        peak1 = data[-1]['y']
        
        # Dip
        for i in range(points_per_section):
            progress = i / points_per_section
            price = data[-1]['y'] - (base_price * 0.04 * scale * progress) + random.uniform(-volatility * scale, volatility * scale) * base_price
            data.append({'x': len(data), 'y': price})
        
        valley = data[-1]['y']
        
        # Second peak
        for i in range(points_per_section):
            progress = i / points_per_section
            price = data[-1]['y'] + (base_price * 0.04 * scale * progress) + random.uniform(-volatility * scale, volatility * scale) * base_price
            data.append({'x': len(data), 'y': price})
        
        peak2 = data[-1]['y']
        
        # Breakdown
        for i in range(points_per_section * 2):
            progress = i / (points_per_section * 2)
            price = data[-1]['y'] - (base_price * 0.08 * scale * progress) + random.uniform(-volatility * scale, volatility * scale) * base_price
            data.append({'x': len(data), 'y': price})
        
        # Resistance line connecting peaks
        trendlines = [{
            'type': 'resistance',
            'x1': points_per_section - 1,
            'y1': peak1,
            'x2': points_per_section * 3 - 1,
            'y2': peak2,
            'color': '#ef4444',
            'label': 'Resistance'
        }]
        
        annotations = [
            {'x': points_per_section - 1, 'y': peak1, 'label': 'Peak 1', 'type': 'peak'},
            {'x': points_per_section * 3 - 1, 'y': peak2, 'label': 'Peak 2', 'type': 'peak'},
        ]
        
        return data, trendlines, annotations
    
    def _generate_double_bottom(self, base_price: float, num_points: int, volatility: float, scale: float = 1.0) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Generate double bottom"""
        data = []
        points_per_section = num_points // 5
        
        # First trough
        for i in range(points_per_section):
            progress = i / points_per_section
            price = base_price - (base_price * 0.06 * progress) + random.uniform(-volatility, volatility) * base_price
            data.append({'x': len(data), 'y': price})
        
        # Rally
        for i in range(points_per_section):
            progress = i / points_per_section
            price = data[-1]['y'] + (base_price * 0.04 * scale * progress) + random.uniform(-volatility * scale, volatility * scale) * base_price
            data.append({'x': len(data), 'y': price})
        
        # Second trough
        for i in range(points_per_section):
            progress = i / points_per_section
            price = data[-1]['y'] - (base_price * 0.04 * scale * progress) + random.uniform(-volatility * scale, volatility * scale) * base_price
            data.append({'x': len(data), 'y': price})
        
        # Breakout
        for i in range(points_per_section * 2):
            progress = i / (points_per_section * 2)
            price = data[-1]['y'] + (base_price * 0.08 * progress) + random.uniform(-volatility, volatility) * base_price
            data.append({'x': len(data), 'y': price})
        
        # Support line connecting troughs
        trendlines = [{
            'type': 'support',
            'x1': points_per_section - 1,
            'y1': data[points_per_section - 1]['y'],
            'x2': points_per_section * 3 - 1,
            'y2': data[points_per_section * 3 - 1]['y'],
            'color': '#22c55e',
            'label': 'Support'
        }]
        
        annotations = [
            {'x': points_per_section - 1, 'y': data[points_per_section - 1]['y'], 'label': 'Trough 1', 'type': 'valley'},
            {'x': points_per_section * 3 - 1, 'y': data[points_per_section * 3 - 1]['y'], 'label': 'Trough 2', 'type': 'valley'},
        ]
        
        return data, trendlines, annotations
    
    def _generate_triple_top(self, base_price: float, num_points: int, volatility: float, scale: float = 1.0) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Generate triple top pattern"""
        data = []
        points_per_section = num_points // 7
        
        for peak_num in range(3):
            # Rise to peak
            for i in range(points_per_section):
                progress = i / points_per_section
                price = data[-1]['y'] + (base_price * 0.05 * progress) if data else base_price + (base_price * 0.05 * progress)
                price += random.uniform(-volatility, volatility) * base_price
                data.append({'x': len(data), 'y': price})
            
            # Decline from peak
            if peak_num < 2:
                for i in range(points_per_section):
                    progress = i / points_per_section
                    price = data[-1]['y'] - (base_price * 0.04 * progress)
                    price += random.uniform(-volatility, volatility) * base_price
                    data.append({'x': len(data), 'y': price})
        
        # Final breakdown
        for i in range(points_per_section):
            progress = i / points_per_section
            price = data[-1]['y'] - (base_price * 0.08 * progress)
            price += random.uniform(-volatility, volatility) * base_price
            data.append({'x': len(data), 'y': price})
        
        # Resistance line connecting peaks
        peak_indices = [points_per_section - 1, points_per_section * 3 - 1, points_per_section * 5 - 1]
        trendlines = [{
            'type': 'resistance',
            'x1': peak_indices[0],
            'y1': data[peak_indices[0]]['y'],
            'x2': peak_indices[2],
            'y2': data[peak_indices[2]]['y'],
            'color': '#ef4444',
            'label': 'Resistance'
        }]
        
        annotations = [
            {'x': peak_indices[0], 'y': data[peak_indices[0]]['y'], 'label': 'Peak 1', 'type': 'peak'},
            {'x': peak_indices[1], 'y': data[peak_indices[1]]['y'], 'label': 'Peak 2', 'type': 'peak'},
            {'x': peak_indices[2], 'y': data[peak_indices[2]]['y'], 'label': 'Peak 3', 'type': 'peak'},
        ]
        
        return data, trendlines, annotations
    
    def _generate_triple_bottom(self, base_price: float, num_points: int, volatility: float, scale: float = 1.0) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Generate triple bottom pattern"""
        data = []
        points_per_section = num_points // 7
        
        for trough_num in range(3):
            # Decline to trough
            for i in range(points_per_section):
                progress = i / points_per_section
                price = data[-1]['y'] - (base_price * 0.05 * progress) if data else base_price - (base_price * 0.05 * progress)
                price += random.uniform(-volatility, volatility) * base_price
                data.append({'x': len(data), 'y': price})
            
            # Rally from trough
            if trough_num < 2:
                for i in range(points_per_section):
                    progress = i / points_per_section
                    price = data[-1]['y'] + (base_price * 0.04 * progress)
                    price += random.uniform(-volatility, volatility) * base_price
                    data.append({'x': len(data), 'y': price})
        
        # Final breakout
        for i in range(points_per_section):
            progress = i / points_per_section
            price = data[-1]['y'] + (base_price * 0.08 * progress)
            price += random.uniform(-volatility, volatility) * base_price
            data.append({'x': len(data), 'y': price})
        
        # Support line connecting troughs
        trough_indices = [points_per_section - 1, points_per_section * 3 - 1, points_per_section * 5 - 1]
        trendlines = [{
            'type': 'support',
            'x1': trough_indices[0],
            'y1': data[trough_indices[0]]['y'],
            'x2': trough_indices[2],
            'y2': data[trough_indices[2]]['y'],
            'color': '#22c55e',
            'label': 'Support'
        }]
        
        annotations = [
            {'x': trough_indices[0], 'y': data[trough_indices[0]]['y'], 'label': 'Trough 1', 'type': 'valley'},
            {'x': trough_indices[1], 'y': data[trough_indices[1]]['y'], 'label': 'Trough 2', 'type': 'valley'},
            {'x': trough_indices[2], 'y': data[trough_indices[2]]['y'], 'label': 'Trough 3', 'type': 'valley'},
        ]
        
        return data, trendlines, annotations
    
    def _generate_ascending_triangle(self, base_price: float, num_points: int, volatility: float, scale: float = 1.0) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Generate ascending triangle with trendlines"""
        data = []
        resistance_level = base_price + (base_price * 0.05 * scale)
        
        for i in range(num_points):
            progress = i / num_points
            low_level = base_price + (base_price * 0.04 * scale * progress)
            
            if i % 10 < 5:
                price = low_level + (resistance_level - low_level) * ((i % 10) / 5)
            else:
                price = resistance_level - (resistance_level - low_level) * ((i % 10 - 5) / 5)
            
            price += random.uniform(-volatility, volatility) * base_price
            data.append({'x': i, 'y': price})
        
        # Breakout
        for i in range(10):
            price = data[-1]['y'] + (base_price * 0.01 * scale * i)
            data.append({'x': len(data), 'y': price})
        
        # Trendlines: flat resistance + rising support
        trendlines = [
            {
                'type': 'resistance',
                'x1': 0,
                'y1': resistance_level,
                'x2': num_points,
                'y2': resistance_level,
                'color': '#ef4444',
                'label': 'Resistance'
            },
            {
                'type': 'support',
                'x1': 0,
                'y1': base_price,
                'x2': num_points,
                'y2': base_price + (base_price * 0.04),
                'color': '#22c55e',
                'label': 'Rising Support'
            }
        ]
        
        annotations = [
            {'x': num_points, 'y': resistance_level, 'label': 'Breakout Zone', 'type': 'breakout'}
        ]
        
        return data, trendlines, annotations
    
    def _generate_descending_triangle(self, base_price: float, num_points: int, volatility: float, scale: float = 1.0) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Generate descending triangle"""
        data = []
        support_level = base_price - (base_price * 0.05 * scale)
        
        for i in range(num_points):
            progress = i / num_points
            high_level = base_price - (base_price * 0.04 * scale * progress)
            
            if i % 10 < 5:
                price = support_level + (high_level - support_level) * ((i % 10) / 5)
            else:
                price = high_level - (high_level - support_level) * ((i % 10 - 5) / 5)
            
            price += random.uniform(-volatility, volatility) * base_price
            data.append({'x': i, 'y': price})
        
        # Breakdown
        for i in range(10):
            price = data[-1]['y'] - (base_price * 0.01 * scale * i)
            data.append({'x': len(data), 'y': price})
        
        trendlines = [
            {
                'type': 'support',
                'x1': 0,
                'y1': support_level,
                'x2': num_points,
                'y2': support_level,
                'color': '#22c55e',
                'label': 'Support'
            },
            {
                'type': 'resistance',
                'x1': 0,
                'y1': base_price,
                'x2': num_points,
                'y2': base_price - (base_price * 0.04),
                'color': '#ef4444',
                'label': 'Falling Resistance'
            }
        ]
        
        annotations = []
        
        return data, trendlines, annotations
    
    def _generate_symmetrical_triangle(self, base_price: float, num_points: int, volatility: float, scale: float = 1.0) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Generate symmetrical triangle"""
        data = []
        
        for i in range(num_points):
            progress = i / num_points
            range_size = base_price * 0.05 * scale * (1 - progress)
            
            if i % 10 < 5:
                price = base_price + range_size * ((i % 10) / 5)
            else:
                price = base_price + range_size - (2 * range_size * ((i % 10 - 5) / 5))
            
            price += random.uniform(-volatility, volatility) * base_price
            data.append({'x': i, 'y': price})
        
        # Breakout
        for i in range(10):
            price = data[-1]['y'] + (base_price * 0.01 * scale * i)
            data.append({'x': len(data), 'y': price})
        
        trendlines = [
            {
                'type': 'resistance',
                'x1': 0,
                'y1': base_price + (base_price * 0.05),
                'x2': num_points,
                'y2': base_price,
                'color': '#ef4444',
                'label': 'Upper Trendline'
            },
            {
                'type': 'support',
                'x1': 0,
                'y1': base_price - (base_price * 0.05),
                'x2': num_points,
                'y2': base_price,
                'color': '#22c55e',
                'label': 'Lower Trendline'
            }
        ]
        
        annotations = [
            {'x': num_points * 0.75, 'y': base_price, 'label': 'Squeeze Zone', 'type': 'squeeze'}
        ]
        
        return data, trendlines, annotations
    
    def _generate_cup_and_handle(self, base_price: float, num_points: int, volatility: float, scale: float = 1.0) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Generate cup and handle"""
        data = []
        points_cup = int(num_points * 0.7)
        points_handle = num_points - points_cup
        
        # Cup (U-shape)
        for i in range(points_cup):
            progress = i / points_cup
            depth = -0.1 * scale * (4 * (progress - 0.5) ** 2 - 1)
            price = base_price + (base_price * depth) + random.uniform(-volatility, volatility) * base_price
            data.append({'x': len(data), 'y': price})
        
        # Handle (small dip)
        for i in range(points_handle):
            progress = i / points_handle
            if progress < 0.5:
                price = data[-1]['y'] - (base_price * 0.02 * progress)
            else:
                price = data[-1]['y'] + (base_price * 0.02 * (progress - 0.5))
            
            price += random.uniform(-volatility, volatility) * base_price
            data.append({'x': len(data), 'y': price})
        
        # Breakout
        for i in range(10):
            price = data[-1]['y'] + (base_price * 0.015 * scale * i)
            data.append({'x': len(data), 'y': price})
        
        # Neckline connecting the edges of the cup
        trendlines = [{
            'type': 'neckline',
            'x1': 0,
            'y1': base_price,
            'x2': points_cup,
            'y2': base_price,
            'color': '#22c55e',
            'label': 'Resistance'
        }]
        
        annotations = [
            {'x': points_cup // 2, 'y': base_price - (base_price * 0.1), 'label': 'Cup Bottom', 'type': 'valley'},
            {'x': points_cup + points_handle // 2, 'y': data[points_cup + points_handle // 2]['y'], 'label': 'Handle', 'type': 'handle'}
        ]
        
        return data, trendlines, annotations
    
    def _generate_inverse_cup_and_handle(self, base_price: float, num_points: int, volatility: float, scale: float = 1.0) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Generate inverse cup and handle"""
        data = []
        points_cup = int(num_points * 0.7)
        points_handle = num_points - points_cup
        
        # Inverted Cup
        for i in range(points_cup):
            progress = i / points_cup
            height = 0.1 * (4 * (progress - 0.5) ** 2 - 1)
            price = base_price - (base_price * height) + random.uniform(-volatility, volatility) * base_price
            data.append({'x': len(data), 'y': price})
        
        # Handle (small rally)
        for i in range(points_handle):
            progress = i / points_handle
            if progress < 0.5:
                price = data[-1]['y'] + (base_price * 0.02 * progress)
            else:
                price = data[-1]['y'] - (base_price * 0.02 * (progress - 0.5))
            
            price += random.uniform(-volatility, volatility) * base_price
            data.append({'x': len(data), 'y': price})
        
        # Breakdown
        for i in range(10):
            price = data[-1]['y'] - (base_price * 0.015 * i)
            data.append({'x': len(data), 'y': price})
        
        trendlines = []
        annotations = []
        
        return data, trendlines, annotations
    
    def _generate_falling_wedge(self, base_price: float, num_points: int, volatility: float, scale: float = 1.0) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Generate falling wedge"""
        data = []
        
        for i in range(num_points):
            progress = i / num_points
            high_level = base_price - (base_price * 0.03 * scale * progress)
            low_level = base_price - (base_price * 0.08 * scale * progress)
            
            if i % 8 < 4:
                price = low_level + (high_level - low_level) * ((i % 8) / 4)
            else:
                price = high_level - (high_level - low_level) * ((i % 8 - 4) / 4)
            
            price += random.uniform(-volatility, volatility) * base_price
            data.append({'x': i, 'y': price})
        
        # Breakout upward
        for i in range(10):
            price = data[-1]['y'] + (base_price * 0.015 * i)
            data.append({'x': len(data), 'y': price})
        
        trendlines = [
            {
                'type': 'resistance',
                'x1': 0,
                'y1': base_price,
                'x2': num_points,
                'y2': base_price - (base_price * 0.03),
                'color': '#ef4444',
                'label': 'Upper Wedge Line'
            },
            {
                'type': 'support',
                'x1': 0,
                'y1': base_price - (base_price * 0.05),
                'x2': num_points,
                'y2': base_price - (base_price * 0.08),
                'color': '#22c55e',
                'label': 'Lower Wedge Line'
            }
        ]
        
        annotations = []
        
        return data, trendlines, annotations
    
    def _generate_rising_wedge(self, base_price: float, num_points: int, volatility: float, scale: float = 1.0) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Generate rising wedge"""
        data = []
        
        for i in range(num_points):
            progress = i / num_points
            high_level = base_price + (base_price * 0.08 * scale * progress)
            low_level = base_price + (base_price * 0.03 * scale * progress)
            
            if i % 8 < 4:
                price = low_level + (high_level - low_level) * ((i % 8) / 4)
            else:
                price = high_level - (high_level - low_level) * ((i % 8 - 4) / 4)
            
            price += random.uniform(-volatility, volatility) * base_price
            data.append({'x': i, 'y': price})
        
        # Breakdown
        for i in range(10):
            price = data[-1]['y'] - (base_price * 0.015 * i)
            data.append({'x': len(data), 'y': price})
        
        trendlines = [
            {
                'type': 'resistance',
                'x1': 0,
                'y1': base_price + (base_price * 0.05),
                'x2': num_points,
                'y2': base_price + (base_price * 0.08),
                'color': '#ef4444',
                'label': 'Upper Wedge Line'
            },
            {
                'type': 'support',
                'x1': 0,
                'y1': base_price,
                'x2': num_points,
                'y2': base_price + (base_price * 0.03),
                'color': '#22c55e',
                'label': 'Lower Wedge Line'
            }
        ]
        
        annotations = []
        
        return data, trendlines, annotations
    
    def _generate_bull_flag(self, base_price: float, num_points: int, volatility: float, scale: float = 1.0) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Generate bull flag with pole and flag trendlines"""
        data = []
        pole_points = int(num_points * 0.3)
        flag_points = num_points - pole_points
        
        # Pole (sharp rise)
        for i in range(pole_points):
            progress = i / pole_points
            price = base_price + (base_price * 0.12 * scale * progress) + random.uniform(-volatility * scale, volatility * scale) * base_price
            data.append({'x': len(data), 'y': price})
        
        pole_top = data[-1]['y']
        flag_start_x = len(data)
        
        # Flag (slight downward consolidation)
        flag_start = data[-1]['y']
        for i in range(flag_points):
            progress = i / flag_points
            price = flag_start - (flag_start * 0.03 * scale * progress) + random.uniform(-volatility * scale, volatility * scale) * base_price
            data.append({'x': len(data), 'y': price})
        
        flag_end = data[-1]['y']
        flag_end_x = len(data)
        
        # Breakout continuation
        for i in range(10):
            price = data[-1]['y'] + (base_price * 0.02 * scale * i)
            data.append({'x': len(data), 'y': price})
        
        # Trendlines for flag channel
        trendlines = [
            {
                'type': 'flag_upper',
                'x1': flag_start_x,
                'y1': pole_top,
                'x2': flag_end_x,
                'y2': pole_top - (pole_top * 0.02),
                'color': '#f59e0b',
                'label': 'Flag Upper'
            },
            {
                'type': 'flag_lower',
                'x1': flag_start_x,
                'y1': pole_top - (pole_top * 0.01),
                'x2': flag_end_x,
                'y2': flag_end,
                'color': '#f59e0b',
                'label': 'Flag Lower'
            }
        ]
        
        annotations = [
            {'x': pole_points // 2, 'y': base_price + (base_price * 0.06), 'label': 'Pole', 'type': 'pole'},
            {'x': flag_start_x + flag_points // 2, 'y': pole_top - (pole_top * 0.015), 'label': 'Flag (Consolidation)', 'type': 'flag'}
        ]
        
        return data, trendlines, annotations
    
    def _generate_bear_flag(self, base_price: float, num_points: int, volatility: float, scale: float = 1.0) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Generate bear flag"""
        data = []
        pole_points = int(num_points * 0.3)
        flag_points = num_points - pole_points
        
        # Pole (sharp drop)
        for i in range(pole_points):
            progress = i / pole_points
            price = base_price - (base_price * 0.12 * scale * progress) + random.uniform(-volatility * scale, volatility * scale) * base_price
            data.append({'x': len(data), 'y': price})
        
        pole_bottom = data[-1]['y']
        flag_start_x = len(data)
        
        # Flag (slight upward consolidation)
        flag_start = data[-1]['y']
        for i in range(flag_points):
            progress = i / flag_points
            price = flag_start + (flag_start * 0.03 * scale * progress) + random.uniform(-volatility * scale, volatility * scale) * base_price
            data.append({'x': len(data), 'y': price})
        
        flag_end = data[-1]['y']
        flag_end_x = len(data)
        
        # Breakdown continuation
        for i in range(10):
            price = data[-1]['y'] - (base_price * 0.02 * scale * i)
            data.append({'x': len(data), 'y': price})
        
        trendlines = [
            {
                'type': 'flag_upper',
                'x1': flag_start_x,
                'y1': pole_bottom + (pole_bottom * 0.01),
                'x2': flag_end_x,
                'y2': flag_end,
                'color': '#f59e0b',
                'label': 'Flag Upper'
            },
            {
                'type': 'flag_lower',
                'x1': flag_start_x,
                'y1': pole_bottom,
                'x2': flag_end_x,
                'y2': pole_bottom + (pole_bottom * 0.02),
                'color': '#f59e0b',
                'label': 'Flag Lower'
            }
        ]
        
        annotations = []
        
        return data, trendlines, annotations
    
    def _generate_pennant(self, base_price: float, num_points: int, volatility: float, scale: float = 1.0) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Generate pennant"""
        data = []
        pole_points = int(num_points * 0.3)
        pennant_points = num_points - pole_points
        
        # Pole (sharp move)
        for i in range(pole_points):
            progress = i / pole_points
            price = base_price + (base_price * 0.1 * progress) + random.uniform(-volatility, volatility) * base_price
            data.append({'x': len(data), 'y': price})
        
        # Pennant (symmetrical triangle consolidation)
        pennant_start = data[-1]['y']
        for i in range(pennant_points):
            progress = i / pennant_points
            range_size = pennant_start * 0.03 * (1 - progress)
            
            if i % 6 < 3:
                price = pennant_start + range_size * ((i % 6) / 3)
            else:
                price = pennant_start + range_size - (2 * range_size * ((i % 6 - 3) / 3))
            
            price += random.uniform(-volatility, volatility) * base_price
            data.append({'x': i + pole_points, 'y': price})
        
        # Breakout
        for i in range(10):
            price = data[-1]['y'] + (base_price * 0.015 * i)
            data.append({'x': len(data), 'y': price})
        
        trendlines = []
        annotations = []
        
        return data, trendlines, annotations
    
    def _generate_rounding_bottom(self, base_price: float, num_points: int, volatility: float, scale: float = 1.0) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Generate rounding bottom"""
        data = []
        
        for i in range(num_points):
            progress = i / num_points
            depth = -0.12 * (4 * (progress - 0.5) ** 2 - 1)
            price = base_price + (base_price * depth) + random.uniform(-volatility, volatility) * base_price
            data.append({'x': i, 'y': price})
        
        # Continuation upward
        for i in range(10):
            price = data[-1]['y'] + (base_price * 0.01 * scale * i)
            data.append({'x': len(data), 'y': price})
        
        trendlines = []
        annotations = [
            {'x': num_points // 2, 'y': base_price - (base_price * 0.12), 'label': 'Bottoming Out', 'type': 'valley'}
        ]
        
        return data, trendlines, annotations
    
    def _generate_rounding_top(self, base_price: float, num_points: int, volatility: float, scale: float = 1.0) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Generate rounding top"""
        data = []
        
        for i in range(num_points):
            progress = i / num_points
            height = 0.12 * (4 * (progress - 0.5) ** 2 - 1)
            price = base_price - (base_price * height) + random.uniform(-volatility, volatility) * base_price
            data.append({'x': i, 'y': price})
        
        # Continuation downward
        for i in range(10):
            price = data[-1]['y'] - (base_price * 0.01 * i)
            data.append({'x': len(data), 'y': price})
        
        trendlines = []
        annotations = [
            {'x': num_points // 2, 'y': base_price + (base_price * 0.12), 'label': 'Topping Out', 'type': 'peak'}
        ]
        
        return data, trendlines, annotations
    
    def _generate_rectangle(self, base_price: float, num_points: int, volatility: float, scale: float = 1.0) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Generate rectangle consolidation"""
        data = []
        resistance = base_price * 1.03
        support = base_price * 0.97
        
        for i in range(num_points):
            if i % 8 < 4:
                price = support + (resistance - support) * ((i % 8) / 4)
            else:
                price = resistance - (resistance - support) * ((i % 8 - 4) / 4)
            
            price += random.uniform(-volatility, volatility) * base_price
            data.append({'x': i, 'y': price})
        
        # Breakout
        for i in range(10):
            price = data[-1]['y'] + (base_price * 0.01 * scale * i)
            data.append({'x': len(data), 'y': price})
        
        trendlines = [
            {
                'type': 'resistance',
                'x1': 0,
                'y1': resistance,
                'x2': num_points,
                'y2': resistance,
                'color': '#ef4444',
                'label': 'Resistance'
            },
            {
                'type': 'support',
                'x1': 0,
                'y1': support,
                'x2': num_points,
                'y2': support,
                'color': '#22c55e',
                'label': 'Support'
            }
        ]
        
        annotations = []
        
        return data, trendlines, annotations
    
    def _generate_broadening_top(self, base_price: float, num_points: int, volatility: float, scale: float = 1.0) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Generate broadening top (megaphone)"""
        data = []
        
        for i in range(num_points):
            progress = i / num_points
            range_size = base_price * 0.03 * (1 + progress)  # Expanding range
            
            if i % 10 < 5:
                price = base_price + range_size * ((i % 10) / 5)
            else:
                price = base_price + range_size - (2 * range_size * ((i % 10 - 5) / 5))
            
            price += random.uniform(-volatility, volatility) * base_price
            data.append({'x': i, 'y': price})
        
        # Breakdown
        for i in range(10):
            price = data[-1]['y'] - (base_price * 0.015 * i)
            data.append({'x': len(data), 'y': price})
        
        trendlines = []
        annotations = []
        
        return data, trendlines, annotations
    
    def _generate_broadening_bottom(self, base_price: float, num_points: int, volatility: float, scale: float = 1.0) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Generate broadening bottom"""
        data = []
        
        for i in range(num_points):
            progress = i / num_points
            range_size = base_price * 0.03 * (1 + progress)
            
            if i % 10 < 5:
                price = base_price - range_size * ((i % 10) / 5)
            else:
                price = base_price - range_size + (2 * range_size * ((i % 10 - 5) / 5))
            
            price += random.uniform(-volatility, volatility) * base_price
            data.append({'x': i, 'y': price})
        
        # Breakout
        for i in range(10):
            price = data[-1]['y'] + (base_price * 0.015 * i)
            data.append({'x': len(data), 'y': price})
        
        trendlines = []
        annotations = []
        
        return data, trendlines, annotations
    
    def _generate_diamond_top(self, base_price: float, num_points: int, volatility: float, scale: float = 1.0) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Generate diamond top"""
        data = []
        half_points = num_points // 2
        
        # Expanding phase
        for i in range(half_points):
            progress = i / half_points
            range_size = base_price * 0.05 * progress
            
            if i % 6 < 3:
                price = base_price + range_size * ((i % 6) / 3)
            else:
                price = base_price + range_size - (2 * range_size * ((i % 6 - 3) / 3))
            
            price += random.uniform(-volatility, volatility) * base_price
            data.append({'x': i, 'y': price})
        
        # Contracting phase
        for i in range(half_points):
            progress = i / half_points
            range_size = base_price * 0.05 * (1 - progress)
            
            if i % 6 < 3:
                price = base_price + range_size * ((i % 6) / 3)
            else:
                price = base_price + range_size - (2 * range_size * ((i % 6 - 3) / 3))
            
            price += random.uniform(-volatility, volatility) * base_price
            data.append({'x': i + half_points, 'y': price})
        
        # Breakdown
        for i in range(10):
            price = data[-1]['y'] - (base_price * 0.015 * i)
            data.append({'x': len(data), 'y': price})
        
        trendlines = []
        annotations = []
        
        return data, trendlines, annotations
    
    def _generate_diamond_bottom(self, base_price: float, num_points: int, volatility: float, scale: float = 1.0) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Generate diamond bottom"""
        data = []
        half_points = num_points // 2
        
        # Expanding phase
        for i in range(half_points):
            progress = i / half_points
            range_size = base_price * 0.05 * progress
            
            if i % 6 < 3:
                price = base_price - range_size * ((i % 6) / 3)
            else:
                price = base_price - range_size + (2 * range_size * ((i % 6 - 3) / 3))
            
            price += random.uniform(-volatility, volatility) * base_price
            data.append({'x': i, 'y': price})
        
        # Contracting phase
        for i in range(half_points):
            progress = i / half_points
            range_size = base_price * 0.05 * (1 - progress)
            
            if i % 6 < 3:
                price = base_price - range_size * ((i % 6) / 3)
            else:
                price = base_price - range_size + (2 * range_size * ((i % 6 - 3) / 3))
            
            price += random.uniform(-volatility, volatility) * base_price
            data.append({'x': i + half_points, 'y': price})
        
        # Breakout
        for i in range(10):
            price = data[-1]['y'] + (base_price * 0.015 * i)
            data.append({'x': len(data), 'y': price})
        
        trendlines = []
        annotations = []
        
        return data, trendlines, annotations
    
    def _get_pattern_description(self, pattern_name: str) -> str:
        """Get educational description of a pattern"""
        descriptions = {
            'head_and_shoulders': 'A bearish reversal pattern with three peaks - a higher middle peak (head) between two lower peaks (shoulders). The neckline acts as critical support.',
            'inverse_head_and_shoulders': 'A bullish reversal pattern with three troughs - a lower middle trough (head) between two higher troughs (shoulders). Breakout above neckline confirms.',
            'double_top': 'A bearish reversal pattern where price tests resistance twice and fails, forming an M shape. Breakdown below support confirms the pattern.',
            'double_bottom': 'A bullish reversal pattern where price tests support twice and holds, forming a W shape. Breakout above resistance confirms the pattern.',
            'triple_top': 'A bearish reversal with three peaks at similar levels. Stronger signal than double top due to multiple rejections at resistance.',
            'triple_bottom': 'A bullish reversal with three troughs at similar levels. Stronger signal than double bottom due to multiple bounces from support.',
            'ascending_triangle': 'A bullish continuation pattern with flat resistance and rising support. Indicates accumulation before upward breakout.',
            'descending_triangle': 'A bearish continuation pattern with flat support and declining resistance. Indicates distribution before downward breakdown.',
            'symmetrical_triangle': 'A continuation pattern with converging trendlines. Breakout direction typically follows prior trend. Squeeze zone near apex.',
            'cup_and_handle': 'A bullish continuation pattern resembling a teacup. The cup forms a rounded bottom, followed by a small handle consolidation.',
            'inverse_cup_and_handle': 'A bearish continuation pattern - inverted teacup shape. Dome top followed by small rally before breakdown.',
            'falling_wedge': 'A bullish reversal with converging trendlines sloping down. Price squeezes into apex before upward breakout.',
            'rising_wedge': 'A bearish reversal with converging trendlines sloping up. Price squeezes into apex before downward breakdown.',
            'bull_flag': 'A bullish continuation with sharp rally (pole) followed by downward-sloping consolidation (flag). Breakout continues uptrend.',
            'bear_flag': 'A bearish continuation with sharp drop (pole) followed by upward-sloping consolidation (flag). Breakdown continues downtrend.',
            'pennant': 'A continuation pattern similar to flag but with symmetrical triangle consolidation. Brief pause before trend continuation.',
            'rounding_bottom': 'A bullish reversal forming smooth U-shape. Gradual shift from selling to buying pressure. Also called saucer bottom.',
            'rounding_top': 'A bearish reversal forming smooth dome. Gradual shift from buying to selling pressure. Also called inverted saucer.',
            'rectangle': 'A continuation pattern with horizontal support and resistance. Price consolidates in range before breakout.',
            'broadening_top': 'A bearish pattern with expanding price swings. Also called megaphone. Indicates increasing volatility and uncertainty.',
            'broadening_bottom': 'A bullish pattern with expanding price swings at bottom. Volatility increases before upward resolution.',
            'diamond_top': 'A bearish reversal combining broadening and narrowing phases. Rare but reliable pattern at market tops.',
            'diamond_bottom': 'A bullish reversal combining broadening and narrowing phases. Rare but reliable pattern at market bottoms.'
        }
        return descriptions.get(pattern_name, 'Chart pattern visualization')
    
    def _get_trading_tips(self, pattern_name: str) -> List[str]:
        """Get trading tips for a pattern"""
        tips = {
            'head_and_shoulders': [
                'Enter short on neckline breakdown',
                'Place stop above right shoulder',
                'Target = neckline - (head - neckline)',
                'Volume should increase on breakdown'
            ],
            'bull_flag': [
                'Enter long on flag breakout',
                'Place stop below flag support',
                'Target = pole height added to breakout',
                'Flag should form in 1-4 weeks'
            ],
            'ascending_triangle': [
                'Enter long on resistance breakout',
                'Place stop below rising support',
                'Target = triangle height added to breakout',
                'Watch for volume expansion on breakout'
            ],
        }
        return tips.get(pattern_name, [
            'Wait for confirmed breakout/breakdown',
            'Use volume to confirm pattern validity',
            'Set stop loss beyond pattern boundaries',
            'Measure target using pattern height'
        ])
    
    def _extract_key_levels(self, data: List[Dict], pattern_name: str) -> Dict:
        """Extract key support/resistance levels from pattern data"""
        if not data:
            return {}
        
        prices = [point['y'] for point in data]
        
        return {
            'high': max(prices),
            'low': min(prices),
            'current': prices[-1] if prices else 0,
            'breakout_level': self._get_breakout_level(data, pattern_name),
            'midpoint': (max(prices) + min(prices)) / 2
        }
    
    def _get_breakout_level(self, data: List[Dict], pattern_name: str) -> Optional[float]:
        """Determine the breakout level for a pattern"""
        if not data:
            return None
        
        prices = [point['y'] for point in data]
        
        # Pattern-specific breakout levels
        if 'top' in pattern_name or pattern_name in ['head_and_shoulders', 'rising_wedge', 'broadening_top', 'diamond_top']:
            return min(prices[:len(prices)//2])
        else:
            return max(prices[:len(prices)//2])


# Singleton instance
pattern_service = PatternRecognitionService()
