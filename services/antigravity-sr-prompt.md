# Support/Resistance & Trendline Detection Feature Request

## Overview
I need to add support/resistance zone detection and trendline drawing capabilities to my Nunno crypto prediction app. This should work across both the **Python backend** (betterpredictormodule.py) and the **React/Vite frontend**.

---

## Backend Requirements (Python)

### 1. Add Support/Resistance Detection to `TradingAnalyzer` class

Create a method that:
- Detects swing highs and lows using `scipy.signal.argrelextrema`
- Clusters nearby price levels (within 0.3-0.5% tolerance) to find zones
- Counts how many times price "touched" each level (came within tolerance and reversed)
- Validates levels requiring minimum 2-3 touches
- Weights recent touches more heavily (decay factor for older touches)
- Returns structured data with:
  - Price level
  - Type (support/resistance)
  - Touch count
  - First/last touch timestamps
  - Strength rating (weak/medium/strong based on touches)
  - All touch points with timestamps

**Key Parameters:**
- `min_touches`: 2 (default)
- `tolerance_pct`: 0.5% (default)
- `lookback_period`: 100-200 candles
- `order`: 3-5 (candles on each side for swing detection)

### 2. Add Trendline Detection

Create a method that:
- Finds consecutive swing lows (uptrend) and swing highs (downtrend)
- Uses **RANSAC regression** to fit lines through swing points (ignores outliers)
- Validates trendlines by:
  - Minimum 3 touches
  - Angle between 10-45 degrees (reject too steep/flat)
  - Recent validity (still respected in last 10-20 candles)
  - Distance tolerance (0.3-0.5% from perfect line)
- Detects parallel channels (opposite side trendline)
- Returns structured data with:
  - Type (uptrend/downtrend/channel)
  - Start/end points with timestamps
  - Slope and intercept (equation)
  - Touch count
  - All touch points
  - Validity score

### 3. Integration with Existing Analysis

In `generate_comprehensive_analysis()`:
- Call S/R and trendline detection on the dataframe
- Add detected levels to the confluences system
- Weight S/R zones and trendlines in the bias calculation:
  - Strong support near current price → bullish confluence
  - Strong resistance near current price → bearish confluence
  - Respected uptrend → bullish confluence
  - Respected downtrend → bearish confluence

**Indicator weights to add:**
```python
'Support Zone': 1.4,
'Resistance Zone': 1.4,
'Uptrend Line': 1.3,
'Downtrend Line': 1.3,
'Channel': 1.5,
```

### 4. Return Format in Analysis Results

Modify the `analyze_token()` return structure to include:

```python
{
  "technical": {
    "current_price": 45234.56,
    "bias": "Bullish",
    "strength": 72.3,
    
    # NEW: Add these
    "support_resistance": [
      {
        "type": "support",
        "price": 44800,
        "touches": 4,
        "strength": "strong",
        "first_touch": "2024-01-05T10:30:00",
        "last_touch": "2024-01-15T14:20:00",
        "distance_from_current": -0.96,  # percentage
        "touch_points": [
          {"time": "2024-01-05T10:30:00", "price": 44750},
          {"time": "2024-01-08T16:45:00", "price": 44820},
          # ...
        ]
      },
      # ... more levels
    ],
    
    "trendlines": [
      {
        "type": "uptrend",
        "start": {"time": "2024-01-01T00:00:00", "price": 43500},
        "end": {"time": "2024-01-15T23:59:59", "price": 45800},
        "slope": 0.0234,
        "intercept": 43500,
        "touches": 5,
        "angle_degrees": 23.5,
        "validity_score": 0.89,
        "touch_points": [...]
      },
      # ... more trendlines
    ],
    
    "channels": [
      {
        "upper_trendline": {...},
        "lower_trendline": {...},
        "width": 1200,  # price difference
        "position": "inside"  # current price position
      }
    ]
  }
}
```

---

## Frontend Requirements (React/Vite)

### 1. Install Lightweight Charts
```bash
npm install lightweight-charts
```

### 2. Create Enhanced Chart Component

Build a component that:
- Renders candlestick chart using Lightweight Charts
- Overlays horizontal lines for S/R zones with:
  - Green dashed lines for support
  - Red dashed lines for resistance
  - Thickness based on strength (2px weak, 3px medium, 4px strong)
  - Labels showing price and touch count
- Draws trendlines with:
  - Blue for uptrends
  - Orange for downtrends
  - Purple for channels (two parallel lines)
  - Touch points marked with small circles
- Interactive features:
  - Hover tooltips showing level details (touches, strength, dates)
  - Click to highlight/unhighlight specific levels
  - Toggle to show/hide S/R zones, trendlines, or both
  - Legend explaining line colors and types

### 3. Component Props Structure

```typescript
interface SupportResistanceLevel {
  type: 'support' | 'resistance';
  price: number;
  touches: number;
  strength: 'weak' | 'medium' | 'strong';
  firstTouch: string;
  lastTouch: string;
  distanceFromCurrent: number;
  touchPoints: Array<{time: string; price: number}>;
}

interface Trendline {
  type: 'uptrend' | 'downtrend';
  start: {time: string; price: number};
  end: {time: string; price: number};
  slope: number;
  touches: number;
  touchPoints: Array<{time: string; price: number}>;
}

interface ChartProps {
  ohlcvData: Array<{time: string; open: number; high: number; low: number; close: number; volume: number}>;
  supportResistance: SupportResistanceLevel[];
  trendlines: Trendline[];
  channels?: Channel[];
  currentPrice: number;
}
```

### 4. Visual Design Specifications

**Support/Resistance Lines:**
- Support: `#26a69a` (green), dashed, label on right
- Resistance: `#ef5350` (red), dashed, label on right
- Zone shading: semi-transparent background between closely clustered levels
- Touch points: small circles (`○`) on the line

**Trendlines:**
- Uptrend: `#2196f3` (blue), solid, 2px
- Downtrend: `#ff9800` (orange), solid, 2px
- Channel: `#9c27b0` (purple), dotted, 1.5px for parallel lines
- Touch points: slightly larger circles (`●`) at exact contact

**Legend:**
- Top-right corner, collapsible
- Shows what each line type represents
- Toggle switches for visibility

### 5. Integration with Existing Nunno App

- Add chart to the technical analysis section
- Position below the bias/strength display
- Sync with current token and timeframe selection
- Update in real-time when user changes inputs
- Show loading state while fetching new data
- Error handling if S/R detection fails

---

## Testing Requirements

### Backend Tests:
1. Verify S/R detection on BTC 15m data (should find 3-5 major levels)
2. Verify trendline detection on trending vs ranging markets
3. Test with different timeframes (1h, 4h, 1d)
4. Edge cases: low volatility, extreme volatility, gaps

### Frontend Tests:
1. Render chart with multiple S/R zones correctly
2. Trendlines extend properly across timeframe
3. Tooltips show accurate touch information
4. Toggle controls work smoothly
5. Performance with 1000+ candles + 10+ lines
6. Responsive on mobile/tablet

---

## Libraries & Dependencies

**Python (add to requirements.txt):**
```
scipy>=1.10.0  # for argrelextrema
scikit-learn>=1.3.0  # for RANSAC
```

**React (package.json):**
```json
{
  "lightweight-charts": "^4.1.0"
}
```

---

## Expected Deliverables

1. ✅ Modified `betterpredictormodule.py` with new methods
2. ✅ Updated return structure in `analyze_token()`
3. ✅ New React component: `TechnicalChart.tsx` (or `.jsx`)
4. ✅ Integration code to connect component to Nunno app
5. ✅ CSS/styling for chart and controls
6. ✅ Documentation on how to interpret the lines

---

## Example Output

When analyzing BTCUSDT on 15m timeframe, should return something like:

**Console:**
```
📊 Detected 3 Support Zones and 2 Resistance Zones
   Support: $44,800 (4 touches, strong)
   Support: $43,200 (2 touches, medium)
   Resistance: $46,500 (3 touches, strong)

📈 Detected 1 Uptrend Line
   Uptrend: 5 touches, 23.5° angle, valid

⚖️ Channel Detected
   Range: $44,000 - $46,000
```

**Frontend:**
- Clean candlestick chart with visible green/red lines at those prices
- Blue uptrend line connecting the swing lows
- Purple channel boundaries
- Current price marker showing position relative to levels

---

## Priority & Timeline

**Priority:** High (enhances core prediction accuracy visualization)
**Estimated Complexity:** Medium-High
**Suggested Timeline:** 
- Backend: 3-4 hours
- Frontend: 4-5 hours
- Testing & Integration: 2 hours

---

## Additional Notes

- Make S/R detection configurable (users might want to adjust sensitivity)
- Consider adding "zones" (ranges) instead of just single price lines for stronger levels
- Future enhancement: Allow users to manually draw their own lines (save to preferences)
- Ensure lines update dynamically when new candles arrive (real-time mode)

---

## Questions to Address

1. Should we show ALL detected levels or only the strongest 3-5?
2. How far back should trendlines extend? (current visible range vs full dataset)
3. Should broken trendlines be shown differently (faded out)?
4. Do you want notifications when price approaches a major S/R level?

Please implement this feature maintaining the existing code style and structure of the Nunno app. Prioritize clean, readable code with proper error handling.