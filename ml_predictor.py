import json
import os
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
from collections import defaultdict

class PredictionTracker:
    """
    Tracks predictions and validates them against actual market movements.
    Learns from successes/failures to improve future predictions.
    """
    
    def __init__(self, storage_path="predictions_db.json"):
        self.storage_path = storage_path
        self.predictions = self._load_predictions()
        
        # Timeframe validation delays (when to check if prediction was correct)
        self.validation_delays = {
            "1m": timedelta(minutes=5),
            "3m": timedelta(minutes=15),
            "5m": timedelta(minutes=25),
            "15m": timedelta(hours=1),
            "30m": timedelta(hours=2),
            "1h": timedelta(hours=4),
            "2h": timedelta(hours=8),
            "4h": timedelta(hours=16),
            "6h": timedelta(hours=24),
            "12h": timedelta(days=2),
            "1d": timedelta(days=5)
        }
        
        # Learning weights - how much each indicator contributes to success
        self.indicator_weights = self._load_indicator_weights()
        
        # Support/Resistance zone tracker
        self.sr_zones = self._load_sr_zones()
        
    def _load_predictions(self):
        """Load prediction history from disk"""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Could not load predictions: {e}")
                return {"predictions": [], "stats": {}}
        return {"predictions": [], "stats": {}}
    
    def _save_predictions(self):
        """Save predictions to disk"""
        try:
            with open(self.storage_path, 'w') as f:
                json.dump(self.predictions, f, indent=2)
        except Exception as e:
            print(f"⚠️ Could not save predictions: {e}")
    
    def _load_indicator_weights(self):
        """Load learned indicator weights"""
        weights_path = "indicator_weights.json"
        if os.path.exists(weights_path):
            try:
                with open(weights_path, 'r') as f:
                    return json.load(f)
            except:
                pass
        
        # Default weights - all indicators start equal
        return {
            "RSI_14": 1.0,
            "Stochastic": 1.0,
            "Williams_R": 1.0,
            "EMA_Alignment": 1.0,
            "MACD": 1.0,
            "ADX": 1.0,
            "Bollinger_Bands": 1.0,
            "Volume": 1.0,
            "CMF": 1.0,
            "Price_Action": 1.0
        }
    
    def _save_indicator_weights(self):
        """Save updated indicator weights"""
        try:
            with open("indicator_weights.json", 'w') as f:
                json.dump(self.indicator_weights, f, indent=2)
        except Exception as e:
            print(f"⚠️ Could not save weights: {e}")
    
    def _load_sr_zones(self):
        """Load learned support/resistance zones"""
        sr_path = "sr_zones.json"
        if os.path.exists(sr_path):
            try:
                with open(sr_path, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {}
    
    def _save_sr_zones(self):
        """Save support/resistance zones"""
        try:
            with open("sr_zones.json", 'w') as f:
                json.dump(self.sr_zones, f, indent=2)
        except Exception as e:
            print(f"⚠️ Could not save SR zones: {e}")
    
    def save_prediction(self, token, timeframe, bias, strength, current_price, 
                       confluences, key_levels, technical_data):
        """
        Save a new prediction for future validation
        """
        prediction = {
            "id": f"{token}_{timeframe}_{datetime.now().timestamp()}",
            "token": token,
            "timeframe": timeframe,
            "timestamp": datetime.now().isoformat(),
            "bias": bias,
            "strength": strength,
            "entry_price": current_price,
            "key_levels": key_levels,
            "confluences": {
                "bullish_count": len(confluences.get("bullish", [])),
                "bearish_count": len(confluences.get("bearish", [])),
                "neutral_count": len(confluences.get("neutral", [])),
                "bullish_indicators": [c["indicator"] for c in confluences.get("bullish", [])],
                "bearish_indicators": [c["indicator"] for c in confluences.get("bearish", [])]
            },
            "technical_snapshot": {
                "RSI_14": technical_data.get("RSI_14"),
                "MACD": technical_data.get("MACD"),
                "ADX": technical_data.get("ADX"),
                "BB_Position": technical_data.get("BB_Position"),
                "Volume_Ratio": technical_data.get("Volume_Ratio"),
                "CMF": technical_data.get("CMF")
            },
            "validation": {
                "validated": False,
                "validation_time": (datetime.now() + self.validation_delays.get(timeframe, timedelta(hours=4))).isoformat(),
                "result": None,
                "actual_move": None,
                "accuracy": None
            }
        }
        
        self.predictions["predictions"].append(prediction)
        self._save_predictions()
        
        print(f"\n💾 Prediction saved! Will validate after {self.validation_delays.get(timeframe)}")
        return prediction["id"]
    
    def validate_predictions(self, analyzer):
        """
        Check all pending predictions and validate them against actual market data
        """
        now = datetime.now()
        validated_count = 0
        
        print("\n🔍 Validating pending predictions...")
        
        for pred in self.predictions["predictions"]:
            # Skip already validated predictions
            if pred["validation"]["validated"]:
                continue
            
            # Check if it's time to validate
            validation_time = datetime.fromisoformat(pred["validation"]["validation_time"])
            if now < validation_time:
                continue
            
            # Fetch current market data
            try:
                token = pred["token"]
                timeframe = pred["timeframe"]
                
                print(f"   Validating {token} ({timeframe}) from {pred['timestamp'][:16]}...")
                
                # Get recent OHLCV data
                df = analyzer.fetch_binance_ohlcv(symbol=token, interval=timeframe, limit=100)
                
                if df is None or len(df) == 0:
                    print(f"   ⚠️ Could not fetch data for validation")
                    continue
                
                # Calculate actual price movement
                entry_price = pred["entry_price"]
                current_price = df.iloc[-1]["Close"]
                price_change_pct = ((current_price - entry_price) / entry_price) * 100
                
                # Determine if prediction was correct
                predicted_bias = pred["bias"]
                was_correct = False
                
                # Check if movement matched prediction
                if predicted_bias == "Bullish Bias" and price_change_pct > 0.5:
                    was_correct = True
                elif predicted_bias == "Bearish Bias" and price_change_pct < -0.5:
                    was_correct = True
                elif predicted_bias == "Mixed/Neutral" and abs(price_change_pct) < 0.5:
                    was_correct = True
                
                # Calculate accuracy score (stronger predictions should be more accurate)
                predicted_strength = pred["strength"]
                accuracy_score = 0
                
                if was_correct:
                    # Reward based on strength and magnitude
                    accuracy_score = min(100, predicted_strength * abs(price_change_pct) / 2)
                else:
                    # Penalty based on how wrong it was
                    accuracy_score = max(0, 50 - abs(price_change_pct) * 10)
                
                # Update prediction with validation results
                pred["validation"]["validated"] = True
                pred["validation"]["result"] = "correct" if was_correct else "incorrect"
                pred["validation"]["actual_move"] = price_change_pct
                pred["validation"]["accuracy"] = accuracy_score
                pred["validation"]["actual_validation_time"] = now.isoformat()
                pred["validation"]["final_price"] = current_price
                
                validated_count += 1
                
                emoji = "✅" if was_correct else "❌"
                print(f"   {emoji} {pred['bias']}: {price_change_pct:+.2f}% (Score: {accuracy_score:.1f})")
                
                # Learn from this prediction
                self._learn_from_prediction(pred)
                
                # Update SR zones based on price action
                self._update_sr_zones(pred, df)
                
            except Exception as e:
                print(f"   ⚠️ Validation error: {e}")
                continue
        
        if validated_count > 0:
            self._save_predictions()
            self._save_indicator_weights()
            self._save_sr_zones()
            self._update_stats()
            print(f"\n✅ Validated {validated_count} predictions")
        else:
            print(f"\n📊 No predictions ready for validation yet")
        
        return validated_count
    
    def _learn_from_prediction(self, pred):
        """
        Adjust indicator weights based on prediction outcome
        """
        was_correct = pred["validation"]["result"] == "correct"
        accuracy = pred["validation"]["accuracy"]
        
        # Learning rate - how much to adjust weights
        learning_rate = 0.05 if was_correct else -0.03
        
        # Adjust weights for indicators that participated in this prediction
        bias = pred["bias"]
        
        if bias == "Bullish Bias":
            indicators = pred["confluences"]["bullish_indicators"]
        elif bias == "Bearish Bias":
            indicators = pred["confluences"]["bearish_indicators"]
        else:
            return  # Don't learn from neutral predictions
        
        # Update weights for involved indicators
        for indicator in indicators:
            # Normalize indicator names to match weight keys
            weight_key = indicator.replace(" ", "_").replace("-", "_")
            
            # Find closest matching key
            for key in self.indicator_weights.keys():
                if key.lower() in weight_key.lower() or weight_key.lower() in key.lower():
                    old_weight = self.indicator_weights[key]
                    
                    # Adjust weight based on accuracy
                    adjustment = learning_rate * (accuracy / 100)
                    new_weight = max(0.1, min(2.0, old_weight + adjustment))
                    
                    self.indicator_weights[key] = new_weight
                    break
    
    def _update_sr_zones(self, pred, df):
        """
        Learn support and resistance zones from price action.
        Updates zones based on whether they held or broke.
        """
        token = pred["token"]
        timeframe = pred["timeframe"]
        
        # Initialize token SR zones if not exists
        zone_key = f"{token}_{timeframe}"
        if zone_key not in self.sr_zones:
            self.sr_zones[zone_key] = {
                "support_zones": [],
                "resistance_zones": []
            }
        
        # Get price levels from the validation period
        high_prices = df["High"].values
        low_prices = df["Low"].values
        close_prices = df["Close"].values
        
        entry_price = pred["entry_price"]
        final_price = pred["validation"]["final_price"]
        price_moved_up = final_price > entry_price
        
        # Identify zones where price consolidated or reversed
        for i in range(1, len(df) - 1):
            current_high = high_prices[i]
            current_low = low_prices[i]
            current_close = close_prices[i]
            
            # Check for resistance (price rejected from above)
            if i > 2:
                # Price approached a level multiple times but couldn't break
                prev_highs = high_prices[max(0, i-3):i]
                if len(prev_highs) > 0 and abs(current_high - np.max(prev_highs)) / current_high < 0.005:
                    # Found potential resistance zone
                    resistance_level = current_high
                    
                    # Check if this zone held during validation
                    zone_held = final_price < resistance_level
                    
                    self._add_or_update_zone(
                        zone_key, "resistance", resistance_level, 
                        zone_held, current_close, timeframe
                    )
            
            # Check for support (price rejected from below)
            if i > 2:
                prev_lows = low_prices[max(0, i-3):i]
                if len(prev_lows) > 0 and abs(current_low - np.min(prev_lows)) / current_low < 0.005:
                    # Found potential support zone
                    support_level = current_low
                    
                    # Check if this zone held during validation
                    zone_held = final_price > support_level
                    
                    self._add_or_update_zone(
                        zone_key, "support", support_level,
                        zone_held, current_close, timeframe
                    )
        
        # Also learn from key levels that were tested
        if "key_levels" in pred:
            support_level = pred["key_levels"]["support"]
            resistance_level = pred["key_levels"]["resistance"]
            
            # Check if support held
            min_price = np.min(low_prices)
            if abs(min_price - support_level) / support_level < 0.01:
                zone_held = final_price > support_level
                self._add_or_update_zone(zone_key, "support", support_level, zone_held, final_price, timeframe)
            
            # Check if resistance held
            max_price = np.max(high_prices)
            if abs(max_price - resistance_level) / resistance_level < 0.01:
                zone_held = final_price < resistance_level
                self._add_or_update_zone(zone_key, "resistance", resistance_level, zone_held, final_price, timeframe)
        
        # Clean up old or weak zones
        self._cleanup_sr_zones(zone_key)
    
    def _add_or_update_zone(self, zone_key, zone_type, price_level, held, current_price, timeframe):
        """
        Add new SR zone or update existing one with test results
        """
        zones_list = self.sr_zones[zone_key][f"{zone_type}_zones"]
        
        # Check if similar zone already exists (within 1%)
        existing_zone = None
        for zone in zones_list:
            if abs(zone["price"] - price_level) / price_level < 0.01:
                existing_zone = zone
                break
        
        if existing_zone:
            # Update existing zone
            existing_zone["tests"] += 1
            if held:
                existing_zone["holds"] += 1
            else:
                existing_zone["breaks"] += 1
            
            # Update strength based on hold rate
            hold_rate = existing_zone["holds"] / existing_zone["tests"]
            existing_zone["strength"] = hold_rate
            
            # Update price to weighted average
            weight = existing_zone["tests"]
            existing_zone["price"] = (existing_zone["price"] * (weight - 1) + price_level) / weight
            
            existing_zone["last_tested"] = datetime.now().isoformat()
            
        else:
            # Create new zone
            new_zone = {
                "price": price_level,
                "strength": 1.0 if held else 0.5,
                "tests": 1,
                "holds": 1 if held else 0,
                "breaks": 0 if held else 1,
                "created": datetime.now().isoformat(),
                "last_tested": datetime.now().isoformat(),
                "timeframe": timeframe
            }
            zones_list.append(new_zone)
    
    def _cleanup_sr_zones(self, zone_key, max_zones=10, min_tests=2, min_strength=0.4):
        """
        Remove weak or irrelevant SR zones
        """
        for zone_type in ["support_zones", "resistance_zones"]:
            zones = self.sr_zones[zone_key][zone_type]
            
            # Filter out weak zones
            strong_zones = [
                z for z in zones 
                if z["tests"] >= min_tests and z["strength"] >= min_strength
            ]
            
            # Also keep recently created zones (give them a chance)
            recent_threshold = datetime.now() - timedelta(days=7)
            recent_zones = [
                z for z in zones
                if datetime.fromisoformat(z["created"]) > recent_threshold and z not in strong_zones
            ]
            
            # Combine and sort by strength
            combined = strong_zones + recent_zones
            combined.sort(key=lambda x: x["strength"], reverse=True)
            
            # Keep top zones
            self.sr_zones[zone_key][zone_type] = combined[:max_zones]
    
    def get_ml_sr_levels(self, token, timeframe, current_price, lookback_timeframes=None):
        """
        Get ML-learned support and resistance levels for a token.
        Can optionally check multiple timeframes for confluence.
        """
        if lookback_timeframes is None:
            lookback_timeframes = [timeframe]
        
        all_support = []
        all_resistance = []
        
        for tf in lookback_timeframes:
            zone_key = f"{token}_{tf}"
            
            if zone_key not in self.sr_zones:
                continue
            
            # Get zones for this timeframe
            support_zones = self.sr_zones[zone_key].get("support_zones", [])
            resistance_zones = self.sr_zones[zone_key].get("resistance_zones", [])
            
            # Filter for relevant zones (near current price)
            price_range = current_price * 0.10  # Within 10%
            
            relevant_support = [
                z for z in support_zones
                if current_price - price_range < z["price"] < current_price
            ]
            
            relevant_resistance = [
                z for z in resistance_zones
                if current_price < z["price"] < current_price + price_range
            ]
            
            all_support.extend(relevant_support)
            all_resistance.extend(relevant_resistance)
        
        # Cluster nearby zones (within 0.5%)
        clustered_support = self._cluster_zones(all_support, current_price)
        clustered_resistance = self._cluster_zones(all_resistance, current_price)
        
        # Sort by strength
        clustered_support.sort(key=lambda x: x["strength"], reverse=True)
        clustered_resistance.sort(key=lambda x: x["strength"], reverse=True)
        
        return {
            "support": clustered_support[:5],  # Top 5 support levels
            "resistance": clustered_resistance[:5],  # Top 5 resistance levels
            "nearest_support": clustered_support[0] if clustered_support else None,
            "nearest_resistance": clustered_resistance[0] if clustered_resistance else None
        }
    
    def _cluster_zones(self, zones, current_price, threshold=0.005):
        """
        Cluster nearby zones into stronger combined zones
        """
        if not zones:
            return []
        
        clustered = []
        zones_sorted = sorted(zones, key=lambda x: x["price"])
        
        current_cluster = [zones_sorted[0]]
        
        for i in range(1, len(zones_sorted)):
            zone = zones_sorted[i]
            last_zone = current_cluster[-1]
            
            # Check if within threshold
            if abs(zone["price"] - last_zone["price"]) / current_price < threshold:
                current_cluster.append(zone)
            else:
                # Finalize current cluster
                clustered.append(self._merge_cluster(current_cluster))
                current_cluster = [zone]
        
        # Don't forget last cluster
        if current_cluster:
            clustered.append(self._merge_cluster(current_cluster))
        
        return clustered
    
    def _merge_cluster(self, cluster):
        """
        Merge multiple zones into one strong zone
        """
        if len(cluster) == 1:
            return cluster[0]
        
        # Weighted average price
        total_strength = sum(z["strength"] * z["tests"] for z in cluster)
        total_weight = sum(z["tests"] for z in cluster)
        
        avg_price = sum(z["price"] * z["strength"] * z["tests"] for z in cluster) / total_strength
        
        # Combined strength (confluence makes it stronger)
        base_strength = sum(z["strength"] for z in cluster) / len(cluster)
        confluence_bonus = min(0.3, len(cluster) * 0.1)  # Up to 30% bonus
        combined_strength = min(1.0, base_strength + confluence_bonus)
        
        return {
            "price": avg_price,
            "strength": combined_strength,
            "tests": sum(z["tests"] for z in cluster),
            "holds": sum(z["holds"] for z in cluster),
            "breaks": sum(z["breaks"] for z in cluster),
            "timeframes": list(set(z["timeframe"] for z in cluster)),
            "confluence_count": len(cluster),
            "last_tested": max(z["last_tested"] for z in cluster)
        }
    
    def _update_stats(self):
        """Update global statistics"""
        validated = [p for p in self.predictions["predictions"] if p["validation"]["validated"]]
        
        if not validated:
            return
        
        total = len(validated)
        correct = len([p for p in validated if p["validation"]["result"] == "correct"])
        
        # Calculate stats by timeframe and bias
        stats = {
            "total_predictions": len(self.predictions["predictions"]),
            "validated_predictions": total,
            "pending_predictions": len(self.predictions["predictions"]) - total,
            "overall_accuracy": (correct / total * 100) if total > 0 else 0,
            "by_timeframe": {},
            "by_bias": {},
            "avg_accuracy_score": np.mean([p["validation"]["accuracy"] for p in validated])
        }
        
        # Group by timeframe
        for tf in set(p["timeframe"] for p in validated):
            tf_preds = [p for p in validated if p["timeframe"] == tf]
            tf_correct = len([p for p in tf_preds if p["validation"]["result"] == "correct"])
            stats["by_timeframe"][tf] = {
                "total": len(tf_preds),
                "correct": tf_correct,
                "accuracy": (tf_correct / len(tf_preds) * 100) if tf_preds else 0
            }
        
        # Group by bias
        for bias in ["Bullish Bias", "Bearish Bias", "Mixed/Neutral"]:
            bias_preds = [p for p in validated if p["bias"] == bias]
            bias_correct = len([p for p in bias_preds if p["validation"]["result"] == "correct"])
            stats["by_bias"][bias] = {
                "total": len(bias_preds),
                "correct": bias_correct,
                "accuracy": (bias_correct / len(bias_preds) * 100) if bias_preds else 0
            }
        
        self.predictions["stats"] = stats
    
    def get_adjusted_confidence(self, bias, confluences, timeframe):
        """
        Calculate adjusted confidence based on learned indicator weights
        """
        if bias == "Bullish Bias":
            indicators = [c["indicator"] for c in confluences.get("bullish", [])]
        elif bias == "Bearish Bias":
            indicators = [c["indicator"] for c in confluences.get("bearish", [])]
        else:
            return 50.0  # Neutral predictions get 50% confidence
        
        # Calculate weighted confidence
        total_weight = 0
        for indicator in indicators:
            for key in self.indicator_weights.keys():
                if key.lower() in indicator.lower() or indicator.lower() in key.lower():
                    total_weight += self.indicator_weights[key]
                    break
        
        # Normalize to 0-100 scale
        base_confidence = min(100, (total_weight / len(indicators)) * 50) if indicators else 50
        
        # Apply timeframe multiplier based on historical accuracy
        if timeframe in self.predictions.get("stats", {}).get("by_timeframe", {}):
            tf_accuracy = self.predictions["stats"]["by_timeframe"][timeframe]["accuracy"]
            timeframe_multiplier = tf_accuracy / 100
            base_confidence *= timeframe_multiplier
        
        return base_confidence
    
    def display_learning_stats(self):
        """Display learning statistics and indicator weights"""
        stats = self.predictions.get("stats", {})
        
        if not stats:
            print("\n📊 No learning data available yet. Make some predictions first!")
            return
        
        print("\n" + "="*70)
        print("🧠 MACHINE LEARNING STATISTICS")
        print("="*70)
        
        print(f"\n📈 Overall Performance:")
        print(f"   Total Predictions: {stats.get('total_predictions', 0)}")
        print(f"   Validated: {stats.get('validated_predictions', 0)}")
        print(f"   Pending: {stats.get('pending_predictions', 0)}")
        print(f"   Overall Accuracy: {stats.get('overall_accuracy', 0):.1f}%")
        print(f"   Avg Accuracy Score: {stats.get('avg_accuracy_score', 0):.1f}/100")
        
        # Timeframe performance
        if stats.get("by_timeframe"):
            print(f"\n⏰ Performance by Timeframe:")
            for tf, data in sorted(stats["by_timeframe"].items()):
                print(f"   {tf:6s}: {data['correct']:3d}/{data['total']:3d} ({data['accuracy']:.1f}%)")
        
        # Bias performance
        if stats.get("by_bias"):
            print(f"\n🎯 Performance by Bias:")
            for bias, data in stats["by_bias"].items():
                if data['total'] > 0:
                    print(f"   {bias:15s}: {data['correct']:3d}/{data['total']:3d} ({data['accuracy']:.1f}%)")
        
        # Indicator weights
        print(f"\n⚖️ Learned Indicator Weights:")
        sorted_weights = sorted(self.indicator_weights.items(), key=lambda x: x[1], reverse=True)
        for indicator, weight in sorted_weights:
            bar_length = int(weight * 20)
            bar = "█" * bar_length
            print(f"   {indicator:20s}: {bar} {weight:.2f}")
        
        # SR Zones summary
        print(f"\n🎯 Support/Resistance Zones Learned:")
        total_zones = sum(
            len(zones.get("support_zones", [])) + len(zones.get("resistance_zones", []))
            for zones in self.sr_zones.values()
        )
        print(f"   Total active zones: {total_zones}")
        print(f"   Tokens tracked: {len(self.sr_zones)}")
        
        print("="*70)
    
    def display_sr_zones(self, token, timeframe, current_price=None):
        """Display learned SR zones for a specific token and timeframe"""
        zone_key = f"{token}_{timeframe}"
        
        if zone_key not in self.sr_zones:
            print(f"\n⚠️ No learned SR zones for {token} on {timeframe} timeframe yet.")
            return
        
        zones = self.sr_zones[zone_key]
        
        print("\n" + "="*70)
        print(f"🎯 ML-LEARNED SUPPORT & RESISTANCE ZONES: {token} ({timeframe})")
        print("="*70)
        
        if current_price:
            print(f"💰 Current Price: ${current_price:.4f}")
        
        # Display resistance zones
        resistance_zones = zones.get("resistance_zones", [])
        if resistance_zones:
            print(f"\n🔴 RESISTANCE ZONES ({len(resistance_zones)}):")
            print("-" * 70)
            for i, zone in enumerate(sorted(resistance_zones, key=lambda x: x["price"], reverse=True), 1):
                strength_bar = "█" * int(zone["strength"] * 20)
                hold_rate = (zone["holds"] / zone["tests"] * 100) if zone["tests"] > 0 else 0
                
                distance = ""
                if current_price:
                    dist_pct = ((zone["price"] - current_price) / current_price) * 100
                    distance = f" [{dist_pct:+.2f}%]"
                
                print(f"{i}. ${zone['price']:.4f}{distance}")
                print(f"   Strength: {strength_bar} {zone['strength']:.2f}")
                print(f"   Tests: {zone['tests']} | Holds: {zone['holds']} | Breaks: {zone['breaks']}")
                print(f"   Hold Rate: {hold_rate:.1f}%")
                print(f"   Last Tested: {zone['last_tested'][:10]}")
                print()
        
        # Display support zones
        support_zones = zones.get("support_zones", [])
        if support_zones:
            print(f"🟢 SUPPORT ZONES ({len(support_zones)}):")
            print("-" * 70)
            for i, zone in enumerate(sorted(support_zones, key=lambda x: x["price"], reverse=True), 1):
                strength_bar = "█" * int(zone["strength"] * 20)
                hold_rate = (zone["holds"] / zone["tests"] * 100) if zone["tests"] > 0 else 0
                
                distance = ""
                if current_price:
                    dist_pct = ((zone["price"] - current_price) / current_price) * 100
                    distance = f" [{dist_pct:+.2f}%]"
                
                print(f"{i}. ${zone['price']:.4f}{distance}")
                print(f"   Strength: {strength_bar} {zone['strength']:.2f}")
                print(f"   Tests: {zone['tests']} | Holds: {zone['holds']} | Breaks: {zone['breaks']}")
                print(f"   Hold Rate: {hold_rate:.1f}%")
                print(f"   Last Tested: {zone['last_tested'][:10]}")
                print()
        
        print("="*70)
    
    def get_best_indicators(self, top_n=5):
        """Return the top N performing indicators"""
        sorted_weights = sorted(self.indicator_weights.items(), key=lambda x: x[1], reverse=True)
        return sorted_weights[:top_n]


# Integration with existing TradingAnalyzer class
def enhance_trading_analyzer_with_ml(analyzer_class):
    """
    Decorator/wrapper to add ML capabilities to TradingAnalyzer
    """
    original_generate_analysis = analyzer_class.generate_comprehensive_analysis
    
    def new_generate_analysis(self, df, save_prediction=False, token=None, timeframe=None):
        # Get original analysis
        confluences, latest_row = original_generate_analysis(self, df)
        
        # If ML tracker exists, adjust confidence
        if hasattr(self, 'ml_tracker') and self.ml_tracker:
            # Calculate original bias
            from betterpredictormodule import TradingAnalyzer
            temp_analyzer = TradingAnalyzer()
            bias, strength = temp_analyzer.calculate_confluence_strength(confluences)
            
            # Get ML-adjusted confidence
            adjusted_confidence = self.ml_tracker.get_adjusted_confidence(
                bias, confluences, timeframe or "15m"
            )
            
            # Get ML SR levels
            if token and timeframe:
                ml_sr_levels = self.ml_tracker.get_ml_sr_levels(
                    token, timeframe, latest_row['Close'],
                    lookback_timeframes=[timeframe, "1h", "4h", "1d"]  # Multi-timeframe confluence
                )
                
                # Add to confluences
                confluences['ml_sr_levels'] = ml_sr_levels
                
                # Check if price is near strong SR zones
                nearest_support = ml_sr_levels.get("nearest_support")
                nearest_resistance = ml_sr_levels.get("nearest_resistance")
                
                if nearest_support and nearest_support["strength"] > 0.7:
                    dist_pct = abs((latest_row['Close'] - nearest_support["price"]) / latest_row['Close']) * 100
                    if dist_pct < 2:  # Within 2%
                        confluences['bullish'].append({
                            'indicator': 'ML Support Zone',
                            'condition': f"Strong support at ${nearest_support['price']:.4f} (Strength: {nearest_support['strength']:.2f})",
                            'implication': f"Price near ML-learned support with {nearest_support['tests']} historical tests ({nearest_support['holds']} holds). High probability bounce zone.",
                            'strength': 'Strong' if nearest_support['strength'] > 0.8 else 'Medium',
                            'timeframe': 'Multi-timeframe'
                        })
                
                if nearest_resistance and nearest_resistance["strength"] > 0.7:
                    dist_pct = abs((latest_row['Close'] - nearest_resistance["price"]) / latest_row['Close']) * 100
                    if dist_pct < 2:  # Within 2%
                        confluences['bearish'].append({
                            'indicator': 'ML Resistance Zone',
                            'condition': f"Strong resistance at ${nearest_resistance['price']:.4f} (Strength: {nearest_resistance['strength']:.2f})",
                            'implication': f"Price near ML-learned resistance with {nearest_resistance['tests']} historical tests ({nearest_resistance['holds']} holds). High probability rejection zone.",
                            'strength': 'Strong' if nearest_resistance['strength'] > 0.8 else 'Medium',
                            'timeframe': 'Multi-timeframe'
                        })
            
            # Save prediction if requested
            if save_prediction and token and timeframe:
                self.ml_tracker.save_prediction(
                    token=token,
                    timeframe=timeframe,
                    bias=bias,
                    strength=adjusted_confidence,
                    current_price=latest_row['Close'],
                    confluences=confluences,
                    key_levels={
                        "support": latest_row['S1'],
                        "resistance": latest_row['R1'],
                        "pivot": latest_row['Pivot']
                    },
                    technical_data=latest_row.to_dict()
                )
            
            # Add ML confidence to results
            confluences['ml_confidence'] = adjusted_confidence
            confluences['ml_enabled'] = True
        
        return confluences, latest_row
    
    analyzer_class.generate_comprehensive_analysis = new_generate_analysis
    return analyzer_class


# Modified main function with ML integration
def main_with_ml():
    """Enhanced main with ML prediction tracking"""
    from betterpredictormodule import TradingAnalyzer, user_input_token, user_input_timeframe
    
    print("🤖 Nunno's ML-Enhanced Prediction Module")
    print("="*70)
    
    # Initialize tracker and analyzer
    ml_tracker = PredictionTracker()
    analyzer = TradingAnalyzer()
    analyzer.ml_tracker = ml_tracker
    
    # Main menu
    while True:
        print("\n" + "="*70)
        print("MAIN MENU")
        print("="*70)
        print("1. Make new prediction (with ML learning)")
        print("2. Validate pending predictions")
        print("3. View learning statistics")
        print("4. View indicator weights")
        print("5. View SR zones for token")
        print("6. Exit")
        
        choice = input("\nYour choice: ").strip()
        
        if choice == "1":
            # Get inputs
            token = user_input_token()
            timeframe = user_input_timeframe()
            
            try:
                # Fetch and analyze
                df = analyzer.fetch_binance_ohlcv(symbol=token, interval=timeframe)
                df = analyzer.add_comprehensive_indicators(df)
                
                # Generate analysis with ML
                confluences, latest_row = analyzer.generate_comprehensive_analysis(
                    df, save_prediction=True, token=token, timeframe=timeframe
                )
                
                # Calculate bias
                bias, strength = analyzer.calculate_confluence_strength(confluences)
                
                # Display with ML confidence
                ml_confidence = confluences.get('ml_confidence', strength)
                ml_sr = confluences.get('ml_sr_levels', {})
                
                print(f"\n🎯 PREDICTION SUMMARY:")
                print(f"   Original Confidence: {strength:.1f}%")
                print(f"   ML-Adjusted Confidence: {ml_confidence:.1f}%")
                print(f"   Bias: {bias}")
                
                # Display ML SR levels if available
                if ml_sr:
                    print(f"\n🎯 ML-LEARNED KEY LEVELS:")
                    
                    nearest_res = ml_sr.get('nearest_resistance')
                    if nearest_res:
                        dist = ((nearest_res['price'] - latest_row['Close']) / latest_row['Close']) * 100
                        print(f"   Nearest Resistance: ${nearest_res['price']:.4f} [{dist:+.2f}%]")
                        print(f"      Strength: {nearest_res['strength']:.2f} | Tests: {nearest_res['tests']}")
                    
                    nearest_sup = ml_sr.get('nearest_support')
                    if nearest_sup:
                        dist = ((nearest_sup['price'] - latest_row['Close']) / latest_row['Close']) * 100
                        print(f"   Nearest Support: ${nearest_sup['price']:.4f} [{dist:+.2f}%]")
                        print(f"      Strength: {nearest_sup['strength']:.2f} | Tests: {nearest_sup['tests']}")
                
                # Full analysis display
                analyzer.display_analysis(token, timeframe, confluences, latest_row)
                
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()
        
        elif choice == "2":
            # Validate predictions
            ml_tracker.validate_predictions(analyzer)
        
        elif choice == "3":
            # Show stats
            ml_tracker.display_learning_stats()
        
        elif choice == "4":
            # Show top indicators
            print("\n🏆 Top Performing Indicators:")
            for indicator, weight in ml_tracker.get_best_indicators(10):
                print(f"   {indicator:20s}: {weight:.3f}")
        
        elif choice == "5":
            # Show SR zones for a token
            print("\n🎯 View SR Zones")
            token = user_input_token()
            timeframe = user_input_timeframe()
            
            try:
                # Get current price
                df = analyzer.fetch_binance_ohlcv(symbol=token, interval=timeframe, limit=10)
                current_price = df.iloc[-1]['Close'] if len(df) > 0 else None
                
                ml_tracker.display_sr_zones(token, timeframe, current_price)
                
                # Also show multi-timeframe view
                if current_price:
                    print("\n" + "="*70)
                    print("🔄 MULTI-TIMEFRAME SR CONFLUENCE")
                    print("="*70)
                    
                    ml_sr = ml_tracker.get_ml_sr_levels(
                        token, timeframe, current_price,
                        lookback_timeframes=["15m", "1h", "4h", "1d"]
                    )
                    
                    print(f"\n💰 Current Price: ${current_price:.4f}\n")
                    
                    if ml_sr['resistance']:
                        print("🔴 Top Resistance Zones (Multi-TF):")
                        for i, zone in enumerate(ml_sr['resistance'][:3], 1):
                            dist = ((zone['price'] - current_price) / current_price) * 100
                            confluence_info = f" (Confluence: {zone.get('confluence_count', 1)} TF)" if zone.get('confluence_count', 1) > 1 else ""
                            print(f"   {i}. ${zone['price']:.4f} [{dist:+.2f}%] - Strength: {zone['strength']:.2f}{confluence_info}")
                    
                    if ml_sr['support']:
                        print("\n🟢 Top Support Zones (Multi-TF):")
                        for i, zone in enumerate(ml_sr['support'][:3], 1):
                            dist = ((zone['price'] - current_price) / current_price) * 100
                            confluence_info = f" (Confluence: {zone.get('confluence_count', 1)} TF)" if zone.get('confluence_count', 1) > 1 else ""
                            print(f"   {i}. ${zone['price']:.4f} [{dist:+.2f}%] - Strength: {zone['strength']:.2f}{confluence_info}")
                    
                    print("="*70)
                    
            except Exception as e:
                print(f"❌ Error: {e}")
        
        elif choice == "6":
            print("\n👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice")


if __name__ == "__main__":
    main_with_ml()