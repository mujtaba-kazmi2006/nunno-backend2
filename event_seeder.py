"""
Event Database Seeder for Crypto Prediction System
Seeds historical events with full context: technicals, price impact, sentiment
"""

import sqlite3
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
import json
import time

class EventDatabaseSeeder:
    def __init__(self, db_path="crypto_prediction.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.setup_database()
        
    def setup_database(self):
        """Create all necessary tables"""
        
        # Events table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_date TEXT NOT NULL,
                event_type TEXT NOT NULL,
                description TEXT,
                severity INTEGER CHECK(severity BETWEEN 1 AND 10),
                expected_impact TEXT CHECK(expected_impact IN ('bullish', 'bearish', 'neutral')),
                actual_impact TEXT,
                notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Event impacts (price movements)
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS event_impacts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id INTEGER NOT NULL,
                token_symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                price_before_24h REAL,
                price_before_12h REAL,
                price_before_6h REAL,
                price_at_event REAL NOT NULL,
                price_after_1h REAL,
                price_after_6h REAL,
                price_after_12h REAL,
                price_after_24h REAL,
                price_after_48h REAL,
                price_after_7d REAL,
                price_after_14d REAL,
                price_after_30d REAL,
                pct_change_1h REAL,
                pct_change_24h REAL,
                pct_change_7d REAL,
                pct_change_30d REAL,
                volume_before_avg REAL,
                volume_at_event REAL,
                volume_spike_pct REAL,
                peak_price REAL,
                peak_time_hours REAL,
                bottom_price REAL,
                bottom_time_hours REAL,
                FOREIGN KEY(event_id) REFERENCES events(id)
            )
        ''')
        
        # Technical snapshots
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS technical_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id INTEGER NOT NULL,
                token_symbol TEXT NOT NULL,
                snapshot_time TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                rsi_14 REAL,
                rsi_21 REAL,
                stoch_k REAL,
                stoch_d REAL,
                macd REAL,
                macd_signal REAL,
                macd_histogram REAL,
                ema_9 REAL,
                ema_21 REAL,
                ema_50 REAL,
                ema_alignment TEXT,
                adx REAL,
                di_plus REAL,
                di_minus REAL,
                bb_upper REAL,
                bb_middle REAL,
                bb_lower REAL,
                bb_width REAL,
                bb_position REAL,
                atr REAL,
                atr_percent REAL,
                volume_ratio REAL,
                market_regime TEXT,
                trend_strength TEXT,
                confluence_score INTEGER,
                dominant_bias TEXT,
                FOREIGN KEY(event_id) REFERENCES events(id)
            )
        ''')
        
        # Sentiment data
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS sentiment_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id INTEGER NOT NULL,
                snapshot_date TEXT NOT NULL,
                fear_greed_index INTEGER,
                fear_greed_label TEXT,
                fear_greed_trend TEXT,
                social_volume INTEGER,
                reddit_sentiment REAL,
                twitter_mentions INTEGER,
                news_sentiment TEXT,
                whale_activity TEXT,
                FOREIGN KEY(event_id) REFERENCES events(id)
            )
        ''')
        
        # Pattern library (aggregated patterns)
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS pattern_library (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                technical_conditions TEXT,
                sample_size INTEGER,
                avg_response_1h REAL,
                avg_response_24h REAL,
                avg_response_7d REAL,
                median_response_24h REAL,
                success_rate REAL,
                best_entry_timing TEXT,
                typical_peak_hours REAL,
                risk_factors TEXT,
                last_updated TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.conn.commit()
        print("✅ Database schema created successfully")
    
    def fetch_historical_prices(self, symbol, event_date, days_before=30, days_after=30, interval="4h"):
        """Fetch historical price data around an event"""
        
        event_dt = datetime.fromisoformat(event_date)
        start_dt = event_dt - timedelta(days=days_before)
        end_dt = event_dt + timedelta(days=days_after)
        
        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)
        
        # Try Binance first
        url = f"https://api.binance.com/api/v3/klines"
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": 1000
        }
        
        try:
            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data, columns=[
                    "Open Time", "Open", "High", "Low", "Close", "Volume",
                    "Close Time", "Quote Asset Volume", "Number of Trades",
                    "Taker Buy Base", "Taker Buy Quote", "Ignore"
                ])
                
                df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
                df = df[["Open Time", "Open", "High", "Low", "Close", "Volume"]].astype({
                    "Open": float, "High": float, "Low": float, 
                    "Close": float, "Volume": float
                })
                df.set_index('Open Time', inplace=True)
                return df
        except Exception as e:
            print(f"⚠️ Binance fetch failed: {e}")
        
        # Fallback to CoinGecko
        print("Trying CoinGecko fallback...")
        return self._fetch_coingecko_fallback(symbol, event_date, days_before, days_after)
    
    def _fetch_coingecko_fallback(self, symbol, event_date, days_before, days_after):
        """Fallback to CoinGecko for historical data"""
        
        # Symbol mapping
        symbol_map = {
            "BTCUSDT": "bitcoin",
            "ETHUSDT": "ethereum",
            "BNBUSDT": "binancecoin",
            "SOLUSDT": "solana",
            "XRPUSDT": "ripple"
        }
        
        coin_id = symbol_map.get(symbol.upper())
        if not coin_id:
            raise Exception(f"Symbol {symbol} not supported for CoinGecko fallback")
        
        total_days = days_before + days_after
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        params = {
            "vs_currency": "usd",
            "days": total_days,
            "interval": "hourly"
        }
        
        response = requests.get(url, params=params, timeout=15)
        if response.status_code == 200:
            data = response.json()
            prices = data['prices']
            volumes = data['total_volumes']
            
            df = pd.DataFrame(prices, columns=['timestamp', 'Close'])
            df['Volume'] = [v[1] for v in volumes]
            df['Open Time'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['Open'] = df['Close']
            df['High'] = df['Close'] * 1.005
            df['Low'] = df['Close'] * 0.995
            
            df = df[["Open Time", "Open", "High", "Low", "Close", "Volume"]]
            df.set_index('Open Time', inplace=True)
            return df
        
        raise Exception("All price data sources failed")
    
    def add_technical_indicators(self, df):
        """Add comprehensive technical indicators to dataframe"""
        
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']
        
        # Momentum
        df['RSI_14'] = RSIIndicator(close, window=14).rsi()
        df['RSI_21'] = RSIIndicator(close, window=21).rsi()
        df['Stoch_K'] = StochasticOscillator(high, low, close, window=14).stoch()
        df['Stoch_D'] = StochasticOscillator(high, low, close, window=14).stoch_signal()
        
        # Trend
        df['EMA_9'] = EMAIndicator(close, window=9).ema_indicator()
        df['EMA_21'] = EMAIndicator(close, window=21).ema_indicator()
        df['EMA_50'] = EMAIndicator(close, window=50).ema_indicator()
        
        macd = MACD(close)
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Histogram'] = macd.macd_diff()
        
        adx = ADXIndicator(high, low, close)
        df['ADX'] = adx.adx()
        df['DI_Plus'] = adx.adx_pos()
        df['DI_Minus'] = adx.adx_neg()
        
        # Volatility
        bb = BollingerBands(close, window=20, window_dev=2)
        df['BB_Upper'] = bb.bollinger_hband()
        df['BB_Middle'] = bb.bollinger_mavg()
        df['BB_Lower'] = bb.bollinger_lband()
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle'] * 100
        df['BB_Position'] = (close - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        df['ATR'] = AverageTrueRange(high, low, close).average_true_range()
        df['ATR_Percent'] = (df['ATR'] / close) * 100
        
        # Volume
        df['Volume_SMA'] = volume.rolling(window=20).mean()
        df['Volume_Ratio'] = volume / df['Volume_SMA']
        
        df.dropna(inplace=True)
        return df
    
    def get_fear_greed_for_date(self, target_date):
        """Fetch Fear & Greed index for specific date"""
        
        try:
            url = "https://api.alternative.me/fng/?limit=0"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()['data']
                target_ts = int(datetime.fromisoformat(target_date).timestamp())
                
                # Find closest date
                closest = min(data, key=lambda x: abs(int(x['timestamp']) - target_ts))
                
                return {
                    'value': int(closest['value']),
                    'classification': closest['value_classification'],
                    'date': closest['timestamp']
                }
        except Exception as e:
            print(f"⚠️ Fear & Greed fetch failed: {e}")
        
        return {'value': None, 'classification': 'Unknown', 'date': None}
    
    def classify_market_regime(self, row):
        """Classify market regime from technical data"""
        
        adx = row['ADX']
        ema_9 = row['EMA_9']
        ema_21 = row['EMA_21']
        ema_50 = row['EMA_50']
        price = row['Close']
        bb_width = row['BB_Width']
        
        if adx > 25:
            if ema_9 > ema_21 > ema_50 and price > ema_50:
                return "Trending_Bull"
            elif ema_9 < ema_21 < ema_50 and price < ema_50:
                return "Trending_Bear"
            else:
                return "Trending_Mixed"
        elif adx < 20 and bb_width < 2:
            return "Ranging"
        elif row['ATR_Percent'] > 3:
            return "Volatile"
        else:
            return "Transitional"
    
    def seed_event(self, event_date, event_type, description, severity, 
                   expected_impact, token_symbol="BTCUSDT", timeframe="4h", notes=""):
        """
        Seed a complete event with all context
        
        Args:
            event_date: "2024-03-20" or "2024-03-20 14:00:00"
            event_type: "Fed_Rate_Cut", "ETF_Approval", "Halving", etc.
            description: Detailed description
            severity: 1-10 impact severity
            expected_impact: "bullish", "bearish", or "neutral"
            token_symbol: Which crypto (default BTC)
            timeframe: Analysis timeframe
            notes: Additional context
        """
        
        print(f"\n{'='*70}")
        print(f"📥 SEEDING EVENT: {event_type}")
        print(f"📅 Date: {event_date}")
        print(f"📊 Token: {token_symbol} | Timeframe: {timeframe}")
        print(f"{'='*70}")
        
        # 1. Insert event record
        self.cursor.execute('''
            INSERT INTO events (event_date, event_type, description, severity, expected_impact, notes)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (event_date, event_type, description, severity, expected_impact, notes))
        
        event_id = self.cursor.lastrowid
        self.conn.commit()
        print(f"✅ Event ID {event_id} created")
        
        # 2. Fetch historical price data
        print("📡 Fetching historical price data...")
        try:
            df = self.fetch_historical_prices(
                symbol=token_symbol,
                event_date=event_date.split()[0],  # Extract date part
                days_before=30,
                days_after=30,
                interval=timeframe
            )
            print(f"✅ Fetched {len(df)} candles")
        except Exception as e:
            print(f"❌ Failed to fetch prices: {e}")
            return None
        
        # 3. Add technical indicators
        print("🔧 Calculating technical indicators...")
        df = self.add_technical_indicators(df)
        print(f"✅ {len(df.columns)} indicators added")
        
        # 4. Find event candle
        event_dt = pd.to_datetime(event_date)
        event_candle_idx = df.index.get_indexer([event_dt], method='nearest')[0]
        event_candle = df.iloc[event_candle_idx]
        
        print(f"🎯 Event candle: {df.index[event_candle_idx]}")
        print(f"💰 Price at event: ${event_candle['Close']:.2f}")
        
        # 5. Extract prices at different intervals
        def safe_price(idx):
            return df.iloc[idx]['Close'] if 0 <= idx < len(df) else None
        
        price_at_event = event_candle['Close']
        price_before_24h = safe_price(event_candle_idx - 6) if timeframe == "4h" else safe_price(event_candle_idx - 24)
        price_before_12h = safe_price(event_candle_idx - 3) if timeframe == "4h" else safe_price(event_candle_idx - 12)
        price_before_6h = safe_price(event_candle_idx - 2) if timeframe == "4h" else safe_price(event_candle_idx - 6)
        
        price_after_1h = safe_price(event_candle_idx + 1)
        price_after_6h = safe_price(event_candle_idx + 2) if timeframe == "4h" else safe_price(event_candle_idx + 6)
        price_after_12h = safe_price(event_candle_idx + 3) if timeframe == "4h" else safe_price(event_candle_idx + 12)
        price_after_24h = safe_price(event_candle_idx + 6) if timeframe == "4h" else safe_price(event_candle_idx + 24)
        price_after_48h = safe_price(event_candle_idx + 12) if timeframe == "4h" else safe_price(event_candle_idx + 48)
        price_after_7d = safe_price(event_candle_idx + 42) if timeframe == "4h" else safe_price(event_candle_idx + 168)
        price_after_14d = safe_price(event_candle_idx + 84) if timeframe == "4h" else safe_price(event_candle_idx + 336)
        price_after_30d = safe_price(event_candle_idx + 180) if timeframe == "4h" else safe_price(event_candle_idx + 720)
        
        # Calculate percentage changes
        pct_1h = ((price_after_1h - price_at_event) / price_at_event * 100) if price_after_1h else None
        pct_24h = ((price_after_24h - price_at_event) / price_at_event * 100) if price_after_24h else None
        pct_7d = ((price_after_7d - price_at_event) / price_at_event * 100) if price_after_7d else None
        pct_30d = ((price_after_30d - price_at_event) / price_at_event * 100) if price_after_30d else None
        
        # Find peak and bottom in next 7 days
        future_window = df.iloc[event_candle_idx:min(event_candle_idx + 42, len(df))]
        peak_price = future_window['High'].max()
        peak_idx = future_window['High'].idxmax()
        peak_hours = (peak_idx - event_candle.name).total_seconds() / 3600
        
        bottom_price = future_window['Low'].min()
        bottom_idx = future_window['Low'].idxmin()
        bottom_hours = (bottom_idx - event_candle.name).total_seconds() / 3600
        
        # Volume analysis
        volume_before_avg = df.iloc[max(0, event_candle_idx-20):event_candle_idx]['Volume'].mean()
        volume_at_event = event_candle['Volume']
        volume_spike = ((volume_at_event - volume_before_avg) / volume_before_avg * 100) if volume_before_avg > 0 else 0
        
        # 6. Store event impact
        self.cursor.execute('''
            INSERT INTO event_impacts (
                event_id, token_symbol, timeframe,
                price_before_24h, price_before_12h, price_before_6h, price_at_event,
                price_after_1h, price_after_6h, price_after_12h, price_after_24h,
                price_after_48h, price_after_7d, price_after_14d, price_after_30d,
                pct_change_1h, pct_change_24h, pct_change_7d, pct_change_30d,
                volume_before_avg, volume_at_event, volume_spike_pct,
                peak_price, peak_time_hours, bottom_price, bottom_time_hours
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            event_id, token_symbol, timeframe,
            price_before_24h, price_before_12h, price_before_6h, price_at_event,
            price_after_1h, price_after_6h, price_after_12h, price_after_24h,
            price_after_48h, price_after_7d, price_after_14d, price_after_30d,
            pct_1h, pct_24h, pct_7d, pct_30d,
            volume_before_avg, volume_at_event, volume_spike,
            peak_price, peak_hours, bottom_price, bottom_hours
        ))
        
        print(f"✅ Price impact stored")
        print(f"   📈 24h change: {pct_24h:+.2f}%" if pct_24h else "   📈 24h change: N/A")
        print(f"   📈 7d change: {pct_7d:+.2f}%" if pct_7d else "   📈 7d change: N/A")
        print(f"   🎯 Peak: ${peak_price:.2f} at {peak_hours:.1f}h")
        
        # 7. Store technical snapshot
        ema_alignment = "bullish" if event_candle['EMA_9'] > event_candle['EMA_21'] > event_candle['EMA_50'] else \
                       "bearish" if event_candle['EMA_9'] < event_candle['EMA_21'] < event_candle['EMA_50'] else "mixed"
        
        market_regime = self.classify_market_regime(event_candle)
        
        # Simple trend strength
        if event_candle['ADX'] > 40:
            trend_strength = "Very_Strong"
        elif event_candle['ADX'] > 25:
            trend_strength = "Strong"
        elif event_candle['ADX'] > 20:
            trend_strength = "Moderate"
        else:
            trend_strength = "Weak"
        
        # Dominant bias
        bullish_signals = 0
        bearish_signals = 0
        
        if event_candle['RSI_14'] > 50: bullish_signals += 1
        else: bearish_signals += 1
        
        if event_candle['MACD'] > event_candle['MACD_Signal']: bullish_signals += 1
        else: bearish_signals += 1
        
        if ema_alignment == "bullish": bullish_signals += 2
        elif ema_alignment == "bearish": bearish_signals += 2
        
        dominant_bias = "bullish" if bullish_signals > bearish_signals else "bearish" if bearish_signals > bullish_signals else "neutral"
        
        self.cursor.execute('''
            INSERT INTO technical_snapshots (
                event_id, token_symbol, snapshot_time, timeframe,
                rsi_14, rsi_21, stoch_k, stoch_d,
                macd, macd_signal, macd_histogram,
                ema_9, ema_21, ema_50, ema_alignment,
                adx, di_plus, di_minus,
                bb_upper, bb_middle, bb_lower, bb_width, bb_position,
                atr, atr_percent, volume_ratio,
                market_regime, trend_strength, confluence_score, dominant_bias
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            event_id, token_symbol, event_candle.name.isoformat(), timeframe,
            event_candle['RSI_14'], event_candle['RSI_21'], 
            event_candle['Stoch_K'], event_candle['Stoch_D'],
            event_candle['MACD'], event_candle['MACD_Signal'], event_candle['MACD_Histogram'],
            event_candle['EMA_9'], event_candle['EMA_21'], event_candle['EMA_50'], ema_alignment,
            event_candle['ADX'], event_candle['DI_Plus'], event_candle['DI_Minus'],
            event_candle['BB_Upper'], event_candle['BB_Middle'], event_candle['BB_Lower'],
            event_candle['BB_Width'], event_candle['BB_Position'],
            event_candle['ATR'], event_candle['ATR_Percent'], event_candle['Volume_Ratio'],
            market_regime, trend_strength, bullish_signals + bearish_signals, dominant_bias
        ))
        
        print(f"✅ Technical snapshot stored")
        print(f"   🎯 Regime: {market_regime}")
        print(f"   📊 RSI: {event_candle['RSI_14']:.1f}")
        print(f"   📈 EMA Alignment: {ema_alignment}")
        
        # 8. Store sentiment data
        fg_data = self.get_fear_greed_for_date(event_date.split()[0])
        
        self.cursor.execute('''
            INSERT INTO sentiment_snapshots (
                event_id, snapshot_date, fear_greed_index, fear_greed_label
            ) VALUES (?, ?, ?, ?)
        ''', (
            event_id, event_date.split()[0], 
            fg_data['value'], fg_data['classification']
        ))
        
        print(f"✅ Sentiment snapshot stored")
        print(f"   😱 Fear & Greed: {fg_data['value']} ({fg_data['classification']})")
        
        self.conn.commit()
        
        print(f"\n{'='*70}")
        print(f"✅ EVENT {event_id} FULLY SEEDED")
        print(f"{'='*70}\n")
        
        return event_id
    
    def seed_multiple_events(self, events_list):
        """Seed multiple events from a list"""
        
        print(f"\n🚀 SEEDING {len(events_list)} EVENTS\n")
        
        successful = 0
        failed = 0
        
        for i, event in enumerate(events_list, 1):
            print(f"\n[{i}/{len(events_list)}] Processing...")
            
            try:
                event_id = self.seed_event(
                    event_date=event['date'],
                    event_type=event['type'],
                    description=event['description'],
                    severity=event['severity'],
                    expected_impact=event['expected_impact'],
                    token_symbol=event.get('symbol', 'BTCUSDT'),
                    timeframe=event.get('timeframe', '4h'),
                    notes=event.get('notes', '')
                )
                
                if event_id:
                    successful += 1
                    time.sleep(2)  # Rate limiting
                else:
                    failed += 1
                    
            except Exception as e:
                print(f"❌ Failed to seed event: {e}")
                failed += 1
                continue
        
        print(f"\n{'='*70}")
        print(f"📊 SEEDING COMPLETE")
        print(f"   ✅ Successful: {successful}")
        print(f"   ❌ Failed: {failed}")
        print(f"{'='*70}\n")
    
    def get_summary(self):
        """Get database summary statistics"""
        
        total_events = self.cursor.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        total_impacts = self.cursor.execute("SELECT COUNT(*) FROM event_impacts").fetchone()[0]
        total_technicals = self.cursor.execute("SELECT COUNT(*) FROM technical_snapshots").fetchone()[0]
        
        event_types = self.cursor.execute('''
            SELECT event_type, COUNT(*) as count 
            FROM events 
            GROUP BY event_type 
            ORDER BY count DESC
        ''').fetchall()
        
        print(f"\n{'='*70}")
        print("📊 DATABASE SUMMARY")
        print(f"{'='*70}")
        print(f"Total Events: {total_events}")
        print(f"Total Impact Records: {total_impacts}")
        print(f"Total Technical Snapshots: {total_technicals}")
        print(f"\nEvent Types:")
        for event_type, count in event_types:
            print(f"  • {event_type}: {count}")
        print(f"{'='*70}\n")
    
    def close(self):
        """Close database connection"""
        self.conn.close()


# ============================================================================
# EXAMPLE USAGE & SEED DATA
# ============================================================================

if __name__ == "__main__":
    
    # Initialize seeder
    seeder = EventDatabaseSeeder(db_path="crypto_prediction.db")
    
    # Define historical events to seed
    MAJOR_EVENTS = [
        {
            "date": "2020-03-15 15:00:00",
            "type": "Fed_Emergency_Cut",
            "description": "Emergency 1.00% rate cut to near zero",
            "severity": 10,
            "expected_impact": "bullish",
            "notes": "COVID-19 response, unprecedented emergency action"
        },
        {
            "date": "2021-11-10 08:30:00",
            "type": "Inflation_CPI",
            "description": "CPI hits 6.2%, highest since 1990",
            "severity": 8,
            "expected_impact": "bullish",
            "notes": "Major inflation concern, flight to Bitcoin"
        },
        {
            "date": "2022-03-16 14:00:00",
            "type": "Fed_Rate_Hike",
            "description": "First rate hike 0.25% in hiking cycle",
            "severity": 9,
            "expected_impact": "bearish",
            "notes": "Start of aggressive tightening cycle"
        },
        {
            "date": "2022-05-09 12:00:00",
            "type": "LUNA_Collapse",
            "description": "UST de-pegs, LUNA ecosystem begins spiral",
            "severity": 10,
            "expected_impact": "bearish",
            "notes": "Massive systemic failure of algorithmic stablecoin"
        },
        {
            "date": "2022-09-15 06:42:00",
            "type": "The_Merge",
            "description": "Ethereum transitions to Proof of Stake",
            "severity": 9,
            "expected_impact": "neutral",
            "notes": "Sell the news event, but major technical milestone"
        },
        {
            "date": "2022-11-08 08:00:00",
            "type": "Exchange_Crisis",
            "description": "FTX collapse announcement",
            "severity": 10,
            "expected_impact": "bearish",
            "notes": "Major exchange insolvency, massive contagion"
        },
        {
            "date": "2023-03-10 09:00:00",
            "type": "Banking_Crisis",
            "description": "Silicon Valley Bank collapse",
            "severity": 9,
            "expected_impact": "bullish",
            "notes": "Traditional finance crisis, crypto seen as alternative"
        },
        {
            "date": "2023-07-13 11:00:00",
            "type": "XRP_Ruling",
            "description": "Judge rules XRP is not a security for retail",
            "severity": 9,
            "expected_impact": "bullish",
            "notes": "Major regulatory victory for crypto"
        },
        {
            "date": "2023-08-29 10:00:00",
            "type": "Grayscale_Victory",
            "description": "Court rules in favor of Grayscale against SEC",
            "severity": 8,
            "expected_impact": "bullish",
            "notes": "Paved the literal way for Spot ETFs"
        },
        {
            "date": "2024-01-10 16:00:00",
            "type": "ETF_Approval",
            "description": "Bitcoin Spot ETF approved by SEC",
            "severity": 10,
            "expected_impact": "bullish",
            "notes": "Massive institutional access milestone"
        },
        {
            "date": "2024-04-20 00:00:00",
            "type": "Bitcoin_Halving",
            "description": "Bitcoin halving event (4th halving)",
            "severity": 9,
            "expected_impact": "bullish",
            "notes": "Supply reduction event, historically bullish"
        },
        {
            "date": "2024-05-23 16:00:00",
            "type": "ETH_ETF_Approval",
            "description": "SEC approves Spot Ethereum ETF 19b-4s",
            "severity": 9,
            "expected_impact": "bullish",
            "notes": "Surprise approval boosting entire altcoin market"
        },
        {
            "date": "2024-09-18 14:00:00",
            "type": "Fed_Rate_Cut",
            "description": "Fed cuts rates by 50bps, first in years",
            "severity": 9,
            "expected_impact": "bullish",
            "notes": "Shift to monetary easing cycle"
        },
        {
            "date": "2024-11-06 02:00:00",
            "type": "US_Election",
            "description": "Trump win confirmed, crypto-friendly stance",
            "severity": 10,
            "expected_impact": "bullish",
            "notes": "Massive rally across all risk assets"
        },
        {
            "date": "2024-11-18 09:00:00",
            "type": "MSTR_Buy",
            "description": "MicroStrategy buys $4.6B of Bitcoin",
            "severity": 7,
            "expected_impact": "bullish",
            "notes": "Massive institutional absorption of supply"
        },
        {
            "date": "2021-05-19 12:00:00",
            "type": "China_Ban",
            "description": "China intensifies crypto mining ban",
            "severity": 9,
            "expected_impact": "bearish",
            "notes": "Hash rate exodus from China"
        }
    ]
    
    # Seed all events
    print("🚀 Starting database seeding process...")
    print("This will take several minutes due to API rate limits\n")
    
    # Option 1: Seed all at once
    seeder.seed_multiple_events(MAJOR_EVENTS)
    
    # Option 2: Seed one by one (more control)
    # for event in MAJOR_EVENTS[:2]:  # Test with first 2
    #     seeder.seed_event(
    #         event_date=event['date'],
    #         event_type=event['type'],
    #         description=event['description'],
    #         severity=event['severity'],
    #         expected_impact=event['expected_impact'],
    #         notes=event.get('notes', '')
    #     )
    #     time.sleep(3)  # Rate limiting
    
    # Show summary
    seeder.get_summary()
    
    # Close connection
    seeder.close()
    
    print("✅ Database seeding complete!")
    print("📁 Database file: crypto_prediction.db")
    print("\nYou can now query this database from your prediction app!")