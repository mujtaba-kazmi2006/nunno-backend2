import requests
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.trend import EMAIndicator, SMAIndicator, MACD, ADXIndicator, IchimokuIndicator
from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel
from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator
from datetime import datetime
import warnings
import time
import random
import sys
import os
from fuzzywuzzy import process

# Filter warnings
warnings.filterwarnings('ignore')

# Try to import local modules
try:
    from tokenomics_utils import ComprehensiveTokenomics
    TOKENOMICS_AVAILABLE = True
except ImportError:
    TOKENOMICS_AVAILABLE = False
    print("⚠️ Tokenomics module not found. Some features will be disabled.")

try:
    from social_scraper_module import CryptoSocialScraper
    SOCIAL_AVAILABLE = True
except ImportError:
    SOCIAL_AVAILABLE = False
    print("⚠️ Social scraper module not found. Some features will be disabled.")


class TradingAnalyzer:
    def __init__(self):
        self.confluence_threshold = 3  # Minimum confluences for strong signals
        
        # Enhanced proxy and fallback system
        self.proxy_endpoints = [
            # Primary fallback APIs (free alternatives)
            "https://api.binance.us/api/v3/klines",  # Binance US
            "https://api.coingecko.com/api/v3/coins/{}/ohlc",  # CoinGecko OHLC
            "https://api.coincap.io/v2/assets/{}/history",  # CoinCap
        ]
        
        # Proxy servers for geo-restricted access
        self.proxy_list = [
            {"https": "https://proxy-server.scraperapi.com:8001"},
            {"https": "https://rotating-residential.scraperapi.com:8001"},
            {"https": "https://premium-datacenter.scraperapi.com:8001"},
        ]
        
        # Headers to mimic different browsers/locations
        self.headers_list = [
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Accept": "application/json",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
            },
            {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
                "Accept": "application/json, text/plain, */*",
                "Accept-Language": "en-us",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive",
            },
            {
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.96 Safari/537.36",
                "Accept": "application/json",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
            }
        ]
    
    def make_request_with_fallback(self, url, max_retries=3):
        """Enhanced request method with proxy fallback and error handling"""
        
        # Try direct connection first
        for attempt in range(max_retries):
            try:
                headers = random.choice(self.headers_list)
                response = requests.get(
                    url, 
                    headers=headers, 
                    timeout=15,
                    verify=True  # Keep SSL verification for security
                )
                if response.status_code == 200:
                    return response
                elif response.status_code == 451:  # Geo-blocked
                    print(f"Geo-blocked (451), trying proxy fallback...")
                    break
                else:
                    print(f"API returned status {response.status_code}, retrying...")
                    
            except requests.exceptions.Timeout:
                print(f"Timeout on attempt {attempt + 1}, retrying...")
                time.sleep(1)
            except requests.exceptions.ConnectionError:
                print(f"Connection error on attempt {attempt + 1}, trying proxy...")
                break
            except Exception as e:
                print(f"Request error: {str(e)}")
                if attempt == max_retries - 1:
                    break
                time.sleep(1)
        
        # If direct connection fails, try with proxies (if available)
        if hasattr(self, 'proxy_api_key') and self.proxy_api_key:
            for proxy in self.proxy_list:
                try:
                    headers = random.choice(self.headers_list)
                    headers['X-API-Key'] = self.proxy_api_key
                    
                    response = requests.get(
                        url,
                        headers=headers,
                        proxies=proxy,
                        timeout=20
                    )
                    if response.status_code == 200:
                        print("Successfully connected via proxy")
                        return response
                except Exception as e:
                    print(f"Proxy attempt failed: {str(e)}")
                    continue
        
        raise Exception("All connection attempts failed. API may be geo-blocked or temporarily unavailable.")
    
    def fetch_binance_ohlcv_with_fallback(self, symbol="BTCUSDT", interval="15m", limit=1000):
        """Fetch OHLCV data with multiple fallback options"""
        
        # Method 1: Try Binance main API
        binance_url = f"https://api.binance.com/api/v3/klines?symbol={symbol.upper()}&interval={interval}&limit={limit}"
        try:
            response = self.make_request_with_fallback(binance_url)
            return self._parse_binance_response(response.json())
        except Exception as e:
            print(f"Binance main API failed: {str(e)}")
        
        # Method 2: Try Binance US API
        try:
            binance_us_url = f"https://api.binance.us/api/v3/klines?symbol={symbol.upper()}&interval={interval}&limit={limit}"
            response = self.make_request_with_fallback(binance_us_url)
            return self._parse_binance_response(response.json())
        except Exception as e:
            print(f"Binance US API failed: {str(e)}")
        
        # Method 3: CoinGecko fallback (different format)
        try:
            # Convert symbol to CoinGecko format
            coingecko_id = self._symbol_to_coingecko_id(symbol)
            if coingecko_id:
                return self._fetch_coingecko_data(coingecko_id, interval, limit)
        except Exception as e:
            print(f"CoinGecko fallback failed: {str(e)}")
        
        # Method 4: Generate synthetic data for demo purposes
        print("All APIs failed. Generating synthetic data for demonstration...")
        return self._generate_synthetic_data(symbol, interval, limit)
    
    def _parse_binance_response(self, data):
        """Parse standard Binance API response"""
        df = pd.DataFrame(data, columns=[
            "Open Time", "Open", "High", "Low", "Close", "Volume",
            "Close Time", "Quote Asset Volume", "Number of Trades",
            "Taker Buy Base", "Taker Buy Quote", "Ignore"
        ])
        
        # Convert timestamps and prices
        df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
        df = df[["Open Time", "Open", "High", "Low", "Close", "Volume"]].astype({
            "Open": float, "High": float, "Low": float, "Close": float, "Volume": float
        })
        df.set_index('Open Time', inplace=True)
        return df
    
    def _symbol_to_coingecko_id(self, symbol):
        """Convert trading symbol to CoinGecko ID"""
        symbol_map = {
            "BTCUSDT": "bitcoin",
            "ETHUSDT": "ethereum", 
            "BNBUSDT": "binancecoin",
            "ADAUSDT": "cardano",
            "SOLUSDT": "solana",
            "XRPUSDT": "ripple",
            "DOGEUSDT": "dogecoin",
            "AVAXUSDT": "avalanche-2",
            "MATICUSDT": "matic-network",
            "DOTUSDT": "polkadot",
            "LINKUSDT": "chainlink",
            "UNIUSDT": "uniswap",
            "LTCUSDT": "litecoin",
            "BCHUSDT": "bitcoin-cash",
        }
        
        # Check static map first
        if symbol.upper() in symbol_map:
            return symbol_map.get(symbol.upper())
            
        # Dynamic lookup fallback
        try:
            print(f"Searching CoinGecko ID for {symbol}...")
            # Clean symbol
            search_ticker = symbol.upper().replace("USDT", "").replace("USD", "")
            
            # Fetch coin list
            url = "https://api.coingecko.com/api/v3/coins/list"
            response = requests.get(url, timeout=10) # Use simple request for list
            if response.status_code == 200:
                coin_list = response.json()
                
                # 1. Exact Symbol Match (Prioritize)
                candidates = []
                for coin in coin_list:
                    if coin['symbol'].upper() == search_ticker:
                        # If ID matches symbol exactly (case-insensitive), it's likely the main token
                        if coin['id'].lower() == search_ticker.lower():
                            return coin['id']
                        candidates.append(coin['id'])
                
                if candidates:
                    # Return the shortest ID as a heuristic (e.g. 'pepe' is better than 'baby-pepe')
                    return min(candidates, key=len)
                
                # 2. Fuzzy Match on ID/Name if no exact symbol
                # Limit to top 2000 coins by implicit structure if possible, but list is flat
                # Just fuzzy match extracted ticker against ids
                choices = [c['id'] for c in coin_list]
                best = process.extractOne(search_ticker.lower(), choices)
                if best and best[1] > 90:
                    return best[0]
                    
        except Exception as e:
            print(f"Dynamic ID lookup error: {e}")
            
        return None
    
    def _fetch_coingecko_data(self, coin_id, interval, limit):
        """Fetch data from CoinGecko API"""
        # CoinGecko has different interval options
        days = min(365, limit // 24) if interval in ["1d", "1day"] else min(30, limit // 96)
        
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc?vs_currency=usd&days={days}"
        response = self.make_request_with_fallback(url)
        data = response.json()
        
        # CoinGecko returns [timestamp, open, high, low, close]
        df = pd.DataFrame(data, columns=["timestamp", "Open", "High", "Low", "Close"])
        df['Open Time'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['Volume'] = np.random.uniform(100000, 1000000, len(df))  # Synthetic volume
        df = df[["Open Time", "Open", "High", "Low", "Close", "Volume"]].astype({
            "Open": float, "High": float, "Low": float, "Close": float, "Volume": float
        })
        df.set_index('Open Time', inplace=True)
        
        # Resample to match requested interval if needed
        if len(df) < limit:
            df = self._resample_data(df, interval, limit)
        
        return df.tail(limit)
    
    def _resample_data(self, df, target_interval, target_length):
        """Resample data to create more granular timeframes"""
        if len(df) >= target_length:
            return df
        
        # Simple interpolation method for demo
        expanded_data = []
        for i in range(len(df) - 1):
            current = df.iloc[i]
            next_row = df.iloc[i + 1]
            
            # Add current row
            expanded_data.append(current)
            
            # Add interpolated rows
            steps = min(4, target_length // len(df))  # Create up to 4 sub-intervals
            for step in range(1, steps):
                ratio = step / steps
                interpolated = {
                    'Open': current['Close'],  # Open is previous close
                    'High': current['High'] + ratio * (next_row['High'] - current['High']),
                    'Low': current['Low'] + ratio * (next_row['Low'] - current['Low']),
                    'Close': current['Close'] + ratio * (next_row['Close'] - current['Close']),
                    'Volume': current['Volume'] * (1 + random.uniform(-0.3, 0.3))
                }
                
                # Create timestamp
                time_diff = next_row.name - current.name
                new_time = current.name + (time_diff * ratio)
                
                new_row = pd.Series(interpolated, name=new_time)
                expanded_data.append(new_row)
        
        # Add final row
        if len(df) > 0:
            expanded_data.append(df.iloc[-1])
        
        result_df = pd.DataFrame(expanded_data)
        return result_df.tail(target_length)
    
    def _generate_synthetic_data(self, symbol, interval, limit):
        """Generate realistic synthetic OHLCV data for demo purposes"""
        print(f"Generating synthetic data for {symbol} ({interval}) - {limit} candles")
        
        # Base prices for different symbols
        base_prices = {
            "BTCUSDT": 45000,
            "ETHUSDT": 2800,
            "BNBUSDT": 320,
            "ADAUSDT": 0.85,
            "SOLUSDT": 95,
            "XRPUSDT": 0.62,
            "DOGEUSDT": 0.085,
            "AVAXUSDT": 28,
            "MATICUSDT": 0.95,
            "DOTUSDT": 7.2
        }
        
        base_price = base_prices.get(symbol.upper(), 1.0)
        
        # Generate time series
        interval_minutes = {
            "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
            "1h": 60, "2h": 120, "4h": 240, "6h": 360, "12h": 720, "1d": 1440
        }
        
        minutes = interval_minutes.get(interval, 15)
        end_time = datetime.now()
        timestamps = pd.date_range(
            end=end_time, 
            periods=limit, 
            freq=f"{minutes}min"
        )
        
        # Generate realistic price movement
        np.random.seed(42)  # For reproducible results
        returns = np.random.normal(0, 0.01, limit)  # 1% volatility
        
        # Add some trending behavior
        trend = np.linspace(-0.02, 0.02, limit)  # Slight upward trend
        returns += trend
        
        # Generate price series
        prices = [base_price]
        for i in range(1, limit):
            new_price = prices[-1] * (1 + returns[i])
            prices.append(max(new_price, base_price * 0.1))  # Prevent negative prices
        
        # Generate OHLCV data
        data = []
        for i, (timestamp, price) in enumerate(zip(timestamps, prices)):
            # Generate realistic OHLC from close price
            volatility = random.uniform(0.005, 0.025)  # 0.5% to 2.5% intrabar movement
            
            open_price = prices[i-1] if i > 0 else price
            close_price = price
            
            high_price = max(open_price, close_price) * (1 + random.uniform(0, volatility))
            low_price = min(open_price, close_price) * (1 - random.uniform(0, volatility))
            
            volume = random.uniform(50000, 500000)  # Random volume
            
            data.append({
                "Open Time": timestamp,
                "Open": open_price,
                "High": high_price, 
                "Low": low_price,
                "Close": close_price,
                "Volume": volume
            })
        
        df = pd.DataFrame(data)
        df.set_index('Open Time', inplace=True)
        
        print(f"Generated {len(df)} synthetic candles for {symbol}")
        return df
    
    def fetch_binance_ohlcv(self, symbol="BTCUSDT", interval="15m", limit=1000):
        """Enhanced fetch method with comprehensive fallback system"""
        
        # Try multiple approaches in order of preference
        methods = [
            ("Direct Binance API", self._try_direct_binance),
            ("MEXC API (Fallback)", self._try_mexc_api),
            ("Binance with Rotation", self._try_binance_with_rotation),
            ("Alternative APIs", self._try_alternative_apis),
            ("Synthetic Data", self._generate_synthetic_fallback)
        ]
        
        for method_name, method_func in methods:
            try:
                print(f"Trying {method_name}...")
                df = method_func(symbol, interval, limit)
                if df is not None and len(df) > 50:  # Minimum viable dataset
                    print(f"✅ Success with {method_name}")
                    df.attrs['data_source'] = method_name
                    return df
                else:
                    print(f"❌ {method_name} returned insufficient data")
            except Exception as e:
                print(f"❌ {method_name} failed: {str(e)}")
                continue
        
        # If all methods fail, raise an exception
        raise Exception("All data fetching methods failed. Please check your internet connection.")
    
    def _try_direct_binance(self, symbol, interval, limit):
        """Try direct Binance API call"""
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol.upper()}&interval={interval}&limit={limit}"
        response = self.make_request_with_fallback(url)
        return self._parse_binance_response(response.json())
    
    def _try_binance_with_rotation(self, symbol, interval, limit):
        """Try Binance with header rotation and delays"""
        urls = [
            f"https://api.binance.com/api/v3/klines?symbol={symbol.upper()}&interval={interval}&limit={limit}",
            f"https://api.binance.us/api/v3/klines?symbol={symbol.upper()}&interval={interval}&limit={limit}",
            f"https://api1.binance.com/api/v3/klines?symbol={symbol.upper()}&interval={interval}&limit={limit}",
            f"https://api2.binance.com/api/v3/klines?symbol={symbol.upper()}&interval={interval}&limit={limit}"
        ]
        
        for url in urls:
            try:
                headers = random.choice(self.headers_list)
                response = requests.get(url, headers=headers, timeout=12)
                if response.status_code == 200:
                    return self._parse_binance_response(response.json())
                time.sleep(random.uniform(0.5, 1.5))  # Rate limiting
            except Exception:
                continue
        
        return None
    
    def _try_mexc_api(self, symbol, interval, limit):
        """Try MEXC API as fallback"""
        # Map intervals to MEXC format if needed (MEXC uses 60m instead of 1h)
        mexc_interval_map = {
            "1h": "60m",
            "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
            "4h": "4h", "1d": "1d", "1w": "1W", "1M": "1M"
        }
        mexc_interval = mexc_interval_map.get(interval, interval)
        
        # MEXC symbols usually appended with USDT (e.g. BTCUSDT) - same as input
        url = f"https://api.mexc.com/api/v3/klines?symbol={symbol.upper()}&interval={mexc_interval}&limit={limit}"
        
        response = self.make_request_with_fallback(url)
        return self._parse_mexc_response(response.json())

    def _parse_mexc_response(self, data):
        """Parse MEXC API response (different column count)"""
        # MEXC returns: [Open Time, Open, High, Low, Close, Volume, Close Time, Quote Asset Volume]
        df = pd.DataFrame(data, columns=[
            "Open Time", "Open", "High", "Low", "Close", "Volume",
            "Close Time", "Quote Asset Volume"
        ])
        
        # Convert timestamps and prices
        df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
        df = df[["Open Time", "Open", "High", "Low", "Close", "Volume"]].astype({
            "Open": float, "High": float, "Low": float, "Close": float, "Volume": float
        })
        df.set_index('Open Time', inplace=True)
        return df
    
    def _try_alternative_apis(self, symbol, interval, limit):
        """Try alternative crypto APIs"""
        
        # Method 1: CoinGecko
        try:
            coingecko_id = self._symbol_to_coingecko_id(symbol)
            if coingecko_id:
                return self._fetch_coingecko_data(coingecko_id, interval, limit)
        except Exception as e:
            print(f"CoinGecko failed: {str(e)}")
        
        # Method 2: Try YFinance format (some symbols work)
        try:
            import yfinance as yf
            # Convert symbol format (BTCUSDT -> BTC-USD)
            if symbol.endswith("USDT"):
                yf_symbol = symbol[:-4] + "-USD"
                ticker = yf.Ticker(yf_symbol)
                
                # Map intervals
                yf_interval = {"1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m", 
                              "1h": "1h", "4h": "4h", "1d": "1d"}.get(interval, "15m")
                
                data = ticker.history(period="30d", interval=yf_interval)
                if not data.empty:
                    df = data.reset_index()
                    df.columns = ["Open Time", "Open", "High", "Low", "Close", "Volume"]
                    df.set_index('Open Time', inplace=True)
                    return df.tail(limit)
        except Exception as e:
            print(f"YFinance failed: {str(e)}")
        
        return None
    
    def _generate_synthetic_fallback(self, symbol, interval, limit):
        """Generate synthetic data as final fallback"""
        return self._generate_synthetic_data(symbol, interval, limit)
    
    def add_comprehensive_indicators(self, df):
        """Add comprehensive technical indicators - unchanged from original"""
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']
        
        # Momentum Indicators
        df['RSI_14'] = RSIIndicator(close, window=14).rsi()
        df['RSI_21'] = RSIIndicator(close, window=21).rsi()
        df['Stoch_K'] = StochasticOscillator(high, low, close, window=14).stoch()
        df['Stoch_D'] = StochasticOscillator(high, low, close, window=14).stoch_signal()
        df['Williams_R'] = WilliamsRIndicator(high, low, close).williams_r()
        
        # Trend Indicators
        df['EMA_9'] = EMAIndicator(close, window=9).ema_indicator()
        df['EMA_21'] = EMAIndicator(close, window=21).ema_indicator()
        df['EMA_50'] = EMAIndicator(close, window=50).ema_indicator()
        df['SMA_20'] = SMAIndicator(close, window=20).sma_indicator()
        df['SMA_50'] = SMAIndicator(close, window=50).sma_indicator()
        
        # MACD
        macd = MACD(close)
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Histogram'] = macd.macd_diff()
        
        # ADX and DI
        adx = ADXIndicator(high, low, close)
        df['ADX'] = adx.adx()
        df['DI_Plus'] = adx.adx_pos()
        df['DI_Minus'] = adx.adx_neg()
        
        # Volatility Indicators
        bb = BollingerBands(close, window=20, window_dev=2)
        df['BB_Upper'] = bb.bollinger_hband()
        df['BB_Middle'] = bb.bollinger_mavg()
        df['BB_Lower'] = bb.bollinger_lband()
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle'] * 100
        df['BB_Position'] = (close - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Keltner Channels
        kc = KeltnerChannel(high, low, close)
        df['KC_Upper'] = kc.keltner_channel_hband()
        df['KC_Lower'] = kc.keltner_channel_lband()
        df['KC_Middle'] = kc.keltner_channel_mband()
        
        # ATR and volatility measures
        df['ATR'] = AverageTrueRange(high, low, close).average_true_range()
        df['ATR_Percent'] = (df['ATR'] / close) * 100
        
        # Volume Indicators  
        df['Volume_SMA'] = volume.rolling(window=20).mean()
        df['Volume_Ratio'] = volume / df['Volume_SMA']
        df['OBV'] = OnBalanceVolumeIndicator(close, volume).on_balance_volume()
        df['CMF'] = ChaikinMoneyFlowIndicator(high, low, close, volume).chaikin_money_flow()
        
        # Price Action
        df['Body_Size'] = abs(df['Close'] - df['Open']) / df['Open'] * 100
        df['Upper_Wick'] = (df['High'] - np.maximum(df['Open'], df['Close'])) / df['Open'] * 100
        df['Lower_Wick'] = (np.minimum(df['Open'], df['Close']) - df['Low']) / df['Open'] * 100
        df['Total_Range'] = (df['High'] - df['Low']) / df['Open'] * 100
        
        # Support/Resistance levels (simplified)
        df['Pivot'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['R1'] = 2 * df['Pivot'] - df['Low']
        df['S1'] = 2 * df['Pivot'] - df['High']
        
        # Rate of Change
        df['ROC_5'] = ((close / close.shift(5)) - 1) * 100
        df['ROC_14'] = ((close / close.shift(14)) - 1) * 100
        
        df.dropna(inplace=True)
        return df
    
    def analyze_momentum_confluence(self, row):
        """Analyze momentum indicators for confluences - unchanged from original"""
        confluences = {'bullish': [], 'bearish': [], 'neutral': []}
        
        # RSI Analysis
        if row['RSI_14'] < 30:
            confluences['bullish'].append({
                'indicator': 'RSI (14)',
                'condition': f"Oversold at {row['RSI_14']:.1f}",
                'implication': "Potential bounce or reversal setup. Watch for bullish divergence or break above 30.",
                'strength': 'Medium',
                'timeframe': 'Short-term'
            })
        elif row['RSI_14'] > 70:
            confluences['bearish'].append({
                'indicator': 'RSI (14)',
                'condition': f"Overbought at {row['RSI_14']:.1f}",
                'implication': "Potential pullback or distribution. Watch for bearish divergence or break below 70.",
                'strength': 'Medium',
                'timeframe': 'Short-term'
            })
        elif 45 <= row['RSI_14'] <= 55:
            confluences['neutral'].append({
                'indicator': 'RSI (14)',
                'condition': f"Neutral at {row['RSI_14']:.1f}",
                'implication': "Balanced momentum. Look for directional break above 55 or below 45.",
                'strength': 'Low',
                'timeframe': 'Short-term'
            })
        
        # Stochastic Analysis
        if row['Stoch_K'] < 20 and row['Stoch_D'] < 20:
            confluences['bullish'].append({
                'indicator': 'Stochastic',
                'condition': f"Both %K ({row['Stoch_K']:.1f}) and %D ({row['Stoch_D']:.1f}) oversold",
                'implication': "Strong oversold condition. Potential reversal when %K crosses above %D.",
                'strength': 'Strong' if row['Stoch_K'] > row['Stoch_D'] else 'Medium',
                'timeframe': 'Short-term'
            })
        elif row['Stoch_K'] > 80 and row['Stoch_D'] > 80:
            confluences['bearish'].append({
                'indicator': 'Stochastic',
                'condition': f"Both %K ({row['Stoch_K']:.1f}) and %D ({row['Stoch_D']:.1f}) overbought",
                'implication': "Strong overbought condition. Potential reversal when %K crosses below %D.",
                'strength': 'Strong' if row['Stoch_K'] < row['Stoch_D'] else 'Medium',
                'timeframe': 'Short-term'
            })
        
        # Williams %R Analysis
        if row['Williams_R'] < -80:
            confluences['bullish'].append({
                'indicator': 'Williams %R',
                'condition': f"Oversold at {row['Williams_R']:.1f}",
                'implication': "Potential buying opportunity. Watch for move above -80 for confirmation.",
                'strength': 'Medium',
                'timeframe': 'Short-term'
            })
        elif row['Williams_R'] > -20:
            confluences['bearish'].append({
                'indicator': 'Williams %R',
                'condition': f"Overbought at {row['Williams_R']:.1f}",
                'implication': "Potential selling pressure. Watch for move below -20 for confirmation.",
                'strength': 'Medium',
                'timeframe': 'Short-term'
            })
        
        return confluences
    
    def analyze_trend_confluence(self, row):
        """Analyze trend indicators for confluences - unchanged from original"""
        confluences = {'bullish': [], 'bearish': [], 'neutral': []}
        
        # EMA Alignment
        ema_alignment = "bullish" if row['EMA_9'] > row['EMA_21'] > row['EMA_50'] else "bearish" if row['EMA_9'] < row['EMA_21'] < row['EMA_50'] else "mixed"
        
        if ema_alignment == "bullish":
            confluences['bullish'].append({
                'indicator': 'EMA Alignment',
                'condition': "EMA 9 > EMA 21 > EMA 50",
                'implication': "Strong bullish trend structure. Expect continuation with pullbacks to EMAs as support.",
                'strength': 'Strong',
                'timeframe': 'Medium-term'
            })
        elif ema_alignment == "bearish":
            confluences['bearish'].append({
                'indicator': 'EMA Alignment',
                'condition': "EMA 9 < EMA 21 < EMA 50",
                'implication': "Strong bearish trend structure. Expect continuation with rallies to EMAs as resistance.",
                'strength': 'Strong',
                'timeframe': 'Medium-term'
            })
        
        # Price vs EMAs
        if row['Close'] > row['EMA_21']:
            confluences['bullish'].append({
                'indicator': 'Price vs EMA 21',
                'condition': f"Price {((row['Close']/row['EMA_21']-1)*100):+.2f}% above EMA 21",
                'implication': "Bullish bias maintained. EMA 21 likely to act as dynamic support.",
                'strength': 'Medium',
                'timeframe': 'Short to Medium-term'
            })
        else:
            confluences['bearish'].append({
                'indicator': 'Price vs EMA 21',
                'condition': f"Price {((row['Close']/row['EMA_21']-1)*100):+.2f}% below EMA 21",
                'implication': "Bearish bias maintained. EMA 21 likely to act as dynamic resistance.",
                'strength': 'Medium',
                'timeframe': 'Short to Medium-term'
            })
        
        # MACD Analysis
        if row['MACD'] > row['MACD_Signal'] and row['MACD_Histogram'] > 0:
            confluences['bullish'].append({
                'indicator': 'MACD',
                'condition': "MACD above signal line with positive histogram",
                'implication': "Bullish momentum building. Watch for histogram expansion for stronger moves.",
                'strength': 'Strong' if row['MACD_Histogram'] > 0 else 'Medium',
                'timeframe': 'Medium-term'
            })
        elif row['MACD'] < row['MACD_Signal'] and row['MACD_Histogram'] < 0:
            confluences['bearish'].append({
                'indicator': 'MACD',
                'condition': "MACD below signal line with negative histogram",
                'implication': "Bearish momentum building. Watch for histogram expansion for stronger moves.",
                'strength': 'Strong' if row['MACD_Histogram'] < 0 else 'Medium',
                'timeframe': 'Medium-term'
            })
        
        # ADX Trend Strength
        if row['ADX'] > 25:
            trend_direction = "bullish" if row['DI_Plus'] > row['DI_Minus'] else "bearish"
            confluences[trend_direction].append({
                'indicator': 'ADX Trend Strength',
                'condition': f"Strong trending market (ADX: {row['ADX']:.1f})",
                'implication': f"Strong {trend_direction} trend in place. Expect trend continuation with minor pullbacks.",
                'strength': 'Strong' if row['ADX'] > 40 else 'Medium',
                'timeframe': 'Medium to Long-term'
            })
        elif row['ADX'] < 20:
            confluences['neutral'].append({
                'indicator': 'ADX Trend Strength',
                'condition': f"Weak trending market (ADX: {row['ADX']:.1f})",
                'implication': "Market in consolidation/ranging phase. Look for breakout setups.",
                'strength': 'Medium',
                'timeframe': 'All timeframes'
            })
        
        return confluences
    
    def analyze_volatility_confluence(self, row):
        """Analyze volatility and mean reversion indicators (unchanged from original)"""
        confluences = {'bullish': [], 'bearish': [], 'neutral': []}
        
        # Bollinger Bands Analysis
        bb_pos = row['BB_Position']
        if bb_pos < 0.1:  # Near lower band
            confluences['bullish'].append({
                'indicator': 'Bollinger Bands',
                'condition': f"Price near lower band (Position: {bb_pos:.2f})",
                'implication': "Potential mean reversion setup. Watch for bounce off lower band or breakdown.",
                'strength': 'Medium',
                'timeframe': 'Short-term'
            })
        elif bb_pos > 0.9:  # Near upper band
            confluences['bearish'].append({
                'indicator': 'Bollinger Bands',
                'condition': f"Price near upper band (Position: {bb_pos:.2f})",
                'implication': "Potential mean reversion setup. Watch for rejection at upper band or breakout.",
                'strength': 'Medium',
                'timeframe': 'Short-term'
            })
        
        # Bollinger Band Width
        if row['BB_Width'] < 2:  # Low volatility
            confluences['neutral'].append({
                'indicator': 'Bollinger Band Width',
                'condition': f"Low volatility environment (Width: {row['BB_Width']:.2f}%)",
                'implication': "Squeeze condition. Expect volatility expansion and potential breakout soon.",
                'strength': 'Strong',
                'timeframe': 'Short to Medium-term'
            })
        elif row['BB_Width'] > 8:  # High volatility
            confluences['neutral'].append({
                'indicator': 'Bollinger Band Width',
                'condition': f"High volatility environment (Width: {row['BB_Width']:.2f}%)",
                'implication': "Volatility expansion phase. Expect potential reversion to mean.",
                'strength': 'Medium',
                'timeframe': 'Short-term'
            })
        
        # ATR Analysis
        if row['ATR_Percent'] > 3:
            confluences['neutral'].append({
                'indicator': 'Average True Range',
                'condition': f"High volatility (ATR: {row['ATR_Percent']:.2f}%)",
                'implication': "Elevated volatility. Use wider stops and smaller position sizes.",
                'strength': 'Medium',
                'timeframe': 'All timeframes'
            })
        
        return confluences
    
    def analyze_volume_confluence(self, row):
        """Analyze volume-based confluences (unchanged from original)"""
        confluences = {'bullish': [], 'bearish': [], 'neutral': []}
        
        # Volume Analysis
        if row['Volume_Ratio'] > 1.5:
            confluences['neutral'].append({
                'indicator': 'Volume',
                'condition': f"Above average volume ({row['Volume_Ratio']:.1f}x normal)",
                'implication': "Strong participation. Moves likely to be more sustainable.",
                'strength': 'Strong' if row['Volume_Ratio'] > 2 else 'Medium',
                'timeframe': 'Short-term'
            })
        elif row['Volume_Ratio'] < 0.7:
            confluences['neutral'].append({
                'indicator': 'Volume',
                'condition': f"Below average volume ({row['Volume_Ratio']:.1f}x normal)",
                'implication': "Low participation. Moves may lack conviction and sustainability.",
                'strength': 'Medium',
                'timeframe': 'Short-term'
            })
        
        # Chaikin Money Flow
        if row['CMF'] > 0.2:
            confluences['bullish'].append({
                'indicator': 'Chaikin Money Flow',
                'condition': f"Strong buying pressure (CMF: {row['CMF']:.2f})",
                'implication': "Money flowing into the asset. Supports bullish bias.",
                'strength': 'Strong' if row['CMF'] > 0.3 else 'Medium',
                'timeframe': 'Medium-term'
            })
        elif row['CMF'] < -0.2:
            confluences['bearish'].append({
                'indicator': 'Chaikin Money Flow',
                'condition': f"Strong selling pressure (CMF: {row['CMF']:.2f})",
                'implication': "Money flowing out of the asset. Supports bearish bias.",
                'strength': 'Strong' if row['CMF'] < -0.3 else 'Medium',
                'timeframe': 'Medium-term'
            })
        
        return confluences
    
    def analyze_price_action(self, row):
        """Analyze price action patterns (unchanged from original)"""
        confluences = {'bullish': [], 'bearish': [], 'neutral': []}
        
        # Candle Analysis
        if row['Body_Size'] > 2:  # Large body
            candle_type = "bullish" if row['Close'] > row['Open'] else "bearish"
            confluences[candle_type].append({
                'indicator': 'Price Action',
                'condition': f"Large {candle_type} candle (Body: {row['Body_Size']:.2f}%)",
                'implication': f"Strong {candle_type} conviction. Expect follow-through in next few candles.",
                'strength': 'Strong' if row['Body_Size'] > 3 else 'Medium',
                'timeframe': 'Short-term'
            })
        
        # Wick Analysis
        if row['Upper_Wick'] > row['Body_Size'] * 2 and row['Close'] > row['Open']:
            confluences['bearish'].append({
                'indicator': 'Price Action - Wicks',
                'condition': f"Long upper wick on bullish candle (Wick: {row['Upper_Wick']:.2f}%)",
                'implication': "Rejection at highs despite bullish close. Potential resistance area.",
                'strength': 'Medium',
                'timeframe': 'Short-term'
            })
        
        if row['Lower_Wick'] > row['Body_Size'] * 2 and row['Close'] < row['Open']:
            confluences['bullish'].append({
                'indicator': 'Price Action - Wicks',
                'condition': f"Long lower wick on bearish candle (Wick: {row['Lower_Wick']:.2f}%)",
                'implication': "Support found at lows despite bearish close. Potential support area.",
                'strength': 'Medium',
                'timeframe': 'Short-term'
            })
        
        return confluences
    
    def generate_comprehensive_analysis(self, df):
        """Generate comprehensive market analysis with enhanced directional bias"""
        latest_row = df.iloc[-1]
        
        # Gather all confluences
        momentum_conf = self.analyze_momentum_confluence(latest_row)
        trend_conf = self.analyze_trend_confluence(latest_row)
        volatility_conf = self.analyze_volatility_confluence(latest_row)
        volume_conf = self.analyze_volume_confluence(latest_row)
        price_action_conf = self.analyze_price_action(latest_row)
        
        # NEW: Enhanced analysis components
        divergence_conf = self.detect_momentum_divergence(df, latest_row)
        pattern_conf = self.analyze_price_action_patterns(df, latest_row)
        
        # Ensure all confluence dictionaries have 'neutral' key
        for conf_dict in [divergence_conf, pattern_conf]:
            if 'neutral' not in conf_dict:
                conf_dict['neutral'] = []
        
        # Combine all confluences
        all_confluences = {
            'bullish': (momentum_conf['bullish'] + trend_conf['bullish'] + 
                       volatility_conf['bullish'] + volume_conf['bullish'] + 
                       price_action_conf['bullish'] + divergence_conf['bullish'] + 
                       pattern_conf['bullish']),
            'bearish': (momentum_conf['bearish'] + trend_conf['bearish'] + 
                       volatility_conf['bearish'] + volume_conf['bearish'] + 
                       price_action_conf['bearish'] + divergence_conf['bearish'] + 
                       pattern_conf['bearish']),
            'neutral': (momentum_conf['neutral'] + trend_conf['neutral'] + 
                       volatility_conf['neutral'] + volume_conf['neutral'] + 
                       price_action_conf['neutral'] + divergence_conf['neutral'] + 
                       pattern_conf['neutral'])
        }
        
        # NEW: Add advanced trend strength to confluences
        advanced_trend = self.calculate_advanced_trend_strength(df, latest_row)
        all_confluences['advanced_trend'] = advanced_trend
        
        return all_confluences, latest_row
    
    def calculate_confluence_strength(self, confluences):
        """Calculate overall confluence strength with enhanced directional bias"""
        # Enhanced indicator weights based on predictive power
        indicator_weights = {
            'MACD': 1.5,
            'RSI (14)': 1.3,
            'Stochastic': 1.2,
            'EMA Alignment': 1.4,
            'Price vs EMA 21': 1.1,
            'ADX Trend Strength': 1.3,
            'Bollinger Bands': 1.0,
            'Volume': 0.8,  # Lower weight as volume is confirmatory
            'Price Action': 1.2,
            'Williams %R': 1.1,
            'ML Support Zone': 1.6,  # ML-learned levels have higher weight
            'ML Resistance Zone': 1.6,  # ML-learned levels have higher weight
        }
        
        # Calculate weighted scores
        bullish_score = 0
        bearish_score = 0
        neutral_score = 0
        
        for conf in confluences['bullish']:
            base_weight = {'Strong': 3, 'Medium': 2, 'Low': 1}[conf['strength']]
            indicator_weight = indicator_weights.get(conf['indicator'], 1.0)
            bullish_score += base_weight * indicator_weight
        
        for conf in confluences['bearish']:
            base_weight = {'Strong': 3, 'Medium': 2, 'Low': 1}[conf['strength']]
            indicator_weight = indicator_weights.get(conf['indicator'], 1.0)
            bearish_score += base_weight * indicator_weight
        
        for conf in confluences['neutral']:
            base_weight = {'Strong': 3, 'Medium': 2, 'Low': 1}[conf['strength']]
            indicator_weight = indicator_weights.get(conf['indicator'], 1.0)
            neutral_score += base_weight * indicator_weight
        
        total_score = bullish_score + bearish_score + neutral_score
        
        if total_score == 0:
            return "No Clear Signal", 0
        
        # Calculate directional bias with enhanced scoring
        bullish_percentage = (bullish_score / total_score) * 100 if total_score > 0 else 0
        bearish_percentage = (bearish_score / total_score) * 100 if total_score > 0 else 0
        
        # Enhanced threshold based on weighted signals
        enhanced_threshold = self.confluence_threshold * 1.2  # Increased threshold for clearer signals
        
        # Calculate difference to determine if there's a clear directional bias
        difference = abs(bullish_score - bearish_score)
        total_directional = bullish_score + bearish_score
        
        if total_directional == 0:
            return "No Clear Signal", 0
        
        # Require a meaningful difference to establish bias
        min_difference_for_bias = total_directional * 0.20  # Need 20% difference to claim bias
        
        if bullish_score > bearish_score and bullish_score >= enhanced_threshold and difference >= min_difference_for_bias:
            # Calculate confidence based on dominance ratio
            dominance_ratio = bullish_score / max(bearish_score, 1)
            # Apply dominance boost but cap it to prevent unrealistic confidence
            base_strength = (bullish_score / total_score) * 100
            dominance_boost = min(25, (dominance_ratio - 1) * 15)  # Max 25% boost
            bias_strength = min(90, base_strength + dominance_boost)
            return "Bullish Bias", bias_strength
        elif bearish_score > bullish_score and bearish_score >= enhanced_threshold and difference >= min_difference_for_bias:
            # Calculate confidence based on dominance ratio
            dominance_ratio = bearish_score / max(bullish_score, 1)
            # Apply dominance boost but cap it to prevent unrealistic confidence
            base_strength = (bearish_score / total_score) * 100
            dominance_boost = min(25, (dominance_ratio - 1) * 15)  # Max 25% boost
            bias_strength = min(90, base_strength + dominance_boost)
            return "Bearish Bias", bias_strength
        else:
            # Mixed signal with calculated strength
            dominant_side = max(bullish_score, bearish_score)
            # Reduce confidence significantly for mixed signals
            bias_strength = (dominant_side / total_score) * 50 if total_score > 0 else 0  # Cap at 50% for mixed
            return "Mixed/Neutral", bias_strength
    
    def detect_momentum_divergence(self, df, latest_row):
        """Detect momentum divergence between price and oscillators"""
        divergences = {'bullish': [], 'bearish': []}
        
        # Calculate recent highs/lows for price
        recent_highs = df['High'].rolling(window=20).max()
        recent_lows = df['Low'].rolling(window=20).min()
        
        # Get last 5 values for comparison
        last_5_highs = recent_highs.tail(5)
        last_5_lows = recent_lows.tail(5)
        last_5_rsi = df['RSI_14'].tail(5)
        last_5_macd = df['MACD'].tail(5)
        
        # Bullish divergence: Price makes lower low, RSI/MACD makes higher low
        if (last_5_lows.iloc[-1] < last_5_lows.iloc[-3] and 
            last_5_rsi.iloc[-1] > last_5_rsi.iloc[-3]):
            divergences['bullish'].append({
                'indicator': 'RSI Divergence',
                'condition': 'Price made lower low but RSI made higher low',
                'implication': 'Potential bullish reversal signal',
                'strength': 'Strong',
                'timeframe': 'Short-term'
            })
        
        if (last_5_lows.iloc[-1] < last_5_lows.iloc[-3] and 
            last_5_macd.iloc[-1] > last_5_macd.iloc[-3]):
            divergences['bullish'].append({
                'indicator': 'MACD Divergence',
                'condition': 'Price made lower low but MACD made higher low',
                'implication': 'Potential bullish reversal signal',
                'strength': 'Strong',
                'timeframe': 'Short-term'
            })
        
        # Bearish divergence: Price makes higher high, RSI/MACD makes lower high
        if (last_5_highs.iloc[-1] > last_5_highs.iloc[-3] and 
            last_5_rsi.iloc[-1] < last_5_rsi.iloc[-3]):
            divergences['bearish'].append({
                'indicator': 'RSI Divergence',
                'condition': 'Price made higher high but RSI made lower high',
                'implication': 'Potential bearish reversal signal',
                'strength': 'Strong',
                'timeframe': 'Short-term'
            })
        
        if (last_5_highs.iloc[-1] > last_5_highs.iloc[-3] and 
            last_5_macd.iloc[-1] < last_5_macd.iloc[-3]):
            divergences['bearish'].append({
                'indicator': 'MACD Divergence',
                'condition': 'Price made higher high but MACD made lower high',
                'implication': 'Potential bearish reversal signal',
                'strength': 'Strong',
                'timeframe': 'Short-term'
            })
        
        return divergences
    
    def analyze_price_action_patterns(self, df, latest_row):
        """Analyze specific price action patterns for directional bias"""
        patterns = {'bullish': [], 'bearish': []}
        
        # Get last few candles
        last_candles = df.tail(10)
        
        # Look for engulfing patterns
        if len(last_candles) >= 2:
            prev_candle = last_candles.iloc[-2]
            curr_candle = last_candles.iloc[-1]
            
            # Bullish engulfing: previous red candle completely engulfed by green candle
            if (prev_candle['Close'] < prev_candle['Open'] and  # Previous red candle
                curr_candle['Close'] > curr_candle['Open'] and  # Current green candle
                curr_candle['Open'] < prev_candle['Close'] and  # Opens below previous close
                curr_candle['Close'] > prev_candle['Open']):   # Closes above previous open
                patterns['bullish'].append({
                    'indicator': 'Engulfing Pattern',
                    'condition': 'Bullish engulfing pattern detected',
                    'implication': 'Strong bullish reversal signal',
                    'strength': 'Strong',
                    'timeframe': 'Short-term'
                })
            
            # Bearish engulfing: previous green candle completely engulfed by red candle
            if (prev_candle['Close'] > prev_candle['Open'] and  # Previous green candle
                curr_candle['Close'] < curr_candle['Open'] and  # Current red candle
                curr_candle['Open'] > prev_candle['Close'] and  # Opens above previous close
                curr_candle['Close'] < prev_candle['Open']):   # Closes below previous open
                patterns['bearish'].append({
                    'indicator': 'Engulfing Pattern',
                    'condition': 'Bearish engulfing pattern detected',
                    'implication': 'Strong bearish reversal signal',
                    'strength': 'Strong',
                    'timeframe': 'Short-term'
                })
        
        # Look for pin bars (potential reversal signals)
        for i in range(1, len(last_candles)):
            candle = last_candles.iloc[-i]
            body_size = abs(candle['Close'] - candle['Open'])
            total_range = candle['High'] - candle['Low']
            
            if total_range > 0:
                upper_wick = candle['High'] - max(candle['Open'], candle['Close'])
                lower_wick = min(candle['Open'], candle['Close']) - candle['Low']
                
                # Bullish pin bar (hammer): long lower wick, small body, little or no upper wick
                if lower_wick > body_size * 2 and upper_wick < body_size:
                    patterns['bullish'].append({
                        'indicator': 'Pin Bar',
                        'condition': f'Bullish hammer pattern at ${candle["Low"]:.4f}',
                        'implication': 'Potential bullish reversal at support',
                        'strength': 'Medium',
                        'timeframe': 'Short-term'
                    })
                
                # Bearish pin bar (shooting star): long upper wick, small body, little or no lower wick
                elif upper_wick > body_size * 2 and lower_wick < body_size:
                    patterns['bearish'].append({
                        'indicator': 'Pin Bar',
                        'condition': f'Bearish shooting star pattern at ${candle["High"]:.4f}',
                        'implication': 'Potential bearish reversal at resistance',
                        'strength': 'Medium',
                        'timeframe': 'Short-term'
                    })
        
        return patterns
    
    def calculate_advanced_trend_strength(self, df, latest_row):
        """Calculate advanced trend strength considering multiple factors"""
        # EMA alignment strength
        ema_9 = latest_row['EMA_9']
        ema_21 = latest_row['EMA_21']
        ema_50 = latest_row['EMA_50']
        
        ema_alignment_score = 0
        if ema_9 > ema_21 > ema_50:
            ema_alignment_score = 30  # Perfect bullish alignment
        elif ema_9 < ema_21 < ema_50:
            ema_alignment_score = 30  # Perfect bearish alignment
        elif ema_9 > ema_21 or ema_21 > ema_50:
            ema_alignment_score = 15  # Partial alignment
        else:
            ema_alignment_score = 0  # No alignment
        
        # ADX strength (trend strength)
        adx = latest_row['ADX']
        adx_score = min(30, max(0, (adx - 20) * 0.75))  # Only score if ADX > 20
        
        # Directional strength (DI+ vs DI-)
        di_plus = latest_row['DI_Plus']
        di_minus = latest_row['DI_Minus']
        di_diff = abs(di_plus - di_minus)
        di_score = min(20, di_diff * 0.2)
        
        # Price vs EMAs (position in trend)
        price = latest_row['Close']
        ema_20 = latest_row['SMA_20']
        price_position_score = min(10, abs((price - ema_20) / ema_20) * 500)  # How far from trend
        
        # Momentum (MACD histogram growth)
        macd_hist = latest_row['MACD_Histogram']
        if len(df) > 1:
            prev_macd_hist = df['MACD_Histogram'].iloc[-2]
            hist_growth = abs(macd_hist - prev_macd_hist)
            momentum_score = min(10, hist_growth * 100)
        else:
            momentum_score = 5
        
        total_score = ema_alignment_score + adx_score + di_score + price_position_score + momentum_score
        
        # Determine trend direction
        trend_direction = "Bullish" if (ema_9 > ema_21 and di_plus > di_minus and price > ema_20) else "Bearish"
        
        return {
            "score": round(total_score, 1),
            "direction": trend_direction,
            "components": {
                "ema_alignment": ema_alignment_score,
                "adx_strength": adx_score,
                "di_strength": di_score,
                "price_position": price_position_score,
                "momentum": momentum_score
            }
        }
    
    def classify_market_regime(self, df, latest_row):
        """Classify the current market regime for context"""
        adx = latest_row['ADX']
        atr_percent = latest_row['ATR_Percent']
        bb_width = latest_row['BB_Width']
        price = latest_row['Close']
        ema_50 = latest_row['EMA_50']
        
        # Check EMA alignment
        ema_bullish = latest_row['EMA_9'] > latest_row['EMA_21'] > latest_row['EMA_50']
        ema_bearish = latest_row['EMA_9'] < latest_row['EMA_21'] < latest_row['EMA_50']
        
        # Determine regime
        if adx > 25:
            if price > ema_50 and ema_bullish:
                regime = "Trending Bull"
                confidence = min(100, (adx / 50) * 100)
                description = "Strong uptrend with clear directional momentum"
            elif price < ema_50 and ema_bearish:
                regime = "Trending Bear"
                confidence = min(100, (adx / 50) * 100)
                description = "Strong downtrend with clear directional momentum"
            else:
                regime = "Trending Mixed"
                confidence = (adx / 50) * 70
                description = "Trending but with conflicting signals"
        elif adx < 20 and bb_width < 2:
            regime = "Ranging/Consolidation"
            confidence = 80
            description = "Low volatility, sideways price action, potential breakout setup"
        elif atr_percent > 3 or bb_width > 8:
            regime = "Volatile"
            confidence = min(100, (atr_percent / 5) * 100)
            description = "High volatility environment, use caution with position sizing"
        else:
            regime = "Transitional"
            confidence = 50
            description = "Market in transition between regimes"
        
        return {
            "regime": regime,
            "confidence": round(confidence, 1),
            "description": description,
            "metrics": {
                "adx": round(adx, 1),
                "atr_percent": round(atr_percent, 2),
                "bb_width": round(bb_width, 2)
            }
        }
    
    def calculate_trend_strength(self, latest_row):
        """Calculate trend strength score (0-100)"""
        score = 0
        components = {}
        
        # ADX Component (0-50 points)
        adx = latest_row['ADX']
        adx_score = min(50, (adx / 50) * 50)
        score += adx_score
        components['adx'] = round(adx_score, 1)
        
        # EMA Alignment Component (0-30 points)
        ema_9 = latest_row['EMA_9']
        ema_21 = latest_row['EMA_21']
        ema_50 = latest_row['EMA_50']
        
        if ema_9 > ema_21 > ema_50:
            ema_score = 30  # Perfect bullish alignment
        elif ema_9 < ema_21 < ema_50:
            ema_score = 30  # Perfect bearish alignment
        elif ema_9 > ema_21 or ema_21 > ema_50:
            ema_score = 15  # Partial alignment
        else:
            ema_score = 0  # No alignment
        
        score += ema_score
        components['ema_alignment'] = round(ema_score, 1)
        
        # Momentum Consistency Component (0-20 points)
        rsi = latest_row['RSI_14']
        macd = latest_row['MACD']
        macd_signal = latest_row['MACD_Signal']
        stoch_k = latest_row['Stoch_K']
        
        momentum_signals = 0
        if rsi > 50 and macd > macd_signal and stoch_k > 50:
            momentum_signals = 3  # All bullish
        elif rsi < 50 and macd < macd_signal and stoch_k < 50:
            momentum_signals = 3  # All bearish
        elif (rsi > 50 and macd > macd_signal) or (rsi < 50 and macd < macd_signal):
            momentum_signals = 2  # Partial agreement
        else:
            momentum_signals = 1  # Mixed
        
        momentum_score = (momentum_signals / 3) * 20
        score += momentum_score
        components['momentum_consistency'] = round(momentum_score, 1)
        
        # Determine level
        if score >= 70:
            level = "Very Strong"
        elif score >= 50:
            level = "Strong"
        elif score >= 30:
            level = "Moderate"
        else:
            level = "Weak"
        
        return {
            "score": round(score, 1),
            "level": level,
            "components": components
        }
    
    def analyze_volume_profile(self, df, latest_row):
        """Analyze volume patterns for accumulation/distribution"""
        cmf = latest_row['CMF']
        volume_ratio = latest_row['Volume_Ratio']
        
        # Get recent price action (last 5 candles)
        recent_df = df.tail(5)
        price_change = ((recent_df['Close'].iloc[-1] - recent_df['Close'].iloc[0]) / recent_df['Close'].iloc[0]) * 100
        
        # Determine profile
        if cmf > 0.2 and volume_ratio > 1.2 and price_change > 0:
            profile = "Accumulation"
            strength = "Strong" if cmf > 0.3 else "Moderate"
            description = "Smart money buying, volume supporting upward price movement"
        elif cmf < -0.2 and volume_ratio > 1.2 and price_change < 0:
            profile = "Distribution"
            strength = "Strong" if cmf < -0.3 else "Moderate"
            description = "Smart money selling, volume supporting downward price movement"
        elif volume_ratio > 1.5:
            profile = "High Activity"
            strength = "Strong"
            description = "Elevated volume but mixed signals, potential volatility"
        elif volume_ratio < 0.7:
            profile = "Low Activity"
            strength = "Weak"
            description = "Below average volume, moves may lack conviction"
        else:
            profile = "Neutral"
            strength = "Moderate"
            description = "Balanced volume, no clear accumulation or distribution"
        
        return {
            "profile": profile,
            "strength": strength,
            "description": description,
            "indicators": {
                "cmf": round(cmf, 3),
                "volume_ratio": round(volume_ratio, 2),
                "price_change_5d": round(price_change, 2)
            }
        }
    
    def build_reasoning_chain(self, df, latest_row, confluences, bias, strength):
        """Build step-by-step reasoning chain showing how bias was determined"""
        chain = []
        
        # Step 1: Market Regime
        regime_data = self.classify_market_regime(df, latest_row)
        chain.append({
            "step": 1,
            "category": "Market Regime",
            "finding": f"Market is in {regime_data['regime']} mode",
            "impact": regime_data['description']
        })
        
        # Step 2: Trend Strength
        trend_data = self.calculate_trend_strength(latest_row)
        chain.append({
            "step": 2,
            "category": "Trend Strength",
            "finding": f"Trend strength is {trend_data['level']} ({trend_data['score']}/100)",
            "impact": f"ADX at {trend_data['components']['adx']}, EMA alignment contributes {trend_data['components']['ema_alignment']} points"
        })
        
        # Step 3: Volume Confirmation
        volume_data = self.analyze_volume_profile(df, latest_row)
        chain.append({
            "step": 3,
            "category": "Volume Profile",
            "finding": f"Volume shows {volume_data['profile']} pattern",
            "impact": volume_data['description']
        })
        
        # Step 4: Key Confluences (top 3-5)
        bullish_count = len(confluences['bullish'])
        bearish_count = len(confluences['bearish'])
        
        # Get strongest signals
        all_signals = []
        for conf in confluences['bullish'][:3]:
            all_signals.append(f"✅ {conf['indicator']}: {conf['condition']}")
        for conf in confluences['bearish'][:3]:
            all_signals.append(f"❌ {conf['indicator']}: {conf['condition']}")
        
        chain.append({
            "step": 4,
            "category": "Signal Confluence",
            "finding": f"{bullish_count} bullish vs {bearish_count} bearish signals detected",
            "impact": " | ".join(all_signals[:5])  # Top 5 signals
        })
        
        # Step 5: Final Bias Determination
        bias_explanation = ""
        if "Bullish" in bias:
            bias_explanation = f"Bullish bias confirmed with {strength:.1f}% confidence due to confluence of upward signals"
        elif "Bearish" in bias:
            bias_explanation = f"Bearish bias confirmed with {strength:.1f}% confidence due to confluence of downward signals"
        else:
            bias_explanation = f"Mixed/Neutral bias with {strength:.1f}% confidence - conflicting signals suggest caution"
        
        chain.append({
            "step": 5,
            "category": "Final Determination",
            "finding": bias,
            "impact": bias_explanation
        })
        
        return chain
    
    def display_analysis(self, symbol, timeframe, confluences, latest_row):
        """Display comprehensive analysis results (unchanged from original)"""
        print(f"\n{'='*80}")
        print(f"🔍 NUNNO'S ENHANCED TECHNICAL ANALYSIS - {symbol} ({timeframe})")
        print(f"{'='*80}")
        print(f"📅 Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"💰 Current Price: ${latest_row['Close']:.4f}")
        print(f"📊 24h Range: ${latest_row['Low']:.4f} - ${latest_row['High']:.4f}")
        
        # Overall Market Bias
        bias, strength = self.calculate_confluence_strength(confluences)
        print(f"\n🎯 OVERALL MARKET BIAS: {bias} ({strength:.1f}% confidence)")
        
        # Bullish Confluences
        if confluences['bullish']:
            print(f"\n🟢 BULLISH CONFLUENCES ({len(confluences['bullish'])} signals):")
            print("-" * 60)
            for i, conf in enumerate(confluences['bullish'], 1):
                print(f"{i}. {conf['indicator']} [{conf['strength']}] - {conf['timeframe']}")
                print(f"   🔍 Condition: {conf['condition']}")
                print(f"   💡 Implication: {conf['implication']}")
                print()
        
        # Bearish Confluences
        if confluences['bearish']:
            print(f"\n🔴 BEARISH CONFLUENCES ({len(confluences['bearish'])} signals):")
            print("-" * 60)
            for i, conf in enumerate(confluences['bearish'], 1):
                print(f"{i}. {conf['indicator']} [{conf['strength']}] - {conf['timeframe']}")
                print(f"   🔍 Condition: {conf['condition']}")
                print(f"   💡 Implication: {conf['implication']}")
                print()
        
        # Neutral/Mixed Signals
        if confluences['neutral']:
            print(f"\n🟡 NEUTRAL/MIXED SIGNALS ({len(confluences['neutral'])} signals):")
            print("-" * 60)
            for i, conf in enumerate(confluences['neutral'], 1):
                print(f"{i}. {conf['indicator']} [{conf['strength']}] - {conf['timeframe']}")
                print(f"   🔍 Condition: {conf['condition']}")
                print(f"   💡 Implication: {conf['implication']}")
                print()
        
        # Key Levels
        print(f"\n📊 KEY LEVELS:")
        print(f"   Pivot Point: ${latest_row['Pivot']:.4f}")
        print(f"   Resistance 1: ${latest_row['R1']:.4f}")
        print(f"   Support 1: ${latest_row['S1']:.4f}")
        print(f"   BB Upper: ${latest_row['BB_Upper']:.4f}")
        print(f"   BB Lower: ${latest_row['BB_Lower']:.4f}")
        print(f"   EMA 21: ${latest_row['EMA_21']:.4f}")
        print(f"   EMA 50: ${latest_row['EMA_50']:.4f}")
        
        # Risk Management
        atr_value = latest_row['ATR']
        print(f"\n⚠️ RISK MANAGEMENT:")
        print(f"   ATR: ${atr_value:.4f} ({latest_row['ATR_Percent']:.2f}%)")
        print(f"   Suggested Stop Distance: ${atr_value * 1.5:.4f}")
        print(f"   Volatility Level: {'High' if latest_row['ATR_Percent'] > 3 else 'Medium' if latest_row['ATR_Percent'] > 1.5 else 'Low'}")
        
        print(f"\n{'='*80}")
        print("⚡ Remember: This analysis is for educational purposes. Always use proper risk management!")
        print(f"{'='*80}")

def user_input_token():
    """Enhanced token selection with more options (unchanged from original)"""
    options = [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT", 
        "XRPUSDT", "DOGEUSDT", "AVAXUSDT", "MATICUSDT", "DOTUSDT",
        "LINKUSDT", "UNIUSDT", "LTCUSDT", "BCHUSDT", "FILUSDT"
    ]
    print("\n🪙 Select a token to analyze:")
    for i, token in enumerate(options[:10], start=1):
        print(f"{i:2d}. {token}")
    print(f"11. More tokens...")
    print(f"12. Enter custom token")
    
    choice = input("\nYour choice: ").strip()
    
    if choice.isdigit():
        choice_num = int(choice)
        if 1 <= choice_num <= 10:
            return options[choice_num-1]
        elif choice_num == 11:
            print("\n📋 Additional tokens:")
            for i, token in enumerate(options[10:], start=11):
                print(f"{i:2d}. {token}")
            sub_choice = input("Select token: ").strip()
            if sub_choice.isdigit() and 11 <= int(sub_choice) <= len(options):
                return options[int(sub_choice)-1]
        elif choice_num == 12:
            custom = input("Enter custom token symbol (e.g., ATOMUSDT): ").upper().strip()
            if custom.endswith('USDT'):
                return custom
            else:
                return custom + 'USDT'
    
    print("Invalid choice. Defaulting to BTCUSDT.")
    return "BTCUSDT"

def user_input_timeframe():
    """Enhanced timeframe selection (unchanged from original)"""
    tf_options = {
        "1": ("1m", "1 Minute - Scalping"),
        "2": ("3m", "3 Minute - Short Scalping"), 
        "3": ("5m", "5 Minute - Scalping"),
        "4": ("15m", "15 Minute - Short Term"),
        "5": ("30m", "30 Minute - Short Term"),
        "6": ("1h", "1 Hour - Medium Term"),
        "7": ("2h", "2 Hour - Medium Term"),
        "8": ("4h", "4 Hour - Swing Trading"),
        "9": ("6h", "6 Hour - Swing Trading"),
        "10": ("12h", "12 Hour - Position"),
        "11": ("1d", "Daily - Position Trading")
    }
    
    print("\n⏰ Select a timeframe:")
    for key, (tf, description) in tf_options.items():
        print(f"{key:2s}. {tf:3s} - {description}")
    
    choice = input("\nYour choice: ").strip()
    selected = tf_options.get(choice, ("15m", "15 Minute - Short Term"))
    return selected[0]

def generate_trading_plan(confluences, latest_row, bias, strength):
    """Generate a structured trading plan based on confluences - returns formatted string"""
    lines = []
    lines.append("## 📋 TRADING PLAN SUGGESTIONS")
    lines.append("")
    
    atr = latest_row['ATR']
    current_price = latest_row['Close']
    
    if bias == "Bullish Bias" and strength > 60:
        lines.append("### 🎯 BULLISH SETUP IDENTIFIED")
        lines.append("")
        lines.append(f"- **Entry Strategy:** Look for pullbacks to EMA 21 (${latest_row['EMA_21']:.4f}) or BB Middle")
        lines.append(f"- **Stop Loss:** Below EMA 50 (${latest_row['EMA_50']:.4f}) or ${atr*1.5:.4f} below entry")
        lines.append(f"- **Target 1:** Pivot R1 (${latest_row['R1']:.4f})")
        lines.append(f"- **Target 2:** BB Upper Band (${latest_row['BB_Upper']:.4f})")
        lines.append(f"- **Risk/Reward:** Aim for 1:2 minimum ratio")
        
    elif bias == "Bearish Bias" and strength > 60:
        lines.append("### 🎯 BEARISH SETUP IDENTIFIED")
        lines.append("")
        lines.append(f"- **Entry Strategy:** Look for rallies to EMA 21 (${latest_row['EMA_21']:.4f}) or BB Middle")
        lines.append(f"- **Stop Loss:** Above EMA 50 (${latest_row['EMA_50']:.4f}) or ${atr*1.5:.4f} above entry")
        lines.append(f"- **Target 1:** Pivot S1 (${latest_row['S1']:.4f})")
        lines.append(f"- **Target 2:** BB Lower Band (${latest_row['BB_Lower']:.4f})")
        lines.append(f"- **Risk/Reward:** Aim for 1:2 minimum ratio")
        
    else:
        lines.append("### ⚖️ MIXED/RANGING MARKET")
        lines.append("")
        lines.append(f"- **Strategy:** Range trading between key levels")
        lines.append(f"- **Buy Zone:** Near BB Lower (${latest_row['BB_Lower']:.4f}) or Support")
        lines.append(f"- **Sell Zone:** Near BB Upper (${latest_row['BB_Upper']:.4f}) or Resistance")
        lines.append(f"- **Stop Loss:** Beyond range boundaries + ${atr:.4f}")
        lines.append(f"- **Wait for:** Clear breakout with volume confirmation")
    
    lines.append("")
    lines.append("### ⚠️ RISK MANAGEMENT RULES")
    lines.append("")
    lines.append(f"- **Position Size:** Risk only 1-2% of capital per trade")
    lines.append(f"- **ATR Stop:** ${atr:.4f} (Current volatility measure)")
    lines.append(f"- **Volume Confirmation:** Wait for volume > {latest_row['Volume_SMA']:.0f}")
    lines.append(f"- **Time Filter:** Avoid news events and low liquidity hours")
    
    # Also print for console compatibility
    plan_text = "\n".join(lines)
    print(plan_text)
    
    return plan_text


class EnhancedCryptoPredictor:
    def __init__(self):
        self.analyzer = TradingAnalyzer()
        self.tokenomics = ComprehensiveTokenomics() if TOKENOMICS_AVAILABLE else None
        
        # Try to get API key from environment
        twitter_token = os.environ.get("TWITTER_BEARER_TOKEN")
        self.social = CryptoSocialScraper(twitter_token) if SOCIAL_AVAILABLE else None
        
    def analyze_token(self, token_symbol: str, timeframe: str = "15m", status_callback=None):
        """
        Perform a comprehensive 3-layer analysis in PARALLEL:
        1. Tokenomics (Fundamentals)
        2. Social Sentiment (Market Psychology)
        3. Technical Analysis (Price Action)
        """
        import concurrent.futures
        
        results = {
            "token": token_symbol,
            "timestamp": datetime.now().isoformat(),
            "layers": {}
        }
        
        print(f"\n🚀 STARTING PARALLEL ANALYSIS FOR {token_symbol}...")
        if status_callback: status_callback(f"🚀 Starting parallel analysis for {token_symbol}...")
        
        # --- Define Worker Functions ---
        
        def run_tokenomics():
            if not self.tokenomics: return None
            try:
                print(f"   📚 Layer 1: Tokenomics started...")
                coin_id = self.analyzer._symbol_to_coingecko_id(token_symbol)
                if coin_id:
                    data = self.tokenomics.fetch_comprehensive_token_data(coin_id)
                    if data:
                        print("   ✅ Tokenomics DONE")
                        return data
                print("   ⚠️ Tokenomics failed or emptry")
                return None
            except Exception as e:
                print(f"   ❌ Tokenomics Error: {e}")
                return None

        def run_social():
            if not self.social: return None
            try:
                print(f"   🧠 Layer 2: Social started...")
                data = self.social.get_comprehensive_social_sentiment(token_symbol)
                if data:
                    print("   ✅ Social DONE")
                    return data
                print("   ⚠️ Social failed")
                return None
            except Exception as e:
                print(f"   ❌ Social Error: {e}")
                return None

        def run_technical():
            try:
                print(f"   📈 Layer 3: Technical started...")
                # Fetch OHLCV
                df = self.analyzer.fetch_coingecko_ohlcv(symbol=token_symbol, interval=timeframe, limit=1000) if hasattr(self.analyzer, 'fetch_coingecko_ohlcv') else self.analyzer.fetch_binance_ohlcv(symbol=token_symbol, interval=timeframe)
                
                # Add indicators
                df = self.analyzer.add_comprehensive_indicators(df)
                
                # Generate TA signals
                confluences, latest_row = self.analyzer.generate_comprehensive_analysis(df)
                bias, strength = self.analyzer.calculate_confluence_strength(confluences)
                
                # Generate trading plan
                trading_plan = generate_trading_plan(confluences, latest_row, bias, strength)
                
                print("   ✅ Technical DONE")
                return {
                    "current_price": latest_row['Close'],
                    "bias": bias,
                    "strength": strength,
                    "confluences": confluences,
                    "latest_data": latest_row.to_dict() if latest_row is not None else None,
                    "plan": trading_plan,
                    "key_levels": {
                        "support": latest_row['S1'],
                        "resistance": latest_row['R1'],
                        "pivot": latest_row['Pivot']
                    },
                    "risk": {
                        "atr": latest_row['ATR'],
                        "volatility": latest_row['ATR_Percent']
                    },
                    "is_synthetic": getattr(df, "attrs", {}).get("data_source") == "Synthetic Data",
                    "dataframe": df  # Include DataFrame for chart generation
                }
            except Exception as e:
                print(f"   ❌ Tech Analysis Error: {e}")
                return None

        # --- Execute in Parallel ---
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_tok = executor.submit(run_tokenomics)
            future_soc = executor.submit(run_social)
            future_tech = executor.submit(run_technical)
            
            # Wait for all (or collect as they complete)
            # We want to populate results dict
            
            tok_res = future_tok.result()
            if tok_res: results["layers"]["tokenomics"] = tok_res
            if status_callback: status_callback("✅ Fundamentals loaded...")
            
            soc_res = future_soc.result()
            if soc_res: results["layers"]["social"] = soc_res
            if status_callback: status_callback("✅ Sentiment loaded...")
            
            tech_res = future_tech.result()
            if tech_res: 
                results["layers"]["technical"] = tech_res
            else:
                # Fallback for tech failure
                results["layers"]["technical"] = {"bias": "Indeterminate", "strength": 0, "plan": "Analysis Failed"}
            if status_callback: status_callback("✅ Technical analysis loaded...")

        print("🏁 All layers complete.")
        return results

    def generate_explanation(self, results):
        """Generate a natural language explanation of the results"""
        token = results.get("token", "Unknown")
        layers = results.get("layers", {})
        
        explanation = []
        explanation.append(f"# 🔮 Nunno Prediction: {token}")
        explanation.append("")
        
        # Fundamental Summary
        if "tokenomics" in layers:
            tok = layers["tokenomics"]
            explanation.append("## 🏗️ Fundamentals (Tokenomics)")
            explanation.append(f"- **Market Cap:** {tok.get('Market_Cap', 'N/A')} ({tok.get('Market_Cap_Category', 'N/A')})")
            explanation.append(f"- **Supply:** {tok.get('Circulating_Percentage', 'N/A')} circulating")
            explanation.append(f"- **Risk assessment:** {tok.get('Risk_Level', 'N/A')} ({tok.get('Risk_Score', '0')})")
            explanation.append("")
            
        # Social Summary
        if "social" in layers:
            soc = layers["social"]
            explanation.append("## 🧠 Market Sentiment")
            
            # Fear & Greed
            fg = soc.get("fear_greed_index", {})
            explanation.append(f"- **Fear & Greed:** {fg.get('value', 'N/A')} ({fg.get('classification', 'N/A')})")
            
            # Twitter Sentiment
            tw = soc.get("twitter_sentiment", {})
            explanation.append(f"- **Social Mood:** {tw.get('sentiment', 'Neutral')} ({tw.get('confidence', 0)}% confidence)")
            
            # Whale Alerts
            whale = soc.get("whale_alerts", {})
            if whale.get("status") == "No Data":
                explanation.append("- **Whale Activity:** Data Unavailable (API Key Required)")
            else:
               explanation.append(f"- **Whale Activity:** {whale.get('status', 'N/A')}")
               if whale.get("summary"):
                   explanation.append(f"  * {whale.get('summary')}")
            
            explanation.append("")
            
        # Technical Summary
        if "technical" in layers:
            ta = layers["technical"]
            explanation.append(f"## 📈 Technical Outlook")
            explanation.append(f"**Bias:** {ta['bias']} ({ta['strength']:.1f}% strength)")
            explanation.append(f"- **Price:** ${ta['current_price']:.4f}")
            explanation.append(f"- **Key Support:** ${ta['key_levels']['support']:.4f}")
            explanation.append(f"- **Key Resistance:** ${ta['key_levels']['resistance']:.4f}")
            
            # Trading Plan - Use the full plan text
            explanation.append("")
            if ta.get('plan'):
                explanation.append(ta['plan'])
            else:
                 explanation.append("### 📋 Suggested Plan")
                 explanation.append("Plan generation unavailable.")
                
        return "\n".join(explanation)

def main():
    """Enhanced main program with 3-layer analysis"""
    
    try:
        print("🚀 Welcome to Nunno's 3-Layer Prediction Module")
        print("=" * 70)
        
        # Get user inputs (reuse existing input functions if available, or simple input)
        # Assuming user_input_token and user_input_timeframe are defined in the file
        token = user_input_token()
        timeframe = user_input_timeframe()
        
        predictor = EnhancedCryptoPredictor()
        
        # Run Analysis
        results = predictor.analyze_token(token, timeframe)
        
        # Display Tokenomics & Social Data (if available)
        if "tokenomics" in results["layers"]:
            print("\n📊 TOKENOMICS SNAPSHOT:")
            for k, v in list(results["layers"]["tokenomics"].items())[:5]:
                print(f"  • {k}: {v}")
                
        if "social" in results["layers"]:
            print("\n🧠 SENTIMENT SNAPSHOT:")
            fg = results["layers"]["social"].get("fear_greed_index", {})
            print(f"  • Fear & Greed: {fg.get('value')} ({fg.get('classification')})")
            
        # Display Technical Analysis (using existing display function)
        if "technical" in results["layers"]:
            # Reconstruct minimal context for display_analysis
            # Note: This is an approximation since we processed inside the class
            # For full detail we might want to refactor display_analysis, but for now:
            print("\n" + "="*40)
            print("TECHNICAL ANALYSIS RESULTS")
            print("="*40)
            ta = results["layers"]["technical"]
            print(f"Bias: {ta['bias']}")
            print(f"Price: {ta['current_price']}")
            
        # Generate and print full explanation
        print("\n" + "="*70)
        explanation = predictor.generate_explanation(results)
        print(explanation)
        print("="*70)
        
    except KeyboardInterrupt:
        print(f"\n\n🛑 Analysis interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        print(f"\n👋 Thank you for using Nunno's Prediction Module!")

if __name__ == "__main__":
    main()