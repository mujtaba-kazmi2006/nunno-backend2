import requests
import logging
from typing import List, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class MarketService:
    """
    Handles fetching market-wide data:
    - Top Gainers
    - Top Losers
    - New Listings (Approximated or fetched)
    - Most Traded
    """
    
    def __init__(self):
        self.binance_base = "https://api.binance.com/api/v3"
        self.binance_us_base = "https://api.binance.us/api/v3"
        self.mexc_base = "https://api.mexc.com/api/v3"
        try:
            from services.cache_service import cache_service
            self.cache = cache_service
        except:
            self.cache = None

    def get_market_highlights(self) -> Dict[str, Any]:
        """
        Get comprehensive market highlights with multiple API fallbacks
        """
        # Check cache (1 minute TTL is enough for market highlights to feel fresh)
        if self.cache:
            cached = self.cache.get("market_highlights")
            if cached:
                return cached

        # Try Primary Binance
        highlights = self._fetch_from_binance(self.binance_base)
        if highlights['gainers']:
            if self.cache: self.cache.set("market_highlights", highlights, ttl_seconds=60)
            return highlights

        # Try Binance US fallback
        logger.info("Main Binance API failed or returned empty. Trying Binance US...")
        highlights = self._fetch_from_binance(self.binance_us_base)
        if highlights['gainers']:
            if self.cache: self.cache.set("market_highlights", highlights, ttl_seconds=60)
            return highlights

        # Try MEXC fallback
        logger.info("Binance US API failed. Trying MEXC...")
        highlights = self._fetch_from_mexc()
        if highlights['gainers']:
            if self.cache: self.cache.set("market_highlights", highlights, ttl_seconds=60)
            return highlights

        return self._get_empty_highlights()

    def _fetch_from_binance(self, base_url: str) -> Dict[str, Any]:
        try:
            response = requests.get(f"{base_url}/ticker/24hr", timeout=8)
            if response.status_code != 200:
                return self._get_empty_highlights()

            data = response.json()
            usdt_pairs = [
                ticker for ticker in data 
                if ticker['symbol'].endswith('USDT') 
                and not any(x in ticker['symbol'] for x in ['UPUSDT', 'DOWNUSDT', 'BEARUSDT', 'BULLUSDT'])
            ]

            formatted = []
            for t in usdt_pairs:
                try:
                    formatted.append({
                        'symbol': t['symbol'],
                        'price': float(t['lastPrice']),
                        'change': float(t['priceChangePercent']),
                        'volume': float(t['quoteVolume']),
                    })
                except: continue

            return self._process_formatted_data(formatted)
        except Exception as e:
            logger.error(f"Binance fetch error ({base_url}): {e}")
            return self._get_empty_highlights()

    def _fetch_from_mexc(self) -> Dict[str, Any]:
        try:
            # MEXC v3 ticker is very similar to Binance
            response = requests.get(f"{self.mexc_base}/ticker/24hr", timeout=8)
            if response.status_code != 200:
                return self._get_empty_highlights()

            data = response.json()
            formatted = []
            for t in data:
                if t['symbol'].endswith('USDT'):
                    try:
                        formatted.append({
                            'symbol': t['symbol'],
                            'price': float(t['lastPrice']),
                            'change': float(t['priceChangePercent']),
                            'volume': float(t['quoteVolume']),
                        })
                    except: continue

            return self._process_formatted_data(formatted)
        except Exception as e:
            logger.error(f"MEXC fetch error: {e}")
            return self._get_empty_highlights()

    def _process_formatted_data(self, formatted_tickers: List[Dict]) -> Dict[str, Any]:
        if not formatted_tickers:
            return self._get_empty_highlights()

        # Sort and slice
        gainers = sorted(formatted_tickers, key=lambda x: x['change'], reverse=True)[:10]
        losers = sorted(formatted_tickers, key=lambda x: x['change'])[:10]
        most_traded = sorted(formatted_tickers, key=lambda x: x['volume'], reverse=True)[:10]
        
        # New Listings (Mocked/Static list with live data update)
        new_listings_seeds = ['PYTHUSDT', 'JUPUSDT', 'MANTAUSDT', 'ALTUSDT', 'ZETAUSDT', 'STRKUSDT', 'PORTALUSDT', 'AXLUSDT']
        available = {t['symbol']: t for t in formatted_tickers}
        final_new = []
        for sym in new_listings_seeds:
            if sym in available:
                t = available[sym]
                final_new.append({
                    'symbol': sym,
                    'name': sym.replace('USDT', ''),
                    'price': t['price'],
                    'change': t['change'],
                    'volume': t['volume']
                })

        return {
            'gainers': gainers,
            'losers': losers,
            'most_traded': most_traded,
            'new_listings': final_new if final_new else gainers[:5]
        }

    def _get_empty_highlights(self) -> Dict[str, Any]:
        return {
            'gainers': [],
            'losers': [],
            'most_traded': [],
            'new_listings': []
        }
