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
        self.coingecko_base = "https://api.coingecko.com/api/v3"

    def get_market_highlights(self) -> Dict[str, Any]:
        """
        Get comprehensive market highlights
        """
        try:
            # Fetch all 24h ticker data
            response = requests.get(f"{self.binance_base}/ticker/24hr", timeout=10)
            if response.status_code != 200:
                logger.error(f"Failed to fetch Binance ticker: {response.status_code}")
                return self._get_empty_highlights()

            data = response.json()
            
            # Filter for USDT pairs only and exclude leveraged tokens if possible
            usdt_pairs = [
                ticker for ticker in data 
                if ticker['symbol'].endswith('USDT') 
                and not ticker['symbol'].endswith('UPUSDT') 
                and not ticker['symbol'].endswith('DOWNUSDT')
            ]

            # Format the data
            formatted_tickers = []
            for t in usdt_pairs:
                try:
                    formatted_tickers.append({
                        'symbol': t['symbol'],
                        'price': float(t['lastPrice']),
                        'change': float(t['priceChangePercent']),
                        'volume': float(t['quoteVolume']),
                        'high': float(t['highPrice']),
                        'low': float(t['lowPrice']),
                        'count': int(t['count'])
                    })
                except (ValueError, KeyError):
                    continue

            # 1. Top Gainers
            gainers = sorted(formatted_tickers, key=lambda x: x['change'], reverse=True)[:10]

            # 2. Top Losers
            losers = sorted(formatted_tickers, key=lambda x: x['change'])[:10]

            # 3. Most Traded (Volume)
            most_traded = sorted(formatted_tickers, key=lambda x: x['volume'], reverse=True)[:10]
            
            # 4. "New" Listings (Mocked/Approximated for now as Binance API doesn't give listing date)
            # In a real app, we might track this in our DB or use a specific Binance announcement scraper
            # For now, we'll return some of the latest popular additions on Binance USDT
            new_listings = [
                {'symbol': 'PYTHUSDT', 'name': 'Pyth Network', 'change': formatted_tickers[0]['change'] if any(t['symbol'] == 'PYTHUSDT' for t in formatted_tickers) else 0},
                {'symbol': 'JUPUSDT', 'name': 'Jupiter', 'change': 1.2},
                {'symbol': 'MANTAUSDT', 'name': 'Manta Network', 'change': -2.5},
                {'symbol': 'ALTUSDT', 'name': 'AltLayer', 'change': 5.7},
                {'symbol': 'ZETAUSDT', 'name': 'ZetaChain', 'change': 12.4},
                {'symbol': 'STRKUSDT', 'name': 'Starknet', 'change': -4.1},
                {'symbol': 'PORTALUSDT', 'name': 'Portal', 'change': 0.8},
                {'symbol': 'AXLUSDT', 'name': 'Axelar', 'change': 15.2},
                {'symbol': 'METISUSDT', 'name': 'Metis', 'change': -3.2},
                {'symbol': 'AEVOUSDT', 'name': 'Aevo', 'change': 2.1}
            ]
            
            # Filter new_listings to ensure they are available in formatted_tickers
            available_symbols = {t['symbol']: t for t in formatted_tickers}
            final_new = []
            for item in new_listings:
                if item['symbol'] in available_symbols:
                    ticker = available_symbols[item['symbol']]
                    final_new.append({
                        'symbol': item['symbol'],
                        'name': item.get('name', item['symbol'].replace('USDT', '')),
                        'price': ticker['price'],
                        'change': ticker['change'],
                        'volume': ticker['volume']
                    })
            
            # If our mock list is empty or small, just take the ones with lowest ID or something (not possible here)
            # Let's just return what we have

            return {
                'gainers': gainers,
                'losers': losers,
                'most_traded': most_traded,
                'new_listings': final_new if final_new else gainers[:5] # Fallback
            }

        except Exception as e:
            logger.error(f"Error in MarketService: {e}")
            return self._get_empty_highlights()

    def _get_empty_highlights(self) -> Dict[str, Any]:
        return {
            'gainers': [],
            'losers': [],
            'most_traded': [],
            'new_listings': []
        }
