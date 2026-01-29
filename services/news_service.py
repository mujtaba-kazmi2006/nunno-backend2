"""
News and Sentiment Service
"""

import requests
import os
from datetime import datetime
from duckduckgo_search import DDGS

class NewsService:
    """
    Market news and sentiment analysis service
    """
    
    def __init__(self):
        try:
            self.news_api_key = os.getenv("NEWS_API_KEY", "")
            self.fear_greed_url = "https://api.alternative.me/fng/"
        except Exception as e:
            print(f"Error initializing NewsService: {e}")
            self.news_api_key = ""
            self.fear_greed_url = "https://api.alternative.me/fng/"
    
    def get_news_sentiment(self, ticker: str):
        """
        Get market news and sentiment
        
        Args:
            ticker: Cryptocurrency ticker
        
        Returns:
            News and sentiment data with beginner explanations
        """
        try:
            # Get Fear & Greed Index
            fear_greed = self._get_fear_greed_index()
            
            # Get news headlines (try API or fallback)
            headlines = self._get_news_headlines(ticker)
            
            # Determine overall sentiment
            sentiment = self._determine_sentiment(fear_greed, headlines)
            
            response = {
                "ticker": ticker,
                "timestamp": datetime.now().isoformat(),
                "fear_greed_index": fear_greed,
                "sentiment": sentiment,
                "headlines": headlines[:5],  # Top 5 headlines
                "explanation": self._create_sentiment_explanation(fear_greed, sentiment),
                "beginner_notes": {
                    "Fear & Greed Index": "This index measures market emotions from 0 (Extreme Fear) to 100 (Extreme Greed). It's like a mood ring for the crypto market!",
                    "Sentiment": "Sentiment is the overall feeling or mood of the market. Positive sentiment means people are optimistic, negative means they're worried.",
                    "News Impact": "News can move markets quickly. Good news often pushes prices up, bad news can cause panic selling."
                }
            }
            
            return response
            
        except Exception as e:
            return {
                "error": str(e),
                "message": "Unable to fetch news and sentiment data."
            }
    
    def _get_fear_greed_index(self):
        """Fetch the crypto Fear & Greed Index"""
        try:
            response = requests.get(self.fear_greed_url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            value = int(data["data"][0]["value"])
            classification = data["data"][0]["value_classification"]
            
            return {
                "value": value,
                "classification": classification,
                "description": self._explain_fear_greed(value)
            }
        except Exception:
            return {
                "value": 50,
                "classification": "Neutral",
                "description": "Unable to fetch current Fear & Greed data"
            }
    
    def _explain_fear_greed(self, value: int) -> str:
        """Explain Fear & Greed value in beginner terms"""
        if value <= 25:
            return "Extreme Fear - People are very worried and selling. This can sometimes be a buying opportunity (like a sale at a store)."
        elif value <= 45:
            return "Fear - The market is nervous. Prices might be lower than usual."
        elif value <= 55:
            return "Neutral - The market is balanced. No strong emotions either way."
        elif value <= 75:
            return "Greed - People are getting excited and buying. Prices might be getting high."
        else:
            return "Extreme Greed - Everyone is very excited and buying. This can be risky (like buying something when it's most expensive)."
    
    def _get_news_headlines(self, ticker: str):
        """Fetch news headlines from multiple free sources"""
        headlines = []
        
        # Extract coin name from ticker
        coin_name = ticker.replace("USDT", "").replace("USD", "").lower()
        
        # Try CryptoPanic API (free, no key needed for public endpoint)
        try:
            print(f"Fetching news from CryptoPanic for {coin_name}...")
            # CryptoPanic public feed
            url = f"https://cryptopanic.com/api/v1/posts/?auth_token=&currencies={coin_name}&public=true"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])[:5]
                
                for item in results:
                    headlines.append({
                        "title": item.get('title', ''),
                        "source": item.get('source', {}).get('title', 'CryptoPanic'),
                        "published": item.get('published_at', datetime.now().isoformat()),
                        "url": item.get('url', '')
                    })
                
                if headlines:
                    print(f"✅ Found {len(headlines)} headlines from CryptoPanic")
                    return headlines
        except Exception as e:
            print(f"CryptoPanic failed: {e}")
        
        # Try CoinGecko trending/news (free API)
        try:
            print(f"Fetching trending from CoinGecko...")
            # Map common tickers to CoinGecko IDs
            coin_id_map = {
                'btc': 'bitcoin',
                'eth': 'ethereum',
                'sol': 'solana',
                'bnb': 'binancecoin',
                'xrp': 'ripple',
                'ada': 'cardano',
                'doge': 'dogecoin',
                'dot': 'polkadot',
                'matic': 'polygon',
                'shib': 'shiba-inu',
                'link': 'chainlink',
                'ltc': 'litecoin',
                'pepe': 'pepe'
            }
            
            coin_id = coin_id_map.get(coin_name, coin_name)
            
            # Get trending coins and news
            trending_url = "https://api.coingecko.com/api/v3/search/trending"
            response = requests.get(trending_url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                trending_coins = data.get('coins', [])
                
                # Create headlines from trending data
                for item in trending_coins[:3]:
                    coin_data = item.get('item', {})
                    coin_symbol = coin_data.get('symbol', '').lower()
                    
                    if coin_symbol == coin_name or coin_data.get('id') == coin_id:
                        headlines.append({
                            "title": f"{coin_data.get('name')} is trending #{item.get('score', 0)+1} on CoinGecko",
                            "source": "CoinGecko Trending",
                            "published": datetime.now().isoformat(),
                            "url": f"https://www.coingecko.com/en/coins/{coin_data.get('id')}"
                        })
                
                if headlines:
                    print(f"✅ Found {len(headlines)} trending items from CoinGecko")
        except Exception as e:
            print(f"CoinGecko trending failed: {e}")
        
        # Enhanced DuckDuckGo search with better queries
        try:
            print(f"Fetching news via DuckDuckGo for {coin_name}...")
            
            # Try multiple search strategies
            search_queries = [
                f"{coin_name} cryptocurrency news today",
                f"{coin_name} price analysis",
                f"{ticker} market update"
            ]
            
            for search_term in search_queries:
                try:
                    with DDGS() as ddgs:
                        results = list(ddgs.text(
                            search_term, 
                            max_results=3,
                            region='wt-wt',  # Worldwide
                            safesearch='off',
                            timelimit='d'  # Last day
                        ))
                    
                    for r in results:
                        title = r.get("title", "")
                        # Filter out irrelevant results
                        if title and (coin_name in title.lower() or ticker.lower() in title.lower()):
                            headlines.append({
                                "title": title,
                                "source": r.get("source", "Web Search"),
                                "published": datetime.now().isoformat(),
                                "url": r.get("href", "")
                            })
                    
                    if headlines:
                        break  # Stop if we found results
                        
                except Exception as query_error:
                    print(f"DDG query '{search_term}' failed: {query_error}")
                    continue
            
            if headlines:
                print(f"✅ Found {len(headlines)} headlines from DuckDuckGo")
                return headlines[:5]  # Limit to 5
                
        except Exception as e:
            print(f"DDG News Search failed: {e}")
        
        # If all sources fail, return generic market update
        if not headlines:
            print("⚠️ All news sources failed, returning generic update")
            headlines = [{
                "title": f"{coin_name.upper()} market continues to show volatility as traders monitor key support and resistance levels",
                "source": "Market Analysis",
                "published": datetime.now().isoformat(),
                "url": f"https://www.coingecko.com/en/coins/{coin_name}"
            }]
        
        return headlines
    
    def _determine_sentiment(self, fear_greed: dict, headlines: list) -> str:
        """Determine overall sentiment"""
        fg_value = fear_greed.get("value", 50)
        
        if fg_value <= 30:
            return "Bearish"
        elif fg_value <= 45:
            return "Slightly Bearish"
        elif fg_value <= 55:
            return "Neutral"
        elif fg_value <= 70:
            return "Slightly Bullish"
        else:
            return "Bullish"
    
    def _create_sentiment_explanation(self, fear_greed: dict, sentiment: str) -> str:
        """Create beginner-friendly sentiment explanation"""
        fg_value = fear_greed.get("value", 50)
        fg_class = fear_greed.get("classification", "Neutral")
        
        explanation = f"The market sentiment is currently **{sentiment}**. "
        explanation += f"The Fear & Greed Index is at {fg_value}/100 ({fg_class}). "
        explanation += fear_greed.get("description", "")
        
        return explanation
