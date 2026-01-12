"""
Chat Service with OpenRouter Integration
Orchestrates Claude Opus 4.5 with tool calling
"""

import os
import json
import requests
from typing import List, Dict, Optional, AsyncGenerator
import asyncio
from openai import AsyncOpenAI

class ChatService:
    """
    Chat orchestration service using Claude Opus 4.5 via OpenRouter
    Implements the "Empathetic Expert" persona with automatic beginner notes
    """
    
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY", "")
        self.model = os.getenv("AI_MODEL", "openai/gpt-4o-mini")
        self.fallback_model = "openai/gpt-4o-mini"  # Small, efficient fallback
        self.base_url = "https://openrouter.ai/api/v1"
        
        # Initialize OpenAI Async Client for OpenRouter
        self.client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )
        
        # Import services for tool calling
        from services.technical_analysis import TechnicalAnalysisService
        from services.tokenomics_service import TokenomicsService
        from services.news_service import NewsService
        from services.web_research_service import WebResearchService
        
        self.technical_service = TechnicalAnalysisService()
        self.tokenomics_service = TokenomicsService()
        self.news_service = NewsService()
        self.web_research_service = WebResearchService()
    
    def _get_system_prompt(self, user_name: str, user_age: int) -> str:
        """Condensed system prompt for token efficiency"""
        return f"""You are Nunno, an empathetic financial educator for beginners.

CORE RULES:
1. FINANCE ONLY: You are a specialized financial assistant. If a user asks about non-financial topics (e.g., cooking, coding, general knowledge), politely decline and steer them back to finance.
   - Example: "I'm designed to help you with finance and investing. Let's look at the markets instead!"

2. USE YOUR TOOLS: You have powerful built-in tools for real-time analysis. USE THEM implicitly when helpful.
   - Technical Analysis: For price predictions, trends, and chart signals.
   - Tokenomics: For coin details, market cap, and supply.
   - News: For market sentiment and recent events.
   - Web Research: For finding specific information online.

3. EXPLAIN SIMPLY: Explain jargon in parentheses: e.g. "RSI at 70 (RSI is a market thermometer - 70 means overbought)".

Persona: {user_name}, Age {user_age}. Helpful, use analogies, be concise (2-3 paragraphs).
Output: Use headings, bullets, emojis. No financial advice, just education.
Founder: Mujtaba Kazmi."""
    
    async def process_message(
        self,
        message: str,
        user_name: str = "User",
        user_age: int = 18,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict:
        """
        Process a chat message with tool calling orchestration (Non-streaming)
        """
        if not self.api_key:
            return {
                "response": "⚠️ OpenRouter API key not configured. Please add OPENROUTER_API_KEY to your .env file.",
                "tool_calls": [],
                "data_used": {}
            }
        
        # Build messages
        messages = [{"role": "system", "content": self._get_system_prompt(user_name, user_age)}]
        
        if conversation_history:
            messages.extend(conversation_history[-2:]) # Aggressive history trimming
        
        messages.append({"role": "user", "content": message})
        
        # Detect tools
        tools_to_call = await self._classify_intent_and_extract_entities(message)
        tool_data = {}
        
        # Execute tool calls
        if tools_to_call:
            for tool_name, params in tools_to_call:
                if tool_name == "technical_analysis":
                    tool_data["technical"] = self.technical_service.analyze(params["ticker"], params.get("interval", "15m"))
                elif tool_name == "tokenomics":
                    coin_id = params["coin_id"].lower()
                    tool_data["tokenomics"] = self.tokenomics_service.analyze(coin_id, params.get("amount", 1000))
                elif tool_name == "news":
                    tool_data["news"] = self.news_service.get_news_sentiment(params["ticker"])
                elif tool_name == "web_research":
                    if "url" in params:
                        tool_data["web_research"] = self.web_research_service.scrape_url(params["url"])
                    else:
                        tool_data["web_research"] = self.web_research_service.search_web(params["query"])
            
            tool_context = self._format_tool_context(tool_data)
            messages.append({
                "role": "user",
                "content": f"Here's the data from my analysis tools:\n\n{tool_context}\n\nPlease synthesize this information and explain it in beginner-friendly terms. Remember to add Beginner's Notes for all technical terms!"
            })
        
        # Call OpenRouter via OpenAI SDK
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=2000,
                temperature=0.7,
                extra_headers={
                    "HTTP-Referer": "https://nunno.finance",
                    "X-Title": "Nunno Finance"
                }
            )
            
            ai_response = response.choices[0].message.content
            
            return {
                "response": ai_response,
                "tool_calls": [tool[0] for tool in tools_to_call] if tools_to_call else [],
                "data_used": tool_data
            }
            
        except Exception as e:
            return {
                "response": f"I'm having trouble connecting to my brain right now. Error: {str(e)}",
                "tool_calls": [],
                "data_used": {}
            }
    
    async def stream_message(
        self,
        message: str,
        user_name: str = "User",
        user_age: int = 18,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream chat responses for real-time display using AsyncOpenAI
        Optimized to run tools in parallel and provide immediate feedback
        """
        if not self.api_key:
            yield f"data: {json.dumps({'response': '⚠️ OpenRouter API key not configured.'})}\n\n"
            return
            
        # 1. Detect tools (fast keyword-based detection)
        print(f"[CHAT] Classifying intent for: {message[:50]}...")
        tools_to_call = await self._classify_intent_and_extract_entities(message)
        print(f"[CHAT] Tools detected: {tools_to_call}")
        tool_data = {}
        
        # 2. Execute tools in parallel with immediate status feedback
        if tools_to_call:
            # Yield initial status immediately
            yield f"data: {json.dumps({'type': 'status', 'content': '🔍 Gathering data...'})}\n\n"
            
            # Create async tasks for parallel execution
            async def run_technical_analysis(params):
                try:
                    # Run in thread pool since it's synchronous
                    return await asyncio.get_event_loop().run_in_executor(
                        None, 
                        self.technical_service.analyze, 
                        params["ticker"]
                    )
                except Exception as e:
                    print(f"Technical analysis error: {e}")
                    return None
            
            async def run_tokenomics(params):
                try:
                    coin_id = params.get("coin_id", "").lower()
                    return await asyncio.get_event_loop().run_in_executor(
                        None,
                        self.tokenomics_service.analyze,
                        coin_id
                    )
                except Exception as e:
                    print(f"Tokenomics error: {e}")
                    return None
            
            async def run_news(params):
                try:
                    return await asyncio.get_event_loop().run_in_executor(
                        None,
                        self.news_service.get_news_sentiment,
                        params["ticker"]
                    )
                except Exception as e:
                    print(f"News error: {e}")
                    return None
            
            async def run_web_research(params):
                try:
                    if "url" in params:
                        return await asyncio.get_event_loop().run_in_executor(
                            None,
                            self.web_research_service.scrape_url,
                            params["url"]
                        )
                    else:
                        return await asyncio.get_event_loop().run_in_executor(
                            None,
                            self.web_research_service.search_web,
                            params["query"]
                        )
                except Exception as e:
                    print(f"Web research error: {e}")
                    return None
            
            # Execute all tools in parallel
            tasks = []
            tool_names = []
            for tool_name, params in tools_to_call:
                tool_names.append(tool_name)
                if tool_name == "technical_analysis":
                    tasks.append(run_technical_analysis(params))
                elif tool_name == "tokenomics":
                    tasks.append(run_tokenomics(params))
                elif tool_name == "news":
                    tasks.append(run_news(params))
                elif tool_name == "web_research":
                    tasks.append(run_web_research(params))
            
            # Wait for all tools to complete in parallel
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Map results back to tool names
                for i, (tool_name, params) in enumerate(tools_to_call):
                    result = results[i]
                    if result and not isinstance(result, Exception):
                        # Map tool_name to the key used in tool_data
                        if tool_name == "technical_analysis":
                            tool_data["technical"] = result
                        elif tool_name == "tokenomics":
                            tool_data["tokenomics"] = result
                        elif tool_name == "news":
                            tool_data["news"] = result
                        elif tool_name == "web_research":
                            tool_data["web_research"] = result
            
            # Send tool data once all complete
            if tool_data:
                yield f"data: {json.dumps({'type': 'data', 'tool_calls': tool_names, 'data_used': tool_data})}\n\n"

        # 3. Build messages
        messages = [{"role": "system", "content": self._get_system_prompt(user_name, user_age)}]
        if conversation_history:
            messages.extend(conversation_history[-2:]) # Aggressive history trimming
        
        if tool_data:
            tool_context = self._format_tool_context(tool_data)
            messages.append({"role": "user", "content": message})
            messages.append({
                "role": "user", 
                "content": f"Here is the data found:\n\n{tool_context}\n\nPlease explain this simply with tables and beginner notes."
            })
        else:
            messages.append({"role": "user", "content": message})

        # 4. Stream from OpenRouter using AsyncOpenAI with automatic fallback
        current_model = self.model
        retry_with_fallback = False
        
        try:
            stream = await self.client.chat.completions.create(
                model=current_model,
                messages=messages,
                max_tokens=1200,
                temperature=0.7,
                stream=True,
                extra_headers={
                    "HTTP-Referer": "https://nunno.finance",
                    "X-Title": "Nunno Finance"
                }
            )
            
            async for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield f"data: {json.dumps({'type': 'text', 'content': content})}\n\n"
                                
        except Exception as e:
            error_msg = str(e).lower()
            
            # Check for token-related errors or Rate Limits (429)
            if any(keyword in error_msg for keyword in ["token", "context_length", "max_tokens", "too long", "402", "429", "rate"]):
                # Try fallback to smaller model
                if current_model != self.fallback_model:
                    retry_with_fallback = True
                    yield f"data: {json.dumps({'type': 'status', 'content': f'⚡ Switching to faster model ({self.fallback_model})...'})}\n\n"
                else:
                    # Already using fallback, show error
                    if "402" in str(e):
                        error_msg = "⚠️ Your OpenRouter account balance is too low. Please top up at https://openrouter.ai/settings/credits"
                    else:
                        error_msg = "⚠️ Message too long even for compact model. Please try a shorter message."
                    yield f"data: {json.dumps({'type': 'error', 'content': error_msg})}\n\n"
            elif "401" in str(e):
                error_msg = "🔑 Invalid API Key. Please check your OPENROUTER_API_KEY in the .env file."
                yield f"data: {json.dumps({'type': 'error', 'content': error_msg})}\n\n"
            else:
                # Generic error
                yield f"data: {json.dumps({'type': 'error', 'content': f'Error: {str(e)}'})}\n\n"
        
        # Retry with fallback model if needed
        if retry_with_fallback:
            try:
                stream = await self.client.chat.completions.create(
                    model=self.fallback_model,
                    messages=messages,
                    max_tokens=1000,  # Slightly reduced for fallback
                    temperature=0.7,
                    stream=True,
                    extra_headers={
                        "HTTP-Referer": "https://nunno.finance",
                        "X-Title": "Nunno Finance"
                    }
                )
                
                async for chunk in stream:
                    content = chunk.choices[0].delta.content
                    if content:
                        yield f"data: {json.dumps({'type': 'text', 'content': content})}\n\n"
            
            except Exception as fallback_error:
                error_msg = f"⚠️ Both models failed. Error: {str(fallback_error)}"
                yield f"data: {json.dumps({'type': 'error', 'content': error_msg})}\n\n"
    
    async def _classify_intent_and_extract_entities(self, message: str) -> List[tuple]:
        """
        Use keyword detection first, then LLM if needed for tool calling
        Async version using AsyncOpenAI to prevent blocking
        """
        # First, try keyword-based detection (FREE!)
        fallback_tools = self._detect_tools_needed_fallback(message)
        
        # Check if LLM is needed
        needs_llm = any(tool[0] == "_needs_llm" for tool in fallback_tools)
        has_tools = any(tool[0] != "_needs_llm" for tool in fallback_tools)
        
        # If keyword detection found tools OR detected intent keywords, use LLM to refine
        # If no tools and no intent found by keywords, skip API call entirely (it's just chat)
        if not fallback_tools:
            # print("[INTENT] No tools needed, skipping API call")
            return []
        
        # SPEED OPTIMIZATION: Skip LLM refinement entirely. 
        # Reliance on strict keyword definitions in _detect_tools_needed_fallback
        # This makes tool detection instantaneous (0ms)
        
        # Filter out the "_needs_llm" marker if present, and just return what we have
        final_tools = [t for t in fallback_tools if t[0] != "_needs_llm"]
        
        if final_tools:
             print(f"[INTENT] Fast-path tools detected: {final_tools}")
             return final_tools
             
        # If we only had the marker but no actual tools (e.g. "Predict future"), 
        # we now skip tools to be fast, rather than asking LLM "what tool?".
        # This means some ambiguous queries might become plain chat, which is acceptable for speed.
        return []

    def _detect_tools_needed_fallback(self, message: str) -> List[tuple]:
        """Legacy keyword-based detection as fallback"""
        message_lower = message.lower()
        tools = []
        
        # FAST PATH: Skip tool detection for greetings and simple questions
        # FAST PATH: Skip tool detection for greetings and simple questions
        greetings = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening", "thanks", "thank you"]
        simple_questions = [
            "how are you", "what can you do", "help", "who are you", "who made you", "who created you",
            "what is your name", "tell me a joke", "you are", "are you", "what do you think", "can you"
        ]
        
        # Check for simple matches (exact or contained)
        if len(message.split()) < 15:
            if any(greeting == message_lower or message_lower.startswith(greeting + " ") for greeting in greetings):
                return []
            if any(q in message_lower for q in simple_questions):
                return []
        
        # Check if this message likely needs analysis tools (even if we can't extract the coin)
        has_prediction_intent = any(w in message_lower for w in [
            "price", "chart", "predict", "prediction", "analysis", "analyze", 
            "outlook", "should i", "buy", "sell", "forecast", "trend"
        ])
        
        # Stricter tokenomics intent: "what is" requires extra context to be a crypto query
        has_definite_tokenomics = any(w in message_lower for w in ["tokenomics", "supply", "market cap", "all time high", "ath"])
        has_weak_tokenomics = any(w in message_lower for w in ["what is", "tell me about", "explain"])
        
        # Potential crypto context words to validate a weak intent
        crypto_context = ["coin", "token", "crypto", "project", "protocol", "chain", "network", "defi", "investment", "price", "value"]
        has_crypto_context = any(c in message_lower for c in crypto_context)
        
        # We scan for coins if there's any hint of tokenomics
        should_scan_coins = has_definite_tokenomics or has_weak_tokenomics
        # But we only fallback to LLM if the intent is strictly crypto-related or definite
        has_tokenomics_intent = has_definite_tokenomics or (has_weak_tokenomics and has_crypto_context)

        has_news_intent = any(w in message_lower for w in [
            "news", "sentiment", "what's happening", "why is it moving", "market"
        ])
        
        # Comprehensive fallback ticker mapping (100+ coins)
        common_tickers = {
            # Top Market Cap
            "bitcoin": "BTCUSDT", "btc": "BTCUSDT",
            "ethereum": "ETHUSDT", "eth": "ETHUSDT",
            "solana": "SOLUSDT", "sol": "SOLUSDT",
            "binance": "BNBUSDT", "bnb": "BNBUSDT",
            "ripple": "XRPUSDT", "xrp": "XRPUSDT",
            "cardano": "ADAUSDT", "ada": "ADAUSDT",
            "avalanche": "AVAXUSDT", "avax": "AVAXUSDT",
            "dogecoin": "DOGEUSDT", "doge": "DOGEUSDT",
            "polkadot": "DOTUSDT", "dot": "DOTUSDT",
            "polygon": "MATICUSDT", "matic": "MATICUSDT",
            "shiba": "SHIBUSDT", "shib": "SHIBUSDT", "shiba inu": "SHIBUSDT",
            "tron": "TRXUSDT", "trx": "TRXUSDT",
            "chainlink": "LINKUSDT", "link": "LINKUSDT",
            "uniswap": "UNIUSDT", "uni": "UNIUSDT",
            "cosmos": "ATOMUSDT", "atom": "ATOMUSDT",
            "litecoin": "LTCUSDT", "ltc": "LTCUSDT",
            "ethereum classic": "ETCUSDT", "etc": "ETCUSDT",
            "stellar": "XLMUSDT", "xlm": "XLMUSDT",
            "near": "NEARUSDT", "near protocol": "NEARUSDT",
            "aptos": "APTUSDT", "apt": "APTUSDT",
            "arbitrum": "ARBUSDT", "arb": "ARBUSDT",
            "optimism": "OPUSDT", "op": "OPUSDT",
            "filecoin": "FILUSDT", "fil": "FILUSDT",
            "hedera": "HBARUSDT", "hbar": "HBARUSDT",
            "vechain": "VETUSDT", "vet": "VETUSDT",
            "algorand": "ALGOUSDT", "algo": "ALGOUSDT",
            "internet computer": "ICPUSDT", "icp": "ICPUSDT",
            
            # DeFi Tokens
            "aave": "AAVEUSDT",
            "curve": "CRVUSDT", "crv": "CRVUSDT",
            "maker": "MKRUSDT", "mkr": "MKRUSDT",
            "sushiswap": "SUSHIUSDT", "sushi": "SUSHIUSDT",
            "compound": "COMPUSDT", "comp": "COMPUSDT",
            "synthetix": "SNXUSDT", "snx": "SNXUSDT",
            "graph": "GRTUSDT", "grt": "GRTUSDT", "the graph": "GRTUSDT",
            "yearn": "YFIUSDT", "yfi": "YFIUSDT", "yearn finance": "YFIUSDT",
            "1inch": "1INCHUSDT",
            "loopring": "LRCUSDT", "lrc": "LRCUSDT",
            "thorchain": "RUNEUSDT", "rune": "RUNEUSDT",
            "lido": "LDOUSDT", "ldo": "LDOUSDT",
            "rocket pool": "RPLUSDT", "rpl": "RPLUSDT",
            "gmx": "GMXUSDT",
            "radiant": "RDNTUSDT", "rdnt": "RDNTUSDT",
            "pendle": "PENDLEUSDT",
            "dydx": "DYDXUSDT",
            
            # Gaming & Metaverse
            "sandbox": "SANDUSDT", "sand": "SANDUSDT", "the sandbox": "SANDUSDT",
            "decentraland": "MANAUSDT", "mana": "MANAUSDT",
            "axie": "AXSUSDT", "axs": "AXSUSDT", "axie infinity": "AXSUSDT",
            "enjin": "ENJUSDT", "enj": "ENJUSDT",
            "gala": "GALAUSDT", "gala games": "GALAUSDT",
            "immutable": "IMXUSDT", "imx": "IMXUSDT",
            "magic": "MAGICUSDT", "treasure": "MAGICUSDT",
            "apecoin": "APEUSDT", "ape": "APEUSDT",
            "stepn": "GMTUSDT", "gmt": "GMTUSDT",
            "illuvium": "ILVUSDT", "ilv": "ILVUSDT",
            "alien worlds": "TLMUSDT", "tlm": "TLMUSDT",
            "smooth love potion": "SLPUSDT", "slp": "SLPUSDT",
            "myneighboralice": "ALICEUSDT", "alice": "ALICEUSDT",
            
            # Meme Coins
            "pepe": "PEPEUSDT",
            "floki": "FLOKIUSDT", "floki inu": "FLOKIUSDT",
            "bonk": "BONKUSDT",
            "dogwifhat": "WIFUSDT", "wif": "WIFUSDT",
            "book of meme": "BOMEUSDT", "bome": "BOMEUSDT",
            "memecoin": "MEMEUSDT", "meme": "MEMEUSDT",
            
            # Layer 2 & Scaling
            "starknet": "STRKUSDT", "strk": "STRKUSDT",
            "blur": "BLURUSDT",
            "jito": "JTOUSDT", "jto": "JTOUSDT",
            "jupiter": "JUPUSDT", "jup": "JUPUSDT",
            
            # AI & Data
            "fetch.ai": "FETUSDT", "fet": "FETUSDT", "fetch": "FETUSDT",
            "singularitynet": "AGIXUSDT", "agix": "AGIXUSDT",
            "ocean": "OCEANUSDT", "ocean protocol": "OCEANUSDT",
            "render": "RNDRUSDT", "rndr": "RNDRUSDT", "render token": "RENDERUSDT",
            "worldcoin": "WLDUSDT", "wld": "WLDUSDT",
            "pyth": "PYTHUSDT", "pyth network": "PYTHUSDT",
            
            # Infrastructure
            "fantom": "FTMUSDT", "ftm": "FTMUSDT",
            "harmony": "ONEUSDT", "one": "ONEUSDT",
            "elrond": "EGLDUSDT", "egld": "EGLDUSDT", "multiversx": "EGLDUSDT",
            "theta": "THETAUSDT", "theta network": "THETAUSDT",
            "tezos": "XTZUSDT", "xtz": "TEZOSUSDT",
            "eos": "EOSUSDT",
            "flow": "FLOWUSDT",
            "injective": "INJUSDT", "inj": "INJUSDT",
            "sui": "SUIUSDT",
            "sei": "SEIUSDT",
            "celestia": "TIAUSDT", "tia": "TIAUSDT",
            "stacks": "STXUSDT", "stx": "STXUSDT",
            "kaspa": "KASUSDT", "kas": "KASUSDT",
            "arweave": "ARUSDT", "ar": "ARUSDT",
            "storj": "STORJUSDT",
            "klaytn": "KLAYUSDT", "klay": "KLAYUSDT",
            "mina": "MINAUSDT", "mina protocol": "MINAUSDT",
            "conflux": "CFXUSDT", "cfx": "CFXUSDT",
            "celo": "CELOUSDT",
            "oasis": "ROSEUSDT", "rose": "ROSEUSDT",
            
            # Privacy Coins
            "zcash": "ZECUSDT", "zec": "ZECUSDT",
            "dash": "DASHUSDT",
            "monero": "XMRUSDT", "xmr": "XMRUSDT",
            
            # Other Notable
            "waves": "WAVESUSDT",
            "neo": "NEOUSDT",
            "kava": "KAVAUSDT",
            "zilliqa": "ZILUSDT", "zil": "ZILUSDT",
            "qtum": "QTUMUSDT",
            "basic attention": "BATUSDT", "bat": "BATUSDT",
            "0x": "ZRXUSDT", "zrx": "ZRXUSDT",
            "omg": "OMGUSDT", "omg network": "OMGUSDT",
            "terra": "LUNAUSDT", "luna": "LUNAUSDT",
            "luna classic": "LUNCUSDT", "lunc": "LUNCUSDT",
            "chiliz": "CHZUSDT", "chz": "CHZUSDT",
            "ens": "ENSUSDT", "ens domains": "ENSUSDT",
            "ssv": "SSVUSDT", "ssv network": "SSVUSDT",
            "looksrare": "LOOKSUSDT", "looks": "LOOKSUSDT",
            "woo": "WOOUSDT", "woo network": "WOOUSDT",
            "jasmycoin": "JASMYUSDT", "jasmy": "JASMYUSDT",
            "mask": "MASKUSDT", "mask network": "MASKUSDT",
            "perpetual": "PERPUSDT", "perp": "PERPUSDT",
        }
        
        # Try to extract coin for predictions
        if has_prediction_intent:
             for name, ticker in common_tickers.items():
                if name in message_lower:
                    tools.append(("technical_analysis", {"ticker": ticker}))
                    break
        
        # Try to extract coin for tokenomics
        if should_scan_coins:
             # CoinGecko ID mapping for tokenomics
             coingecko_map = {
                 "bitcoin": "bitcoin", "btc": "bitcoin", "ethereum": "ethereum", "eth": "ethereum",
                 "solana": "solana", "sol": "solana", "binance": "binancecoin", "bnb": "binancecoin",
                 "ripple": "ripple", "xrp": "ripple", "cardano": "cardano", "ada": "cardano",
                 "avalanche": "avalanche-2", "avax": "avalanche-2", "dogecoin": "dogecoin", "doge": "dogecoin",
                 "polkadot": "polkadot", "dot": "polkadot", "polygon": "matic-network", "matic": "matic-network",
                 "shiba": "shiba-inu", "shib": "shiba-inu", "shiba inu": "shiba-inu",
                 "tron": "tron", "trx": "tron", "chainlink": "chainlink", "link": "chainlink",
                 "uniswap": "uniswap", "uni": "uniswap", "cosmos": "cosmos", "atom": "cosmos",
                 "litecoin": "litecoin", "ltc": "litecoin", "ethereum classic": "ethereum-classic", "etc": "ethereum-classic",
                 "stellar": "stellar", "xlm": "stellar", "near": "near", "near protocol": "near",
                 "aptos": "aptos", "apt": "aptos", "arbitrum": "arbitrum", "arb": "arbitrum",
                 "optimism": "optimism", "op": "optimism", "filecoin": "filecoin", "fil": "filecoin",
                 "hedera": "hedera-hashgraph", "hbar": "hedera-hashgraph", "vechain": "vechain", "vet": "vechain",
                 "algorand": "algorand", "algo": "algorand", "internet computer": "internet-computer", "icp": "internet-computer",
                 "aave": "aave", "curve": "curve-dao-token", "crv": "curve-dao-token",
                 "maker": "maker", "mkr": "maker", "sushiswap": "sushi", "sushi": "sushi",
                 "compound": "compound-governance-token", "comp": "compound-governance-token",
                 "synthetix": "havven", "snx": "havven", "graph": "the-graph", "grt": "the-graph", "the graph": "the-graph",
                 "yearn": "yearn-finance", "yfi": "yearn-finance", "1inch": "1inch",
                 "loopring": "loopring", "lrc": "loopring", "thorchain": "thorchain", "rune": "thorchain",
                 "lido": "lido-dao", "ldo": "lido-dao", "rocket pool": "rocket-pool", "rpl": "rocket-pool",
                 "gmx": "gmx", "radiant": "radiant-capital", "rdnt": "radiant-capital",
                 "pendle": "pendle", "dydx": "dydx",
                 "sandbox": "the-sandbox", "sand": "the-sandbox", "the sandbox": "the-sandbox",
                 "decentraland": "decentraland", "mana": "decentraland",
                 "axie": "axie-infinity", "axs": "axie-infinity", "axie infinity": "axie-infinity",
                 "enjin": "enjincoin", "enj": "enjincoin", "gala": "gala", "gala games": "gala",
                 "immutable": "immutable-x", "imx": "immutable-x", "magic": "magic",
                 "apecoin": "apecoin", "ape": "apecoin", "stepn": "stepn", "gmt": "stepn",
                 "illuvium": "illuvium", "ilv": "illuvium", "alien worlds": "alien-worlds", "tlm": "alien-worlds",
                 "smooth love potion": "smooth-love-potion", "slp": "smooth-love-potion",
                 "myneighboralice": "my-neighbor-alice", "alice": "my-neighbor-alice",
                 "pepe": "pepe", "floki": "floki", "floki inu": "floki", "bonk": "bonk",
                 "dogwifhat": "dogwifcoin", "wif": "dogwifcoin", "book of meme": "book-of-meme", "bome": "book-of-meme",
                 "memecoin": "memecoin", "meme": "memecoin", "starknet": "starknet", "strk": "starknet",
                 "blur": "blur", "jito": "jito-governance-token", "jto": "jito-governance-token",
                 "jupiter": "jupiter-exchange-solana", "jup": "jupiter-exchange-solana",
                 "fetch.ai": "fetch-ai", "fet": "fetch-ai", "fetch": "fetch-ai",
                 "singularitynet": "singularitynet", "agix": "singularitynet",
                 "ocean": "ocean-protocol", "ocean protocol": "ocean-protocol",
                 "render": "render-token", "rndr": "render-token", "worldcoin": "worldcoin-wld", "wld": "worldcoin-wld",
                 "pyth": "pyth-network", "fantom": "fantom", "ftm": "fantom",
                 "harmony": "harmony", "one": "harmony", "elrond": "elrond-erd-2", "egld": "elrond-erd-2", "multiversx": "elrond-erd-2",
                 "theta": "theta-token", "tezos": "tezos", "xtz": "tezos", "eos": "eos", "flow": "flow",
                 "injective": "injective-protocol", "inj": "injective-protocol", "sui": "sui", "sei": "sei-network",
                 "celestia": "celestia", "tia": "celestia", "stacks": "blockstack", "stx": "blockstack",
                 "kaspa": "kaspa", "kas": "kaspa", "arweave": "arweave", "ar": "arweave", "storj": "storj",
                 "klaytn": "klay-token", "klay": "klay-token", "mina": "mina-protocol",
                 "conflux": "conflux-token", "cfx": "conflux-token", "celo": "celo",
                 "oasis": "oasis-network", "rose": "oasis-network", "zcash": "zcash", "zec": "zcash",
                 "dash": "dash", "monero": "monero", "xmr": "monero", "waves": "waves", "neo": "neo",
                 "kava": "kava", "zilliqa": "zilliqa", "zil": "zilliqa", "qtum": "qtum",
                 "basic attention": "basic-attention-token", "bat": "basic-attention-token",
                 "0x": "0x", "zrx": "0x", "omg": "omisego", "omg network": "omisego",
                 "terra": "terra-luna-2", "luna": "terra-luna-2", "luna classic": "terra-luna", "lunc": "terra-luna",
                 "chiliz": "chiliz", "chz": "chiliz", "ens": "ethereum-name-service",
                 "ssv": "ssv-network", "looksrare": "looksrare", "looks": "looksrare",
                 "woo": "woo-network", "jasmycoin": "jasmycoin", "jasmy": "jasmycoin",
                 "mask": "mask-network", "perpetual": "perpetual-protocol", "perp": "perpetual-protocol",
             }
             for name in coingecko_map:
                if name in message_lower:
                    tools.append(("tokenomics", {"coin_id": coingecko_map[name]}))
                    break
        
        # IMPORTANT: If we found intent keywords but couldn't extract the coin, 
        # add a marker to signal the LLM should be called
        if (has_prediction_intent or has_tokenomics_intent or has_news_intent) and not tools:
            tools.append(("_needs_llm", {}))  # Special marker
                    
        return tools

    def _format_tool_context(self, tool_data: Dict) -> str:
        """Minified context for token efficiency"""
        context_parts = []
        
        # Optimize Technical Analysis
        if "technical" in tool_data:
            tech = tool_data["technical"].copy()
            # Remove heavy fields not needed by LLM (it should use the pre-computed explanation)
            for k in ["price_history", "beginner_notes", "data_source", "is_synthetic"]: 
                tech.pop(k, None)
            context_parts.append(f"TECH:{json.dumps(tech)}")
            
        # Optimize Tokenomics
        if "tokenomics" in tool_data:
            t = tool_data["tokenomics"]
            if isinstance(t, dict):
                # Keep only essential fields
                s = {k: t.get(v) for k, v in {"name": "Token_Name", "price": "Current_Price", "rank": "Market_Cap_Rank"}.items()}
                context_parts.append(f"TOKEN:{json.dumps(s)}")
                
        # Optimize News
        if "news" in tool_data:
            n = tool_data["news"].copy()
            # Remove redundant text
            for k in ["explanation", "beginner_notes", "headlines"]: n.pop(k, None)
            context_parts.append(f"NEWS:{json.dumps(n)}")
            
        # Optimize Web Research (Fixes dict slicing bug)
        if "web_research" in tool_data:
            wr = tool_data["web_research"]
            if isinstance(wr, dict):
                # Single page scrape result
                summary = {
                    "title": wr.get("title"),
                    "url": wr.get("url"),
                    "content": wr.get("content", "")[:800] + "..." # Truncate to save tokens
                }
                context_parts.append(f"WEB_PAGE:{json.dumps(summary)}")
            elif isinstance(wr, list):
                # Search results list
                top_results = []
                for res in wr[:2]: # Take top 2
                    top_results.append({
                        "title": res.get("title"),
                        "href": res.get("href"),
                        "body": res.get("body", "")[:150] # Truncate body
                    })
                context_parts.append(f"SEARCH_RESULTS:{json.dumps(top_results)}")
                
        return "|".join(context_parts)
