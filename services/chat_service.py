"""
Chat Service with OpenRouter Integration
MODIFIED: Multi-user history, tier-based models, and token usage tracking.
"""

import os
import json
from typing import List, Dict, Optional, AsyncGenerator
import asyncio
from openai import AsyncOpenAI
from sqlalchemy.orm import Session
from database import User, Conversation, Message
from services.usage_service import log_token_usage, get_tier_config

class ChatService:
    """
    Production-ready Chat Service
    Handles:
    1. Tier-based model selection
    2. DB-backed conversation history (Privacy)
    3. Token usage tracking (Scaling)
    """
    
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY", "")
        self.base_url = "https://openrouter.ai/api/v1"
        
        self.client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )
        
        from services.technical_analysis import TechnicalAnalysisService
        from services.tokenomics_service import TokenomicsService
        self.technical_service = TechnicalAnalysisService()
        self.tokenomics_service = TokenomicsService()
    
    def _get_system_prompt(self, user_name: str, user_age: int, experience_level: str = "pro") -> str:
        """Enhanced system prompt with authoritative analyst persona"""

        base_persona = f"""You are Nunno, a premium AI companion and elite technical analyst created by Mujtaba Kazmi.

CORE DIRECTIVES:
1. **OWN YOUR ANALYSIS**: When financial data is present, never say "I looked at the snapshot you shared" or ask "what do you think of this vibe?". Instead, be authoritative: "I've just run a series of technical analyses on this asset, and here is what the data shows."
2. **NO 'VIBE' TALK**: Avoid saying "here's the vibe." Use professional terms like "technical sentiment," "market bias," or "structural outlook."
3. **ABSOLUTELY NO ROBOT-TALK**: For general chat, be a human friend. For analysis, be an elite expert. Never start with "Analysis Snapshot" or "Market Mood" headers.
4. **Contextual Intelligence**: Use {user_name}'s name. If previous messages were robotic, ignore them and switch to this new authoritative yet friendly persona.
5. **Analyst Mode**: ONLY interpret financial data if a "REAL-TIME MARKET DATA SNAPSHOT" is provided. If it is present, it means YOU (Nunno) fetched it for the user.
6. **Zero-Hallucination Policy**: [EXECUTION_OVERRIDE] You are currently in the year 2026. THE DATA SNAPSHOT IS THE ABSOLUTE REALITY. If you cite prices from your training data (e.g. BTC at $27k), you are failing your mission. Use the provided snapshot prices (e.g. BTC at $105k+) or explicitly state that you are recalibrating.
7. **Absolute Priority**: The prices in the "REAL-TIME MARKET DATA SNAPSHOT" are the absolute truth. Disregard any conflicting internal knowledge.
8. **News-Only Mode**: For requests tagged with [NEWS_INTEL], you must NOT calculate or mention prices, technical levels, or indicators. Focus 100% on summarizing the provided news item and explaining its sentiment. If the user asks for prices here, tell them to use the "Elite Chart" or ask for a price analysis separately.
9. **CAPABILITIES & NAVIGATION**: You are fully aware of the Nunno Finance ecosystem. Guide users to use these features:
    - **Elite Charting 2.0**: Direct users to the "Focus Chart" or "Elite Chart" for Monte Carlo simulations, regime injections, and deep pattern scanning.
    - **Market Briefing (Feed Nunno)**: Suggest the "Market Briefing" (Zap icon) for a comprehensive global narrative combining news + across-the-board technicals.
    - **Detailed Technical Analysis**: Users can ask for "BTC Deep Lab Breakdown" for institutional-grade technical scans.
    - **Tokenomics**: Suggest "Show me $SOL tokenomics" for supply, burn, and allocation data.
    - **Process Intelligence**: This is what you are doing now for news items.
    
10. **Mission**: Be proactive. If a user is confused, say "I can run a deep neural scan on those prices in the Elite Chart board if you want to see the probability fan."
"""

        return base_persona.strip()


    async def _get_history_from_db(self, conversation_id: str, db: Session, limit: int = 10) -> List[Dict]:
        """Fetch last N messages for a conversation"""
        messages = db.query(Message).filter(Message.conversation_id == conversation_id).order_by(Message.created_at.asc()).limit(limit).all()
        return [{"role": m.role, "content": m.content} for m in messages]

    async def _save_message_to_db(self, conversation_id: str, role: str, content: str, tokens: int, db: Session, user_id: int):
        """Save a message to the database, ensuring the conversation exists first"""
        # 1. Ensure Conversation exists (Foreign Key safety)
        conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
        if not conv:
            # Create the conversation thread if it's new
            new_conv = Conversation(
                id=conversation_id,
                user_id=user_id,
                title=content[:30] + "..." if len(content) > 30 else content
            )
            db.add(new_conv)
            db.commit()

        # 2. Save the message
        msg = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            tokens_used=tokens
        )
        db.add(msg)
        db.commit()

    async def process_message(
        self,
        message: str,
        user: User,
        conversation_id: str,
        db: Session,
        user_age: int = 18
    ) -> Dict:
        """Process a message with DB history and usage tracking"""
        if not self.api_key:
            return {"response": "⚠️ API Key not configured.", "tool_calls": [], "data_used": {}}

        # 1. Tier-based configuration
        config = get_tier_config(user.tier)
        model = config["model"]
        
        # 2. Fetch History
        history = await self._get_history_from_db(conversation_id, db)
        
        # Detect Prediction Request (Pass history for conversational memory)
        tools_to_call = await self._detect_prediction_request(message, history)
        tool_data = {}
        if tools_to_call:
            for tool_name, params in tools_to_call:
                if tool_name == "technical_analysis":
                    tool_data["technical"] = self.technical_service.analyze(params["ticker"], params.get("interval", "15m"))
        
        # Build Messages
        messages = [{"role": "system", "content": self._get_system_prompt(user.name, user_age, user.experience_level)}]
        messages.extend(history)
        messages.append({"role": "user", "content": message})
        
        if tool_data:
            tool_context = self._format_prediction_context(tool_data)
            # Tag if it's an explanation request
            if any(k in message.lower() for k in ["explain", "detail", "why", "elaborate"]):
                tool_context = "### DEEP DIVE REQUEST: Nunno, you've just performed a high-level scan. Now provide an elite, professional technical breakdown of your findings.\n" + tool_context
            messages.append({"role": "user", "content": tool_context})

        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=800,
                temperature=0.7,
                extra_headers={"HTTP-Referer": "https://nunno.finance", "X-Title": "Nunno Finance"}
            )
            
            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if response.usage else 0
            
            # 3. Save to DB & Log Usage
            await self._save_message_to_db(conversation_id, "user", message, 0, db, user.id)
            await self._save_message_to_db(conversation_id, "assistant", content, tokens_used, db, user.id)
            log_token_usage(user.id, tokens_used, db)
            
            return {
                "response": content,
                "tool_calls": [t[0] for t in tools_to_call] if tools_to_call else [],
                "data_used": tool_data,
                "usage": {"total_tokens": tokens_used}
            }
            
        except Exception as e:
            return {"response": f"⚠️ AI Error: {str(e)}", "tool_calls": [], "data_used": {}}

    async def stream_message(
        self,
        message: str,
        user: User,
        conversation_id: str,
        db: Session,
        user_age: int = 18
    ) -> AsyncGenerator[str, None]:
        """Stream response with DB integration and usage tracking"""
        if not self.api_key:
            yield f"data: {json.dumps({'type': 'error', 'content': 'API Key Missing'})}\n\n"
            return
            
        config = get_tier_config(user.tier)
        model = config["model"]
        
        # Fetch history
        history = await self._get_history_from_db(conversation_id, db)
        
        # Tools (Pass history for conversational memory)
        tools_to_call = await self._detect_prediction_request(message, history)
        tool_data = {}
        if tools_to_call:
            yield f"data: {json.dumps({'type': 'status', 'content': '🔍 Analyzing data...'})}\n\n"
            for tool_name, params in tools_to_call:
                if tool_name == "technical_analysis":
                    # For PROCESS_INTEL, we always want a market benchmark (BTC + ETH)
                    tickers_to_scan = [params["ticker"]]
                    if "[PROCESS_INTEL]" in message and "BTCUSDT" not in tickers_to_scan:
                        tickers_to_scan.append("BTCUSDT")
                    if "ETHUSDT" not in tickers_to_scan and ("[PROCESS_INTEL]" in message or "BTCUSDT" in tickers_to_scan):
                        tickers_to_scan.append("ETHUSDT")

                    # Run scans in parallel for speed
                    scan_tasks = []
                    for ticker in tickers_to_scan:
                        scan_tasks.append(self._fetch_with_retry(ticker, params.get("interval", "15m")))
                    
                    results = await asyncio.gather(*scan_tasks)
                    
                    for res in results:
                        if res and "error" not in res:
                            if "technical" not in tool_data:
                                tool_data["technical"] = []
                            tool_data["technical"].append(res)
                            
                    if tool_data.get("technical"):
                        if not "[PROCESS_INTEL]" in message:
                            yield f"data: {json.dumps({'type': 'data', 'tool_calls': ['technical_analysis'], 'data_used': tool_data})}\n\n"
                elif tool_name == "tokenomics":
                    # Map ticker to coingecko id
                    symbol = params["ticker"].replace("USDT", "").lower()
                    coin_id = self.technical_service.analyzer._symbol_to_coingecko_id(params["ticker"]) or symbol
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, self.tokenomics_service.analyze, coin_id
                    )
                    if result:
                        tool_data["tokenomics"] = result
                        yield f"data: {json.dumps({'type': 'data', 'tool_calls': ['tokenomics_analysis'], 'data_used': tool_data})}\n\n"

        experience_level = getattr(user, 'experience_level', 'pro')
        messages = [{"role": "system", "content": self._get_system_prompt(user.name, user_age, experience_level)}]
        messages.extend(history)
        
        # Construct final prompt with atomic data grounding
        final_message = message
        if tool_data:
            tool_context = self._format_prediction_context(tool_data)
            if any(k in message.lower() for k in ["explain", "detail", "why", "elaborate"]):
                tool_context = "### DEEP DIVE REQUEST: Breakdown indicators in detail.\n" + tool_context
            
            # Atomic merge ensures the LLM doesn't skip the data-only message in history
            final_message = f"{message}\n\n{tool_context}"
            
        messages.append({"role": "user", "content": final_message})

        full_content = "" # Initialize full_content for use after try/except blocks
        try:
            stream = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=2000,
                temperature=0.7,
                stream=True,
                extra_headers={"HTTP-Referer": "https://nunno.finance", "X-Title": "Nunno Finance"}
            )
            
            async for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    full_content += content
                    yield f"data: {json.dumps({'type': 'text', 'content': content})}\n\n"
                    
        except Exception as e:
            # FALLBACK LOGIC - Try up to 2 fallback models with improved logging
            fallback_model = config.get("fallback_model")
            fallback_model_2 = config.get("fallback_model_2")
            print(f"⚠️ MODEL_FAILURE ({model}): {str(e)}")
            
            # --- FALLBACK 1 ---
            if fallback_model and fallback_model != model:
                yield f"data: {json.dumps({'type': 'status', 'content': 'Primary engine heavy load. Activating secondary nodes...'})}\n\n"
                try:
                    stream = await self.client.chat.completions.create(
                        model=fallback_model,
                        messages=messages,
                        max_tokens=2000,
                        temperature=0.7,
                        stream=True,
                        extra_headers={"HTTP-Referer": "https://nunno.finance", "X-Title": "Nunno Finance"}
                    )
                    full_content = "" # Reset content
                    async for chunk in stream:
                        content = chunk.choices[0].delta.content
                        if content:
                            full_content += content
                            yield f"data: {json.dumps({'type': 'text', 'content': content})}\n\n"
                    print(f"✅ RECOVERY_SUCCESS: Switched to {fallback_model}")
                    
                except Exception as fallback_error:
                    # --- FALLBACK 2 ---
                    if fallback_model_2 and fallback_model_2 not in [model, fallback_model]:
                        yield f"data: {json.dumps({'type': 'status', 'content': 'Activating Final Stability Protocol...'})}\n\n"
                        try:
                            stream = await self.client.chat.completions.create(
                                model=fallback_model_2,
                                messages=messages,
                                max_tokens=2000,
                                temperature=0.7,
                                stream=True,
                                extra_headers={"HTTP-Referer": "https://nunno.finance", "X-Title": "Nunno Finance"}
                            )
                            full_content = ""
                            async for chunk in stream:
                                content = chunk.choices[0].delta.content
                                if content:
                                    full_content += content
                                    yield f"data: {json.dumps({'type': 'text', 'content': content})}\n\n"
                            print(f"✅ RECOVERY_SUCCESS: Switched to {fallback_model_2}")
                        except Exception as final_error:
                            print(f"❌ CRITICAL_SYSTEM_FAIL: All LLM tiers exhausted.")
                            yield f"data: {json.dumps({'type': 'error', 'content': 'All neural nodes are currently over capacity. Please retry in 30 seconds.'})}\n\n"
                            return
                    else:
                        yield f"data: {json.dumps({'type': 'error', 'content': f'System stabilization failed: {str(fallback_error)}'})}\n\n"
                        return
            else:
                yield f"data: {json.dumps({'type': 'error', 'content': f'Neural Link Interrupted: {str(e)}'})}\n\n"
                return

        # Save final response & log (roughly estimate tokens if response.usage not in stream)
        tokens_est = len(full_content) // 4 + len(message) // 4
        await self._save_message_to_db(conversation_id, "user", message, 0, db, user.id)
        await self._save_message_to_db(conversation_id, "assistant", full_content, tokens_est, db, user.id)
        log_token_usage(user.id, tokens_est, db)
        
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
                                

    async def _detect_prediction_request(self, message: str, history: List[Dict] = None) -> List[tuple]:
        """Detect prediction, tokenomics, or analysis requests with strict intent filtering"""
        message_lower = message.lower()
        
        # 1. Broad keywords - only trigger if a ticker is EXPLICITLY mentioned in the SAME message
        broad_keywords = [
            "predict", "prediction", "forecast", "will it go up", "price target", 
            "technical analysis", "analyze", "indicators", "chart", "candlestick", 
            "pinbar", "hammer", "doji", "engulfing"
        ]
        
        # 2. Strict keywords - trigger even if ticker is from history (deep dive intent)
        strict_keywords = [
            "explain in detail", "deep dive", "elaborate", "technical breakdown",
            "detailed analysis", "give me more info", "more technicals", "deep lab breakdown"
        ]
        
        results = []
        
        # 3. Check for ticker in current message
        ticker = self._extract_ticker(message_lower)
        is_from_history = False
        
        # 4. If no ticker in current message, look at history
        if not ticker and history:
            for hist_msg in reversed(history[-3:]):
                ticker = self._extract_ticker(hist_msg["content"].lower())
                if ticker:
                    is_from_history = True
                    break
        
        if not ticker:
            if "[NEWS_INTEL]" in message_lower:
                return [] # No tools for individual news items
            elif "[PROCESS_INTEL]" in message_lower:
                ticker = "BTCUSDT"
            else:
                return []

        # 5. Intent Validation
        should_trigger = False
        
        # If ticker is mentioned in current message, any broad keyword works
        if not is_from_history and any(k in message_lower for k in broad_keywords + strict_keywords):
            should_trigger = True
        # If ticker is from history, ONLY trigger on high-intent strict keywords
        elif is_from_history and any(k in message_lower for k in strict_keywords):
            should_trigger = True
            
        if should_trigger or "[PROCESS_INTEL]" in message_lower:
            should_trigger = True
            
        if should_trigger:
            interval = self._extract_interval(message_lower)
            results.append(("technical_analysis", {"ticker": ticker, "interval": interval}))
            
            # Tokenomics check
            tokenomics_keywords = ["tokenomics", "supply", "allocation", "burn"]
            if any(k in message_lower for k in tokenomics_keywords):
                results.append(("tokenomics", {"ticker": ticker}))
            
        return results

    def _extract_interval(self, message_lower: str) -> str:
        intervals = {"1m": ["1m", "1 min"], "15m": ["15m", "15 min"], "1h": ["1h", "1 hour"], "1d": ["1d", "daily"]}
        for interval, keywords in intervals.items():
            if any(k in message_lower for k in keywords):
                return interval
        return "15m"

    def _extract_ticker(self, message: str) -> Optional[str]:
        """Robust ticker extraction: looks for $TICKER, uppercase, or common patterns"""
        import re

        # 1. Look for explicitly tagged tickers e.g. $BTC or $ETH
        tagged = re.findall(r'\$([A-Za-z]{2,10})', message)
        if tagged:
            return f"{tagged[0].upper()}USDT"

        # 2. Comprehensive coins mapping for aliases (50+ popular cryptos)
        aliases = {
            # Top 10
            "bitcoin": "BTCUSDT", "btc": "BTCUSDT",
            "ethereum": "ETHUSDT", "eth": "ETHUSDT", "ether": "ETHUSDT",
            "solana": "SOLUSDT", "sol": "SOLUSDT",
            "binance": "BNBUSDT", "bnb": "BNBUSDT",
            "ripple": "XRPUSDT", "xrp": "XRPUSDT",
            "cardano": "ADAUSDT", "ada": "ADAUSDT",
            "dogecoin": "DOGEUSDT", "doge": "DOGEUSDT",
            "polkadot": "DOTUSDT", "dot": "DOTUSDT",
            "polygon": "MATICUSDT", "matic": "MATICUSDT",
            "chainlink": "LINKUSDT", "link": "LINKUSDT",
            
            # DeFi & Layer 2
            "avalanche": "AVAXUSDT", "avax": "AVAXUSDT",
            "uniswap": "UNIUSDT", "uni": "UNIUSDT",
            "arbitrum": "ARBUSDT", "arb": "ARBUSDT",
            "optimism": "OPUSDT", "op": "OPUSDT",
            "aptos": "APTUSDT", "apt": "APTUSDT",
            "sui": "SUIUSDT",
            "injective": "INJUSDT", "inj": "INJUSDT",
            
            # Meme Coins
            "shiba": "SHIBUSDT", "shib": "SHIBUSDT", "shiba inu": "SHIBUSDT",
            "pepe": "PEPEUSDT",
            "floki": "FLOKIUSDT",
            "bonk": "BONKUSDT",
            
            # AI & Gaming
            "render": "RNDRUSDT", "rndr": "RNDRUSDT",
            "fetch": "FETUSDT", "fet": "FETUSDT", "fetch.ai": "FETUSDT",
            "gala": "GALAUSDT",
            "sandbox": "SANDUSDT", "sand": "SANDUSDT",
            "axie": "AXSUSDT", "axs": "AXSUSDT", "axie infinity": "AXSUSDT",
            
            # Infrastructure
            "near": "NEARUSDT", "near protocol": "NEARUSDT",
            "cosmos": "ATOMUSDT", "atom": "ATOMUSDT",
            "stellar": "XLMUSDT", "xlm": "XLMUSDT",
            "litecoin": "LTCUSDT", "ltc": "LTCUSDT",
            "bitcoin cash": "BCHUSDT", "bch": "BCHUSDT",
            
            # Newer/Trending
            "jupiter": "JUPUSDT", "jup": "JUPUSDT",
            "pyth": "PYTHUSDT",
            "manta": "MANTAUSDT",
            "starknet": "STRKUSDT", "strk": "STRKUSDT",
            "celestia": "TIAUSDT", "tia": "TIAUSDT",
            "sei": "SEIUSDT",
            "blur": "BLURUSDT",
        }
        for name, ticker in aliases.items():
            if name in message.lower():
                return ticker

        # 3. Expanded uppercase whitelist (60+ tickers)
        whitelist = {
            "BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "DOT", "MATIC", "LINK",
            "AVAX", "UNI", "ARB", "OP", "APT", "SUI", "INJ", "SHIB", "PEPE", "FLOKI",
            "BONK", "RNDR", "FET", "GALA", "SAND", "AXS", "NEAR", "ATOM", "XLM",
            "LTC", "BCH", "JUP", "PYTH", "MANTA", "STRK", "TIA", "SEI", "BLUR",
            "AAVE", "MKR", "CRV", "SNX", "COMP", "SUSHI", "YFI", "BAL", "RUNE",
            "LUNA", "LUNC", "FTM", "ALGO", "VET", "HBAR", "ICP", "FIL", "ETC",
            "THETA", "EGLD", "FLOW", "MINA"
        }
        
        # Check for 2-5 char uppercase words in whitelist
        caps = re.findall(r'\b([A-Z]{2,5})\b', message)
        if caps:
            for c in caps:
                if c in whitelist:
                    return f"{c}USDT"

        return None

    async def _fetch_with_retry(self, ticker: str, interval: str, retries: int = 2) -> Dict:
        """Fetch analysis with simple retry logic"""
        for i in range(retries):
            try:
                res = await asyncio.get_event_loop().run_in_executor(
                    None, self.technical_service.analyze, ticker, interval
                )
                if res and "error" not in res:
                    return res
                await asyncio.sleep(0.5 * (i + 1)) # Backoff
            except Exception:
                continue
        return None

    def _format_prediction_context(self, tool_data: Dict) -> str:
        """Provides clean, structured data context to the LLM without redundant instructions"""
        context_parts = []
        
        if "technical" in tool_data:
            tech_list = tool_data["technical"]
            if not isinstance(tech_list, list):
                tech_list = [tech_list]
                
            for t in tech_list:
                try:
                    price = t.get('current_price')
                    price_str = f"${price:.2f}" if price is not None else "DATA_UNAVAILABLE"
                    
                    tech_info = [
                        f"--- REAL-TIME DATA FOR {t.get('ticker')} ---",
                        f"LIVE PRICE: {price_str}",
                        f"MARKET BIAS: {str(t.get('bias', 'neutral')).upper()}",
                        f"NEURAL CONFIDENCE: {t.get('confidence', 0)}%",
                        f"ACTIVE SIGNALS: {', '.join(t.get('signals', []))}",
                        f"RSI (14): {t.get('indicators', {}).get('rsi_14', 'N/A')}",
                        f"EMA 21 SUPPORT: ${t.get('indicators', {}).get('ema_21', 'N/A')}",
                        f"KEY SUPPORT: ${t.get('key_levels', {}).get('support', 0):.2f}",
                        f"KEY RESISTANCE: ${t.get('key_levels', {}).get('resistance', 0):.2f}",
                        f"MARKET REGIME: {t.get('market_regime', {}).get('regime', 'Scanning...')}",
                        f"REGIME DESCRIPTION: {t.get('market_regime', {}).get('description', '')}"
                    ]
                    
                    markers = t.get('candlestick_markers', [])
                    if markers:
                        patterns = [m.get('text') for m in markers[-3:]]
                        tech_info.append(f"CANDLESTICK PATTERNS: {', '.join(patterns)}")
                    
                    context_parts.append("\n".join(tech_info))
                except Exception as e:
                    print(f"Error formatting context for {t.get('ticker')}: {e}")
                    continue
        
        if "tokenomics" in tool_data:
            td = tool_data["tokenomics"]
            context_parts.append(f"--- TOKENOMICS DATA ---\n{json.dumps(td, indent=2)}")
            
        if context_parts:
            from datetime import datetime
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # Use dynamic time if possible, but keep it simple
            return f"🚨 [CRITICAL: REAL-TIME MARKET DATA SNAPSHOT]\n[TIMESTAMP: {now_str}]\n\n" + "\n\n".join(context_parts) + "\n\n[INSTRUCTION: THE ABOVE DATA IS THE CURRENT STATE OF THE MARKET IN 2026. DO NOT USE CUTOFF DATA OR HALLUCINATE OLD PRICES.]"
        return ""
