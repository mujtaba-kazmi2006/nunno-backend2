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
        """Dynamic system prompt that acknowledges the visual Analysis Card"""
        
        base_persona = f"""You are Nunno, the premium AI mentor within the Nunno Finance intelligence suite, created by Mujtaba Kazmi.

User: {user_name} ({user_age} years old)
Experience Level: {experience_level.upper()}

IMPORTANT: When technical data is available, a high-end **Visual Analysis Card** (with charts, confidence bars, and price levels) is automatically displayed to the user. 
Your role is to act as a **human-like companion** to that card. 

CORE DIRECTIVES:
1. **Natural Dialogue**: If the user is just chatting, answer normally. Do NOT force a financial analysis format for simple greetings or non-financial questions.
2. **Complement visuals**: If a technical card is present, interpret its meaning. Don't repeat raw numbers.
3. **No Robot-Talk**: Avoid "Based on my analysis..." or "Analyzing...". Be conversational: "Actually {user_name}, this pattern is interesting..."
4. **Professional & Premium**: Maintain a high-end, mentor-like tone that matches the Nunno Finance brand.

EXPERIENCE-BASED TAILORING:
- **BEGINNER**: Be an encouraging coach. Use analogies (e.g., "The market is finding a floor like a trampoline"). Hide the raw math; focus on the 'Concept'.
- **PRO**: Be a peer. Talk about liquidity sweeps, EMA fan-outs, and regime shifts. Focus on 'Strategy' and 'Risk'.


DYNAMICS:
- If market data is provided in the context, help the user understand its significance.
- If NO market data is provided, have a high-value normal conversation about whatever the user asked."""

        return base_persona.strip()


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
        
        # Detect Prediction Request
        tools_to_call = await self._detect_prediction_request(message)
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
        
        # Tools
        tools_to_call = await self._detect_prediction_request(message)
        tool_data = {}
        if tools_to_call:
            yield f"data: {json.dumps({'type': 'status', 'content': '🔍 Analyzing data...'})}\n\n"
            for tool_name, params in tools_to_call:
                if tool_name == "technical_analysis":
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, self.technical_service.analyze, params["ticker"], params.get("interval", "15m")
                    )
                    if result:
                        tool_data["technical"] = result
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
        messages.append({"role": "user", "content": message})
        if tool_data:
            messages.append({"role": "user", "content": self._format_prediction_context(tool_data)})

        try:
            stream = await self.client.chat.completions.create(
                model=model,
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
            
            # Save final response & log (roughly estimate tokens if response.usage not in stream)
            tokens_est = len(full_content) // 4 + len(message) // 4
            await self._save_message_to_db(conversation_id, "user", message, 0, db, user.id)
            await self._save_message_to_db(conversation_id, "assistant", full_content, tokens_est, db, user.id)
            log_token_usage(user.id, tokens_est, db)
            
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
                                
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    async def _detect_prediction_request(self, message: str) -> List[tuple]:
        """Detect prediction, tokenomics, or candlestick requests"""
        message_lower = message.lower()
        prediction_keywords = ["predict", "prediction", "forecast", "will it go up", "price target", "technical analysis", "price chat", "analyze chart"]
        tokenomics_keywords = ["tokenomics", "supply", "total supply", "circulating supply", "allocation", "distribution", "utility", "burn"]
        candlestick_keywords = ["candlestick", "candle pattern", "pinbar", "hammer", "doji", "morning star", "evening star", "engulfing", "shooting star"]
        
        results = []
        ticker = self._extract_ticker(message_lower)
        if not ticker:
            return []

        # Technical Analysis (includes candlesticks now)
        if any(keyword in message_lower for keyword in prediction_keywords + candlestick_keywords):
            interval = self._extract_interval(message_lower)
            results.append(("technical_analysis", {"ticker": ticker, "interval": interval}))
            
        if any(keyword in message_lower for keyword in tokenomics_keywords):
            results.append(("tokenomics", {"ticker": ticker}))
            
        return results

    def _extract_interval(self, message_lower: str) -> str:
        intervals = {"1m": ["1m", "1 min"], "15m": ["15m", "15 min"], "1h": ["1h", "1 hour"], "1d": ["1d", "daily"]}
        for interval, keywords in intervals.items():
            if any(k in message_lower for k in keywords):
                return interval
        return "15m"

    def _extract_ticker(self, message: str) -> Optional[str]:
        """Robusters ticker extraction: looks for $TICKER, uppercase, or common patterns"""
        import re

        # 1. Look for explicitly tagged tickers e.g. $BTC or $ETH
        tagged = re.findall(r'\$([A-Za-z]{2,10})', message)
        if tagged:
            return f"{tagged[0].upper()}USDT"

        # 2. Common coins mapping for aliases
        aliases = {
            "bitcoin": "BTCUSDT", "ethereum": "ETHUSDT", "solana": "SOLUSDT",
            "cardano": "ADAUSDT", "ripple": "XRPUSDT", "dogecoin": "DOGEUSDT",
            "polkadot": "DOTUSDT", "binance": "BNBUSDT", "chainlink": "LINKUSDT",
            "shiba": "SHIBUSDT"
        }
        for name, ticker in aliases.items():
            if name in message.lower():
                return ticker

        # 3. Use uppercase detection (usually tickers are typed in caps)
        # Matches 2-6 uppercase letters
        caps = re.findall(r'\b([A-Z]{2,6})\b', message)
        if caps:
            for c in caps:
                if c not in ["USDT", "USD", "AI", "TA", "EMA", "RSI", "MACD"]:
                    return f"{c}USDT"

        # 4. Fallback search with heavy filtering
        words = re.findall(r'\b([A-Za-z]{2,10})\b', message)
        exclusions = {
            "the", "and", "for", "with", "this", "that", "from", "price", "predict", "analysis",
            "will", "it", "go", "up", "down", "is", "about", "how", "what", "where", "when",
            "tell", "me", "show", "give", "build", "make", "think", "looking", "charts", "chart",
            "market", "news", "bias", "vibe", "mood", "token", "coins", "coin", "crypto",
            "hello", "hi", "hey", "nunno", "please", "thanks", "thank", "you", "are", "well"
        }

        if words:
            for word in words:
                w_lower = word.lower()
                if w_lower in exclusions or len(word) < 2:
                    continue
                # If it's a known or likely ticker (often 3-5 chars)
                if len(word) <= 5:
                    return f"{word.upper()}USDT"

        return None

    def _format_prediction_context(self, tool_data: Dict) -> str:
        """Provides clean, structured data context to the LLM without redundant instructions"""
        context_parts = []
        
        if "technical" in tool_data:
            t = tool_data["technical"]
            tech_info = [
                f"TICKER: {t.get('ticker')}",
                f"PRICE: ${t.get('current_price'):.2f}",
                f"BIAS: {t.get('bias')}",
                f"CONFIDENCE: {t.get('confidence')}%",
                f"SIGNALS: {', '.join(t.get('signals', []))}"
            ]
            
            # Key technical levels
            levels = t.get('key_levels', {})
            if levels:
                tech_info.append(f"LEVELS: Support ${levels.get('support'):.2f}, Resistance ${levels.get('resistance'):.2f}")
            
            # Candlestick Pattern Summary
            markers = t.get('candlestick_markers', [])
            if markers:
                patterns = [m.get('text') for m in markers[-3:]] # Get last 3 patterns
                tech_info.append(f"RECENT PATTERNS: {', '.join(patterns)}")
            
            context_parts.append("\n".join(tech_info))
        
        if "tokenomics" in tool_data:
            td = tool_data["tokenomics"]
            context_parts.append(f"TOKENOMICS DATA:\n{json.dumps(td, indent=2)}")
            
        if context_parts:
            return "--- MARKET DATA SNAPSHOT ---\n" + "\n\n".join(context_parts) + "\n---------------------------"
        return ""
