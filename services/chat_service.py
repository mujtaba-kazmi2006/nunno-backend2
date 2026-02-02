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
        self.technical_service = TechnicalAnalysisService()
    
    def _get_system_prompt(self, user_name: str, user_age: int) -> str:
        """Enhanced system prompt"""
        return f"""You are Nunno, a friendly AI financial educator by Mujtaba Kazmi.

User: {user_name}, {user_age} years old

For PREDICTIONS with technical data:
1. **Price Summary**: Start with current price and direction.
2. **Technical Scorecard (Table)**: Markdown table with Indicator | Value | Meaning.
3. **Indicator Deep Dive**: Explain 2-3 key indicators using simple analogies.
4. **Levels & Strategy**: Support/Resistance levels and their meaning.
5. **Final Verdict**: Clear, encouraging conclusion.

Keep it structured, use bold headers, and emojis! 📈💡

For OTHER questions:
- Explain concepts simply with real examples
- Be concise (2-3 paragraphs)
- Never give financial advice - educate only!"""

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
        messages = [{"role": "system", "content": self._get_system_prompt(user.name, user_age)}]
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
            for _, params in tools_to_call:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, self.technical_service.analyze, params["ticker"], params.get("interval", "15m")
                )
                if result:
                    tool_data["technical"] = result
                    yield f"data: {json.dumps({'type': 'data', 'tool_calls': ['technical_analysis'], 'data_used': tool_data})}\n\n"

        messages = [{"role": "system", "content": self._get_system_prompt(user.name, user_age)}]
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
        """Detect ONLY prediction requests"""
        message_lower = message.lower()
        prediction_keywords = ["predict", "prediction", "forecast", "will it go up", "price target", "technical analysis"]
        
        if any(keyword in message_lower for keyword in prediction_keywords):
            ticker = self._extract_ticker(message_lower)
            if ticker:
                interval = self._extract_interval(message_lower)
                return [("technical_analysis", {"ticker": ticker, "interval": interval})]
        return []

    def _extract_interval(self, message_lower: str) -> str:
        intervals = {"1m": ["1m", "1 min"], "15m": ["15m", "15 min"], "1h": ["1h", "1 hour"], "1d": ["1d", "daily"]}
        for interval, keywords in intervals.items():
            if any(k in message_lower for k in keywords):
                return interval
        return "15m"

    def _extract_ticker(self, message_lower: str) -> Optional[str]:
        tickers = {"bitcoin": "BTCUSDT", "btc": "BTCUSDT", "eth": "ETHUSDT", "sol": "SOLUSDT"}
        for name, ticker in tickers.items():
            if name in message_lower:
                return ticker
        return None

    def _format_prediction_context(self, tool_data: Dict) -> str:
        if "technical" in tool_data:
            t = tool_data["technical"]
            return f"Asset: {t.get('ticker')}\nPrice: ${t.get('current_price'):.2f}\nBias: {t.get('bias')}\nConfidence: {t.get('confidence')}%\n\nAnalyze this for the user."
        return ""
