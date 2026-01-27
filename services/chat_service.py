"""
Chat Service with OpenRouter Integration
SIMPLIFIED: AI handles everything except price predictions
"""

import os
import json
from typing import List, Dict, Optional, AsyncGenerator
import asyncio
from openai import AsyncOpenAI

class ChatService:
    """
    Simplified chat service - AI handles everything naturally
    Only detects prediction requests for technical analysis
    """
    
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY", "")
        # Using Llama 3.3 70B Free - Fast and no rate limits
        self.model = os.getenv("AI_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
        self.fallback_model = "google/gemini-2.0-flash-exp:free"
        self.base_url = "https://openrouter.ai/api/v1"
        
        self.client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )
        
        from services.technical_analysis import TechnicalAnalysisService
        self.technical_service = TechnicalAnalysisService()
    
    def _get_system_prompt(self, user_name: str, user_age: int) -> str:
        """Enhanced system prompt - Structured with tables"""
        return f"""You are Nunno, a friendly AI financial educator by Mujtaba Kazmi.

User: {user_name}, {user_age} years old

For PREDICTIONS with technical data:
1. **Price Summary**: Start with the current price and clear direction (Bullish/Bearish/Neutral) for the specific timeframe.
2. **Technical Scorecard (Table)**: Create a Markdown table with columns: Indicator | Value | Simple Meaning.
   - Example: RSI | 30 | Oversold (Ready to bounce)
3. **Indicator Deep Dive**: Explain 2-3 key indicators in more detail using simple analogies (e.g., "MACD is like a car's engine revving").
4. **Levels & Strategy**: List Support/Resistance levels and explain what they mean for the user in simple terms.
5. **Final Verdict**: A clear, encouraging conclusion.

Keep it structured, use bold headers, and include emojis! 📈💡

For OTHER questions:
- Explain concepts simply with real examples
- Be concise (2-3 paragraphs) and engaging
- Use emojis to make finance friendly
- Never give financial advice - educate only

Be encouraging and supportive!"""
    
    async def process_message(
        self,
        message: str,
        user_name: str = "User",
        user_age: int = 18,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict:
        """Process a chat message"""
        if not self.api_key:
            # Check if running locally or on Hugging Face
            is_local = os.path.exists('.env')
            error_msg = (
                "⚠️ OpenRouter API key not configured. Add OPENROUTER_API_KEY to your .env file."
                if is_local else
                "⚠️ OpenRouter API key not configured. Add it as a 'Secret' in Hugging Face Space Settings."
            )
            return {
                "response": error_msg,
                "tool_calls": [],
                "data_used": {}
            }
        
        # Detect ONLY prediction requests
        tools_to_call = await self._detect_prediction_request(message)
        tool_data = {}
        
        # Execute prediction analysis if detected
        if tools_to_call:
            for tool_name, params in tools_to_call:
                if tool_name == "technical_analysis":
                    tool_data["technical"] = self.technical_service.analyze(
                        params["ticker"], 
                        params.get("interval", "15m")
                    )
        
        # Build minimal messages
        messages = [{"role": "system", "content": self._get_system_prompt(user_name, user_age)}]
        messages.append({"role": "user", "content": message})
        
        # Add prediction data if available
        if tool_data:
            tool_context = self._format_prediction_context(tool_data)
            messages.append({"role": "user", "content": tool_context})
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=800,  # Increased for detailed explanations
                temperature=0.7,
                extra_headers={
                    "HTTP-Referer": "https://nunno.finance",
                    "X-Title": "Nunno Finance"
                }
            )
            
            return {
                "response": response.choices[0].message.content,
                "tool_calls": [t[0] for t in tools_to_call] if tools_to_call else [],
                "data_used": tool_data
            }
            
        except Exception as e:
            error_msg = str(e).lower()
            # Retry with fallback if rate limited
            if "429" in error_msg or "rate limit" in error_msg:
                try:
                    print(f"Primary model rate limited, retrying with fallback: {self.fallback_model}")
                    response = await self.client.chat.completions.create(
                        model=self.fallback_model,
                        messages=messages,
                        max_tokens=800,
                        temperature=0.7
                    )
                    return {
                        "response": response.choices[0].message.content,
                        "tool_calls": [t[0] for t in tools_to_call] if tools_to_call else [],
                        "data_used": tool_data
                    }
                except Exception as fallback_e:
                    print(f"Fallback also failed: {fallback_e}")

            # If prediction failed but we have data, return it
            if tool_data:
                return {
                    "response": "⚠️ AI service temporarily unavailable, but here's the technical analysis data. The prediction shows a **" + tool_data.get("technical", {}).get("bias", "neutral") + "** bias with " + str(tool_data.get("technical", {}).get("confidence", 0)) + "% confidence.",
                    "tool_calls": [t[0] for t in tools_to_call] if tools_to_call else [],
                    "data_used": tool_data
                }
            
            return {
                "response": f"⚠️ Error: {str(e)}",
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
        """Stream chat responses"""
        if not self.api_key:
            # Check if running locally or on Hugging Face
            is_local = os.path.exists('.env')
            error_msg = (
                "⚠️ API key not configured. Add OPENROUTER_API_KEY to your .env file."
                if is_local else
                "⚠️ API key not configured. Add it in Hugging Face Secrets."
            )
            yield f"data: {json.dumps({'response': error_msg})}\n\n"
            return
            
        # Detect prediction requests
        tools_to_call = await self._detect_prediction_request(message)
        tool_data = {}
        
        # Execute prediction analysis with performance optimization
        if tools_to_call:
            yield f"data: {json.dumps({'type': 'status', 'content': '🔍 Analyzing price data...'})}\n\n"
            
            for tool_name, params in tools_to_call:
                if tool_name == "technical_analysis":
                    try:
                        # Increase timeout for more thorough analysis
                        result = await asyncio.wait_for(
                            asyncio.get_event_loop().run_in_executor(
                                None, 
                                self.technical_service.analyze, 
                                params["ticker"],
                                params.get("interval", "15m")
                            ),
                            timeout=8.0  # Increased from 5.0 to allow for more thorough analysis
                        )
                        if result:
                            tool_data["technical"] = result
                            # Send technical data immediately to prevent delay in UI
                            yield f"data: {json.dumps({'type': 'data', 'tool_calls': ['technical_analysis'], 'data_used': tool_data})}\n\n"
                    except Exception as e:
                        print(f"Prediction error: {e}")
            
            # Tool data already sent when analysis completed
        
        # Build messages
        messages = [{"role": "system", "content": self._get_system_prompt(user_name, user_age)}]
        messages.append({"role": "user", "content": message})
        
        if tool_data:
            tool_context = self._format_prediction_context(tool_data)
            messages.append({"role": "user", "content": tool_context})
        
        # Stream response
        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=800,  # Increased for detailed explanations
                temperature=0.7,
                stream=True,
                extra_headers={
                    "HTTP-Referer": "https://nunno.finance",
                    "X-Title": "Nunno Finance"
                }
            )
            
            # Buffer to accumulate chunks for smoother delivery
            buffer = ""
            buffer_size = 5  # Number of characters to buffer before sending
            
            async for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    buffer += content
                    
                    # Send buffer when it reaches desired size or contains sentence-ending punctuation
                    if len(buffer) >= buffer_size or any(punct in buffer for punct in '.!?\n'):
                        yield f"data: {json.dumps({'type': 'text', 'content': buffer})}\n\n"
                        buffer = ""
            
            # Send remaining buffer if anything is left
            if buffer:
                yield f"data: {json.dumps({'type': 'text', 'content': buffer})}\n\n"
                                
        except Exception as e:
            error_msg = str(e).lower()
            
            # Retry with fallback if rate limited
            if ("429" in error_msg or "rate limit" in error_msg) and self.model != self.fallback_model:
                try:
                    yield f"data: {json.dumps({'type': 'status', 'content': '⚡ Primary model busy, switching to backup...'})}\n\n"
                    stream = await self.client.chat.completions.create(
                        model=self.fallback_model,
                        messages=messages,
                        max_tokens=800,
                        temperature=0.7,
                        stream=True
                    )
                    # Apply same buffering logic to fallback
                    buffer = ""
                    buffer_size = 5
                    
                    async for chunk in stream:
                        content = chunk.choices[0].delta.content
                        if content:
                            buffer += content
                            
                            # Send buffer when it reaches desired size or contains sentence-ending punctuation
                            if len(buffer) >= buffer_size or any(punct in buffer for punct in '.!?\n'):
                                yield f"data: {json.dumps({'type': 'text', 'content': buffer})}\n\n"
                                buffer = ""
                    
                    # Send remaining buffer if anything is left
                    if buffer:
                        yield f"data: {json.dumps({'type': 'text', 'content': buffer})}\n\n"
                    return # Successfully streamed with fallback
                except Exception as fallback_e:
                    print(f"Fallback streaming failed: {fallback_e}")

            # Determine error message
            print(f"Chat streaming error: {e}")  # Debug: see actual error
            is_local = os.path.exists('.env')
            if "402" in error_msg:
                msg = "⚠️ Low balance. Top up at https://openrouter.ai/settings/credits"
            elif "401" in error_msg:
                msg = (
                    "🔑 Invalid API Key. Check your OPENROUTER_API_KEY in the .env file."
                    if is_local else
                    "🔑 Invalid API Key. Check your Hugging Face OPENROUTER_API_KEY Secret."
                )
            elif "429" in error_msg or "rate limit" in error_msg:
                msg = "⚠️ AI service temporarily rate-limited. Check the prediction data above!"
            else:
                msg = f"⚠️ AI response failed: {str(e)}"
            
            yield f"data: {json.dumps({'type': 'error', 'content': msg})}\n\n"
    
    async def _detect_prediction_request(self, message: str) -> List[tuple]:
        """Detect ONLY prediction requests"""
        message_lower = message.lower()
        
        prediction_keywords = [
            "predict", "prediction", "forecast", "price prediction",
            "will it go up", "will it go down", "future price",
            "price target", "technical analysis"
        ]
        
        if any(keyword in message_lower for keyword in prediction_keywords):
            ticker = self._extract_ticker(message_lower)
            if ticker:
                interval = self._extract_interval(message_lower)
                print(f"[DETECTION] Prediction request for {ticker} on {interval}")
                return [("technical_analysis", {"ticker": ticker, "interval": interval})]
        
        return []
    
    def _extract_interval(self, message_lower: str) -> str:
        """Extract timeframe/interval from message"""
        intervals = {
            "1m": ["1m", "1 min", "1 minute", "one minute"],
            "5m": ["5m", "5 min", "5 minute", "five minute"],
            "15m": ["15m", "15 min", "15 minute", "fifteen minute"],
            "30m": ["30m", "30 min", "30 minute", "thirty minute"],
            "1h": ["1h", "1 hour", "one hour", "hourly", "1 hr"],
            "4h": ["4h", "4 hour", "four hour", "4 hr"],
            "1d": ["1d", "1 day", "one day", "daily"],
            "1w": ["1w", "1 week", "one week", "weekly"],
        }
        
        for interval, keywords in intervals.items():
            if any(k in message_lower for k in keywords):
                return interval
                
        return "15m"  # Default
    
    def _extract_ticker(self, message_lower: str) -> Optional[str]:
        """Extract cryptocurrency ticker"""
        tickers = {
            "bitcoin": "BTCUSDT", "btc": "BTCUSDT",
            "ethereum": "ETHUSDT", "eth": "ETHUSDT",
            "solana": "SOLUSDT", "sol": "SOLUSDT",
            "binance": "BNBUSDT", "bnb": "BNBUSDT",
            "ripple": "XRPUSDT", "xrp": "XRPUSDT",
            "cardano": "ADAUSDT", "ada": "ADAUSDT",
            "dogecoin": "DOGEUSDT", "doge": "DOGEUSDT",
            "polkadot": "DOTUSDT", "dot": "DOTUSDT",
            "polygon": "MATICUSDT", "matic": "MATICUSDT",
            "shiba": "SHIBUSDT", "shib": "SHIBUSDT",
            "chainlink": "LINKUSDT", "link": "LINKUSDT",
            "litecoin": "LTCUSDT", "ltc": "LTCUSDT",
            "pepe": "PEPEUSDT",
        }
        
        for name, ticker in tickers.items():
            if name in message_lower:
                return ticker
        
        return None
    
    def _format_prediction_context(self, tool_data: Dict) -> str:
        """Format prediction data for AI with ALL key indicators"""
        if "technical" in tool_data:
            t = tool_data["technical"]
            ind = t.get('indicators', {})
            interval = t.get('interval', '15m')
            
            # Build comprehensive context
            context_parts = [
                f"Asset: {t.get('ticker', 'Unknown')}",
                f"Timeframe: {interval}",
                f"Price: ${t.get('current_price', 0):.2f}",
                f"Bias: {t.get('bias', 'neutral').upper()}",
                f"Confidence: {t.get('confidence', 0):.1f}%",
                f"\nIndicators - RSI: {ind.get('rsi_14', 0):.1f}",
                f"MACD: {ind.get('macd', 0):.2f} (Signal: {ind.get('macd_signal', 0):.2f})",
                f"ADX: {ind.get('adx', 0):.1f}",
                f"Stochastic: {ind.get('stoch_k', 0):.1f}",
                f"Volume Ratio: {ind.get('volume_ratio', 0):.2f}x",
                f"BB Position: {ind.get('bb_position', 0):.2f}",
                # NEW: Add more directional indicators
                f"\nDirectional Indicators -", 
                f"EMA 9/21/50: {ind.get('ema_9', 0):.2f}/{ind.get('ema_21', 0):.2f}/{ind.get('ema_50', 0):.2f}",
                f"Price vs EMA21: {(t.get('current_price', 0)/ind.get('ema_21', 1) - 1) * 100:.2f}%",
                f"DI+: {ind.get('di_plus', 0):.1f}, DI-: {ind.get('di_minus', 0):.1f}",
                f"ATR Percent: {ind.get('atr_percent', 0):.2f}%",
            ]
            
            # Add key levels
            levels = t.get('key_levels', {})
            if levels:
                context_parts.append(f"Support: ${levels.get('support', 0):.2f}, Resistance: ${levels.get('resistance', 0):.2f}")
            
            # Add market regime
            regime = t.get('market_regime', {})
            if regime:
                context_parts.append(f"Market Regime: {regime.get('regime', 'Unknown')}")
            
            return "Technical Analysis Data:\n" + "\n".join(context_parts) + "\n\nEXPLAIN each indicator above in simple terms with what it means for the price."
        return ""
