"""
Nunno Finance - FastAPI Backend
Main application entry point
"""

from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
import os
import asyncio
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)
print("DEBUG: Successfully loaded environment variables")
print("DEBUG: Starting Nunno Finance Backend...")

# Import database and auth
from database import init_db, get_db, User, Prediction
from services.auth_service import create_access_token, verify_token, hash_password, verify_password
from services.usage_service import can_user_search, log_search, get_tier_config

# Import services
from services.technical_analysis import TechnicalAnalysisService
from services.chat_service import ChatService
from services.tokenomics_service import TokenomicsService
from services.news_service import NewsService
from services.websocket_service import BinanceWebSocketService
from services.pattern_recognition_service import pattern_service
from services.market_service import MarketService

# Initialize WebSocket service
websocket_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global websocket_service
    init_db()
    print("✅ Database initialized successfully!")
    
    # Start WebSocket service
    websocket_service = BinanceWebSocketService()
    asyncio.create_task(websocket_service.start())
    print("✅ WebSocket service started!")
    
    yield
    
    # Shutdown
    if websocket_service:
        await websocket_service.stop()
        print("✅ WebSocket service stopped!")

app = FastAPI(
    title="Nunno Finance API",
    description="Empathetic AI Financial Educator for Beginners",
    version="1.0.0",
    lifespan=lifespan
)

# Global Exception Handler (Unbreakable Guard)
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    import traceback
    print(f"🔥 UNCAUGHT_CRITICAL_ERROR: {str(exc)}")
    traceback.print_exc()
    return json.dumps({
        "error": "Internal Systems Disrupted",
        "message": "Nunno's neural nodes are stabilizing. Please retry in a moment.",
        "type": "NeuralSystemTimeout"
    }), 500

# CORS configuration for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for production/hosting
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer(auto_error=False) # Allow optional auth

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), db = Depends(get_db)):
    """Get current authenticated user from JWT token"""
    if not credentials:
        print("❌ AUTH ERROR: No credentials provided")
        raise HTTPException(status_code=401, detail="Authentication required")
        
    token = credentials.credentials
    payload = verify_token(token)
    if not payload:
        print("❌ AUTH ERROR: Token verification failed")
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    # Cast to int for Postgres strictness
    try:
        user_id = int(payload.get("user_id"))
        user = db.query(User).filter(User.id == user_id).first()
    except (ValueError, TypeError):
        print(f"❌ AUTH ERROR: Invalid user_id format in token: {payload.get('user_id')}")
        raise HTTPException(status_code=401, detail="Invalid token format")
        
    if not user:
        print(f"❌ AUTH ERROR: User {user_id} not found in DB")
        raise HTTPException(status_code=401, detail="User not found")
    return user

def get_optional_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), db = Depends(get_db)):
    """Get current user if authenticated, otherwise return None (Guest)"""
    if not credentials:
        return None
        
    token = credentials.credentials
    payload = verify_token(token)
    if not payload:
        return None
    
    try:
        user_id = int(payload.get("user_id"))
        user = db.query(User).filter(User.id == user_id).first()
        return user
    except:
        return None

# Initialize services with error handling
try:
    technical_service = TechnicalAnalysisService()
except Exception as e:
    print(f"Failed to initialize TechnicalAnalysisService: {e}")
    technical_service = None

try:
    chat_service = ChatService()
except Exception as e:
    print(f"Failed to initialize ChatService: {e}")
    chat_service = None

try:
    tokenomics_service = TokenomicsService()
except Exception as e:
    print(f"Failed to initialize TokenomicsService: {e}")
    tokenomics_service = None

try:
    news_service = NewsService()
except Exception as e:
    print(f"Failed to initialize NewsService: {e}")
    news_service = None

try:
    market_service = MarketService()
except Exception as e:
    print(f"Failed to initialize MarketService: {e}")
    market_service = None

# Request/Response Models
class SignupRequest(BaseModel):
    email: str
    password: str
    name: str
    experience_level: Optional[str] = "pro"

class LoginRequest(BaseModel):
    email: str
    password: str

class ChatRequest(BaseModel):
    message: str
    conversation_id: str # Required for history separation
    user_age: int = 18

class ChatResponse(BaseModel):
    response: str
    tool_calls: Optional[List[str]] = []
    data_used: Optional[Dict[str, Any]] = {}

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "Nunno Finance API",
        "version": "1.0.0"
    }

# ==================== AUTHENTICATION ENDPOINTS ====================

@app.post("/api/auth/signup")
async def signup(request: SignupRequest, db = Depends(get_db)):
    """Register a new user with robust defaults"""
    try:
        # Check if user exists
        existing = db.query(User).filter(User.email == request.email).first()
        if existing:
            raise HTTPException(status_code=400, detail="Email already registered")
        
        from datetime import datetime
        # Create user with all scaling columns initialized
        user = User(
            email=request.email,
            password_hash=hash_password(request.password),
            name=request.name,
            tier="free",
            experience_level=request.experience_level,
            tokens_remaining=10000,
            tokens_used_today=0,
            searches_today=0,
            last_reset=datetime.utcnow()
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        
        # Create token
        token = create_access_token({"user_id": str(user.id), "email": user.email})
        
        # Get tier limits
        from services.usage_service import get_tier_config
        limits = get_tier_config(user.tier)
        
        print(f"✅ Successful signup for: {user.email}")
        return {
            "token": token,
            "user": {
                "id": str(user.id),
                "email": user.email,
                "name": user.name,
                "tier": user.tier,
                "experience_level": user.experience_level,
                "tokens_remaining": user.tokens_remaining,
                "searches_today": user.searches_today,
                "tokens_used_today": user.tokens_used_today,
                "limits": limits
            }
        }
    except Exception as e:
        print(f"❌ SIGNUP ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Signup failed: {str(e)}")

@app.post("/api/auth/login")
async def login(request: LoginRequest, db = Depends(get_db)):
    """Login existing user with full profile return"""
    try:
        user = db.query(User).filter(User.email == request.email).first()
        if not user:
            print(f"❌ LOGIN AUTH ERROR: Email {request.email} not found")
            raise HTTPException(status_code=401, detail="Invalid email or password")
            
        if not verify_password(request.password, user.password_hash):
            print(f"❌ LOGIN AUTH ERROR: Password mismatch for {request.email}")
            raise HTTPException(status_code=401, detail="Invalid email or password")
        
        token = create_access_token({"user_id": str(user.id), "email": user.email})
        
        # Get tier limits
        limits = get_tier_config(user.tier)
        
        print(f"✅ SUCCESS: Logged in {user.email} (ID: {user.id})")
        return {
            "token": token,
            "user": {
                "id": str(user.id),
                "email": user.email,
                "name": user.name,
                "tier": user.tier,
                "experience_level": user.experience_level,
                "tokens_remaining": user.tokens_remaining,
                "tokens_used_today": user.tokens_used_today,
                "searches_today": user.searches_today,
                "limits": limits
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ LOGIN CRITICAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error during login")

@app.get("/api/v1/me")
@app.get("/api/auth/me")
async def get_me(current_user: User = Depends(get_current_user), db = Depends(get_db)):
    """Get current user info"""
    # Get tier limits
    limits = get_tier_config(current_user.tier)
    
    return {
        "id": str(current_user.id),
        "email": current_user.email,
        "name": current_user.name,
        "tier": current_user.tier,
        "experience_level": current_user.experience_level,
        "tokens_remaining": current_user.tokens_remaining,
        "tokens_used_today": current_user.tokens_used_today,
        "searches_today": current_user.searches_today,
        "limits": limits
    }

# ==================== EXISTING ENDPOINTS ====================

@app.get("/api/v1/technical/{ticker}")
async def get_technical_analysis(ticker: str, interval: str = "15m", current_user: Optional[User] = Depends(get_optional_current_user), db = Depends(get_db)):
    """
    Get technical analysis for a cryptocurrency (Quota limited for authenticated users)
    """
    if current_user:
        from services.usage_service import can_user_search, log_search
        if not can_user_search(current_user, db):
            raise HTTPException(status_code=402, detail="Daily technical analysis limit reached.")
        log_search(current_user.id, db)
        
    try:
        result = technical_service.analyze(ticker, interval)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/simulate/monte-carlo/{ticker}")
async def get_monte_carlo_simulation(ticker: str, interval: str = "15m", enhanced: bool = True):
    """
    Get Monte Carlo probability fan for a ticker (Elite Chart 2.0)
    
    Args:
        ticker: Trading pair (e.g., BTCUSDT)
        interval: Timeframe (e.g., 15m, 1h, 4h, 1d)
        enhanced: If True, returns comprehensive scenarios with risk metrics (default: True)
    """
    try:
        if not technical_service:
            raise HTTPException(status_code=503, detail="Technical service unavailable")
        return technical_service.get_monte_carlo(ticker.upper(), interval, enhanced)
    except Exception as e:
        print(f"Simulation Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/simulate/regime/{ticker}")
async def get_regime_simulation(ticker: str, type: str, interval: str = "15m"):
    """Get specific regime-injected simulation path (Elite Chart 2.0)"""
    try:
        if not technical_service:
            raise HTTPException(status_code=503, detail="Technical service unavailable")
        return technical_service.simulate_scenario(ticker.upper(), type, interval)
    except Exception as e:
        print(f"Regime Simulation Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/tokenomics/{coin_id}")
async def get_tokenomics(coin_id: str, investment_amount: float = 1000):
    """
    Get comprehensive tokenomics analysis
    
    Args:
        coin_id: CoinGecko coin ID (e.g., bitcoin, ethereum)
        investment_amount: Investment amount for calculations
    
    Returns:
        Tokenomics data with beginner explanations
    """
    try:
        result = tokenomics_service.analyze(coin_id, investment_amount)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/news/{ticker}")
async def get_news(ticker: str):
    """
    Get market news and sentiment
    
    Args:
        ticker: Cryptocurrency ticker
    
    Returns:
        News and sentiment analysis
    """
    try:
        result = news_service.get_news_sentiment(ticker)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== WEBSOCKET ENDPOINT ====================

@app.websocket("/ws/prices")
async def websocket_prices(websocket: WebSocket):
    """
    WebSocket endpoint for real-time cryptocurrency prices
    Streams live price updates from Binance
    """
    await websocket.accept()
    
    if not websocket_service:
        await websocket.close(code=1011, reason="WebSocket service unavailable")
        return
    
    try:
        # Add client to service
        await websocket_service.add_client(websocket)
        
        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Wait for messages from client (ping/pong or subscription requests)
                data = await websocket.receive_text()
                
                # Parse the message
                try:
                    message = json.loads(data)
                    message_type = message.get('type')
                    
                    if message_type == 'subscribe_kline':
                        # Handle kline subscription
                        symbol = message.get('symbol', 'btcusdt')
                        interval = message.get('interval', '1m')
                        await websocket_service.add_kline_client(websocket, symbol.upper(), interval)
                    elif message_type == 'unsubscribe_kline':
                        # Handle kline unsubscription
                        symbol = message.get('symbol', 'btcusdt')
                        interval = message.get('interval', '1m')
                        await websocket_service.remove_kline_client(websocket, symbol.upper(), interval)
                    elif message_type == 'ping':
                        # Respond to ping
                        await websocket.send_text(json.dumps({"type": "pong"}))
                    else:
                        # Unknown message type, send pong as default
                        await websocket.send_text(json.dumps({"type": "pong"}))
                        
                except json.JSONDecodeError:
                    # If not JSON, treat as ping
                    await websocket.send_text(json.dumps({"type": "pong"}))
                    
            except WebSocketDisconnect:
                break
            except Exception as e:
                print(f"WebSocket error: {e}")
                break
                
    finally:
        # Remove client from service correctly
        await websocket_service.remove_client(websocket)
        # Note: Kline unsubscription is handled internally if the client is added correctly

@app.get("/api/v1/macro/summary")
async def get_macro_summary():
    """
    Get a pre-synthesized macro-to-crypto intelligence summary.
    """
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            # High-impact macro queries
            results = list(ddgs.text("global economy news today crypto impacts", max_results=5, timelimit='d'))
            headlines = [r.get('title') for r in results if r.get('title')]
        
        # Use ChatService to create a tiny summary
        news_context = "\n".join(headlines)
        prompt = f"Summarize these macro headlines into 3 bullet points explaining their effect on the crypto market (Bullish/Bearish sentiment). Use elite, professional language.\n\nHeadlines:\n{news_context}"
        
        # Quick non-streaming call
        response = await chat_service.client.chat.completions.create(
            model="gpt-3.5-turbo", # Fixed small model for speed
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.5
        )
        return {
            "summary": response.choices[0].message.content,
            "headlines": headlines
        }
    except Exception as e:
        return {"summary": "Macro catalysts are currently stabilizing. Focus on technical liquidity zones.", "headlines": []}

# ==================== REST ENDPOINTS ====================


@app.get("/api/v1/price-history/{ticker}")
async def get_price_history(ticker: str, timeframe: str = "24H"):
    """
    Get price history for charts with selectable timeframe
    Timeframes: 24H (default), 7D, 30D, 1Y
    """
    try:
        # Use existing technical service to fetch data
        if not technical_service:
            raise HTTPException(status_code=503, detail="Technical service unavailable")
            
        # Map timeframe to Binance interval and limit
        # Binance max limit is usually 1000. We want a decent resolution.
        timeframe_map = {
            "24H": {"interval": "15m", "limit": 96},   # 96 * 15m = 24 hours
            "7D":  {"interval": "1h",  "limit": 168},  # 168 * 1h = 7 days
            "30D": {"interval": "4h",  "limit": 180},  # 180 * 4h = 30 days
            "1Y":  {"interval": "1d",  "limit": 365}   # 365 * 1d = 1 year
        }
        
        config = timeframe_map.get(timeframe, timeframe_map["24H"])
        
        df = technical_service.analyzer.fetch_binance_ohlcv(
            symbol=ticker, 
            interval=config["interval"], 
            limit=config["limit"]
        )
        
        # Format for recharts [ { time: '...', price: 123 } ]
        history = []
        for index, row in df.iterrows():
            # Format time label based on timeframe
            if timeframe == "24H":
                time_label = index.strftime("%H:%M")
            elif timeframe == "7D":
                time_label = index.strftime("%a %H:%M")
            elif timeframe == "30D":
                time_label = index.strftime("%b %d")
            else: # 1Y
                time_label = index.strftime("%b %d %Y")

            history.append({
                "time": time_label,
                "price": float(row['Close']),
                "date": index.isoformat() # Full ISO date for tooltips
            })
            
        # Calculate percent change
        if len(df) > 0:
            current_price = float(df.iloc[-1]['Close'])
            open_price = float(df.iloc[0]['Close']) 
            percent_change = ((current_price - open_price) / open_price) * 100
            
            # Additional stats
            high_price = float(df['High'].max())
            low_price = float(df['Low'].min())
        else:
            current_price = 0
            percent_change = 0
            high_price = 0
            low_price = 0

        return {
            "ticker": ticker,
            "current_price": current_price,
            "percent_change": percent_change,
            "high_price": high_price,
            "low_price": low_price,
            "history": history,
            "timeframe": timeframe
        }
    except Exception as e:
        print(f"Error fetching price history: {e}")
        # Return mock data on failure to prevent UI crash
        import random
        points = 20
        mock_history = [{"time": str(i), "price": 50000 + random.randint(-1000, 1000)} for i in range(points)]
        return {
            "ticker": ticker,
            "current_price": 50000,
            "percent_change": 2.5,
            "high_price": 52000,
            "low_price": 48000,
            "history": mock_history,
            "is_mock": True
        }

@app.get("/api/v1/simulation/scenario/{ticker}")
async def get_simulation_scenario(ticker: str, injection_type: str, interval: str = "15m"):
    """Generate a specific injected market scenario"""
    if not technical_service:
        raise HTTPException(status_code=503, detail="Technical service unavailable")
    try:
        return technical_service.simulate_scenario(ticker.upper(), injection_type, interval)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/simulation/monte-carlo/{ticker}")
async def get_monte_carlo(ticker: str, interval: str = "15m", enhanced: bool = True):
    """
    Generate Monte Carlo probability fan with enhanced features
    """
    if not technical_service:
        raise HTTPException(status_code=503, detail="Technical service unavailable")
    try:
        return technical_service.get_monte_carlo(ticker.upper(), interval, enhanced)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/conversations")
async def get_conversations(current_user: User = Depends(get_current_user), db = Depends(get_db)):
    """List user chats"""
    conversations = db.query(Conversation).filter(Conversation.user_id == current_user.id).order_by(Conversation.created_at.desc()).all()
    return [{
        "id": c.id,
        "title": c.title,
        "created_at": c.created_at
    } for c in conversations]

@app.post("/api/v1/chat")
async def chat(request: ChatRequest, current_user: User = Depends(get_current_user), db = Depends(get_db)):
    """Authenticated Chat with Quota Management"""
    from services.usage_service import can_user_chat
    
    # 1. Check Quota
    if not can_user_chat(current_user, db):
        raise HTTPException(status_code=402, detail="Daily token limit reached. Upgrade to Pro for more!")

    # 2. Process
    try:
        response = await chat_service.process_message(
            message=request.message,
            user=current_user,
            conversation_id=request.conversation_id,
            db=db,
            user_age=request.user_age
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/chat/stream")
async def chat_stream(request: ChatRequest, current_user: User = Depends(get_current_user), db = Depends(get_db)):
    """Streaming Chat with Quota Management"""
    from services.usage_service import can_user_chat
    if not can_user_chat(current_user, db):
        raise HTTPException(status_code=402, detail="Daily token limit reached.")

    return StreamingResponse(
        chat_service.stream_message(
            message=request.message,
            user=current_user,
            conversation_id=request.conversation_id,
            db=db,
            user_age=request.user_age
        ),
        media_type="text/event-stream"
    )

# ==================== FEED NUNNO ENDPOINT ====================

class FeedNunnoRequest(BaseModel):
    symbol: str = "BTCUSDT"
    timeframe: str = "15m"
    user_name: str = "User"

@app.post("/api/v1/analyze/feed-nunno")
async def feed_nunno(request: FeedNunnoRequest, current_user: User = Depends(get_current_user), db = Depends(get_db)):
    """
    Feed Nunno - Comprehensive Market Intelligence Report (Authenticated)
    """
    from services.usage_service import can_user_search, log_search
    
    # 1. Quota Check (This is an expensive search-like operation)
    if not can_user_search(current_user, db):
        raise HTTPException(status_code=402, detail="Daily intelligence report limit reached.")

    if not chat_service or not technical_service or not news_service:
        raise HTTPException(status_code=503, detail="Required services unavailable")
    
    # Log usage
    log_search(current_user.id, db)
    
    try:
        async def generate_feed_report():
            # Parallel aggregation of all required data to save 5-10 seconds
            yield f"data: {json.dumps({'type': 'status', 'message': 'Initializing Neural Aggregators...'})}\n\n"
            
            # Define helper for macro news to keep it in a thread
            def get_macro_news():
                headlines = []
                try:
                    from duckduckgo_search import DDGS
                    with DDGS() as ddgs:
                        # Broader queries for better hit rate
                        queries = ["global crypto market news today", "FED interest rates impact crypto", "bitcoin ethereum regulation news"]
                        for query in queries:
                            results = list(ddgs.text(query, max_results=3, timelimit='d'))
                            for r in results:
                                headlines.append({
                                    "title": r.get("title", ""),
                                    "source": "Global Macro",
                                    "url": r.get("href", "")
                                })
                            if len(headlines) >= 5: break
                except Exception as e:
                    print(f"Macro news fetch fail: {e}")
                return headlines

            # Start all heavy I/O tasks in parallel
            timeframes = ["15m", "1h", "4h", "1d"]
            tasks = [
                asyncio.to_thread(news_service.get_news_sentiment, request.symbol),
                asyncio.to_thread(get_macro_news),
                asyncio.to_thread(technical_service.analyzer.fetch_binance_ohlcv, symbol=request.symbol, interval="15m", limit=96)
            ]
            # Add technical analysis tasks for each timeframe
            for tf in timeframes:
                tasks.append(asyncio.to_thread(technical_service.analyze, request.symbol, tf))

            # Wait for all data (this happens much faster now)
            yield f"data: {json.dumps({'type': 'status', 'message': 'Synchronizing Global Data Nodes...'})}\n\n"
            results = await asyncio.gather(*tasks)
            
            news_data = results[0]
            macro_headlines = results[1]
            df_24h = results[2]
            multi_tf_data = {tf: results[i+3] for i, tf in enumerate(timeframes)}

            # Phase 3: Market Density & Stats calculation
            curr_price = 0
            stats_24h = {"high": 0, "low": 0, "var": 0, "std": 0}
            if not df_24h.empty:
                curr_price = float(df_24h.iloc[-1]['Close'])
                stats_24h["high"] = float(df_24h['High'].max())
                stats_24h["low"] = float(df_24h['Low'].min())
                stats_24h["var"] = ((curr_price - float(df_24h.iloc[0]['Open'])) / float(df_24h.iloc[0]['Open'])) * 100
                stats_24h["std"] = float(df_24h['Close'].std())

            # Phase 4: Construct Hyper-Detailed Prompt
            yield f"data: {json.dumps({'type': 'status', 'message': 'Synthesizing Narrative Confluences...'})}\n\n"
            
            # Format Data for Prompt with explicit labels
            crypto_news_list = news_data.get('headlines', [])
            macro_news_list = macro_headlines
            
            crypto_news_str = "\n".join([f"- {h.get('title')} [{h.get('source')}]" for h in crypto_news_list[:5]])
            macro_news_str = "\n".join([f"- {h.get('title')}" for h in macro_news_list[:5]])
            
            tf_details = ""
            for tf, data in multi_tf_data.items():
                if data:
                    ind = data.get('indicators', {})
                    tf_details += f"\nTIME_FRAME [{tf.upper()}]:\n"
                    tf_details += f" - BIAS: {data.get('bias')} | Confidence: {data.get('confidence')}%\n"
                    tf_details += f" - RSI_INDEX: {ind.get('rsi_14', ind.get('rsi', 'N/A'))}\n"
                    tf_details += f" - MACD_VAL: {ind.get('macd', 'N/A')} | MACD_SIGNAL: {ind.get('macd_signal', 'N/A')}\n"
                    tf_details += f" - ADX_STRENGTH: {ind.get('adx', 'N/A')}\n"
                    tf_details += f" - VOL_RATIO: {ind.get('volume_ratio', 'N/A')}x\n"
                    tf_details += f" - LEVELS: Support ${data.get('support_levels', ['N/A'])[0]} | Resistance ${data.get('resistance_levels', ['N/A'])[0]}\n"

            prompt = f"""SYSTEM: You are Nunno, an Elite Neural Analyst. 
Generate a MARKET INTELLIGENCE BRIEFING for {request.symbol}.

STRICT REQUIREMENT: 
Your analysis MUST lead with a synthesis of the **specific news headlines** provided below. Explain precisely how these events are creating the technical conditions shown in the data. Do not hallucinate external news.

DATA FEED:
1. RAW ASSET STATS:
 - Spot Price: ${curr_price:,.2f}
 - 24h Variance: {stats_24h['var']:+.2f}%
 - Volatility (STD): ${stats_24h['std']:.2f}
 - Fear & Greed: {news_data.get('fear_greed_index', {}).get('value')}/100

2. ACTUAL NEWS HEADLINES (CRITICAL):
{crypto_news_str if crypto_news_str else "N/A - Asset news is flat. Rely on Global Macro."}

3. GLOBAL MACRO CONTEXT:
{macro_news_str if macro_news_str else "N/A - Monitor liquidity for catalysts."}

4. TECHNICAL MATRIX:
{tf_details}

OUTPUT_NARRATIVE:
1. **News-Technical Correlation**: Start by explicitly summarizing the **5 news headlines** listed under ACTUAL NEWS HEADLINES above. Explain how these specific events are directly influencing {request.symbol}'s current bias and creating the technical levels observed (RSI, MACD etc).
2. **Structural Snapshot**: Provide a MARKDOWN TABLE comparing the 4 timeframes. Use the specific values provided in the DATA FEED (RSI, ADX, Support/Resistance).
3. **Indicator Confluence**: Deep dive into the 15m/1h indicators. Explain the RSI and MACD values provided.
4. **Market Density**: Is the market 'Chopping' or 'Rupturing'? Use the Variance and STD data to explain.
5. **Architectural Sentiment**: 3-5 high-level tactical observations.
6. **The Path Forward**: An executive summary on the next probable move.

BE DETAILED. USE PROFESSIONAL TERMINOLOGY. USE MARKDOWN.
Stream the FULL ANALYTICAL BRIEFING now:"""

            # Phase 5: Stream AI Analysis
            yield f"data: {json.dumps({'type': 'status', 'message': 'Generating Final Intelligence Briefing...'})}\n\n"
            async for chunk in chat_service.stream_message(
                message=prompt,
                user=current_user,
                conversation_id=f"feed_{request.symbol}_{current_user.id}",
                db=db,
                user_age=18
            ):
                yield chunk
            
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        return StreamingResponse(generate_feed_report(), media_type="text/event-stream")
        
    except Exception as e:
        print(f"Feed Nunno Critical Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== PATTERN RECOGNITION ENDPOINTS ====================

class PatternRequest(BaseModel):
    query: str
    base_price: Optional[float] = 50000
    num_points: Optional[int] = 50
    interval: Optional[str] = "1d"

@app.post("/api/v1/pattern/recognize")
async def recognize_pattern(request: PatternRequest):
    """
    Recognize chart pattern from user query and generate visualization data
    
    Args:
        query: User's pattern request (e.g., "show me a head and shoulders pattern")
        base_price: Starting price for pattern generation
        num_points: Number of data points to generate
        interval: Timeframe interval (e.g., "1m", "1h", "1d")
    
    Returns:
        Pattern data structured for Recharts visualization
    """
    try:
        # Recognize pattern from query
        pattern_name = pattern_service.recognize_pattern(request.query)
        
        if not pattern_name:
            # Fallback to general chat if no specific pattern is recognized
            if chat_service:
                chat_response = await chat_service.process_message(request.query)
                return {
                    "success": True, # Still success, just not a pattern
                    "message": chat_response.get("response", "I couldn't recognize that pattern or question."),
                    "is_pattern": False
                }
            
            return {
                "success": False,
                "message": "I couldn't recognize a specific chart pattern in your request. Try asking for patterns like 'head and shoulders', 'double top', 'ascending triangle', etc.",
                "available_patterns": list(pattern_service.PATTERNS.keys())
            }
        
        # Generate pattern data
        pattern_data = pattern_service.generate_pattern_data(
            pattern_name=pattern_name,
            base_price=request.base_price,
            num_points=request.num_points,
            interval=request.interval
        )
        
        return {
            "success": True,
            "pattern": pattern_data,
            "message": f"Generated {pattern_name.replace('_', ' ').title()} pattern"
        }
        
    except Exception as e:
        print(f"Pattern recognition error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/pattern/list")
async def list_patterns():
    """
    Get list of all available chart patterns
    
    Returns:
        List of available patterns with their metadata
    """
    patterns = []
    for pattern_name, pattern_info in pattern_service.PATTERNS.items():
        patterns.append({
            "name": pattern_name,
            "display_name": pattern_name.replace('_', ' ').title(),
            "type": pattern_info['type'],
            "direction": pattern_info['direction'],
            "keywords": pattern_info['keywords']
        })
    
    return {
        "patterns": patterns,
        "total": len(patterns)
    }

@app.get("/api/v1/market/highlights")
async def get_market_highlights():
    """Get market-wide highlights (gainers, losers, new listings)"""
    return market_service.get_market_highlights()

if __name__ == "__main__":
    import uvicorn
    # Use import string for reload to work correctly and avoid warnings
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
