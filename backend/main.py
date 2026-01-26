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
from services.usage_service import can_user_search, log_usage, get_tier_limits

# Import services
from services.technical_analysis import TechnicalAnalysisService
from services.chat_service import ChatService
from services.tokenomics_service import TokenomicsService
from services.news_service import NewsService
from services.websocket_service import BinanceWebSocketService
from services.pattern_recognition_service import pattern_service

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

# CORS configuration for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer(auto_error=False) # Allow optional auth

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), db = Depends(get_db)):
    """Get current authenticated user from JWT token"""
    if not credentials:
        raise HTTPException(status_code=401, detail="Authentication required")
        
    token = credentials.credentials
    payload = verify_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    user = db.query(User).filter(User.id == payload.get("user_id")).first()
    if not user:
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
    
    user = db.query(User).filter(User.id == payload.get("user_id")).first()
    return user

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

# Request/Response Models
class SignupRequest(BaseModel):
    email: str
    password: str
    name: str

class LoginRequest(BaseModel):
    email: str
    password: str

class ChatRequest(BaseModel):
    message: str
    user_name: str = "User"
    user_age: int = 18
    conversation_history: Optional[List[Dict[str, str]]] = []

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
    """Register a new user"""
    # Check if user exists
    existing = db.query(User).filter(User.email == request.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create user
    user = User(
        email=request.email,
        password_hash=hash_password(request.password),
        name=request.name,
        tier="free",
        tokens_remaining=1000
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    
    # Create token
    token = create_access_token({"user_id": str(user.id), "email": user.email})
    
    return {
        "token": token,
        "user": {
            "id": str(user.id),
            "email": user.email,
            "name": user.name,
            "tier": user.tier,
            "tokens_remaining": user.tokens_remaining,
            "searches_today": user.searches_today
        }
    }

@app.post("/api/auth/login")
async def login(request: LoginRequest, db = Depends(get_db)):
    """Login existing user"""
    user = db.query(User).filter(User.email == request.email).first()
    if not user or not verify_password(request.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    token = create_access_token({"user_id": str(user.id), "email": user.email})
    
    return {
        "token": token,
        "user": {
            "id": str(user.id),
            "email": user.email,
            "name": user.name,
            "tier": user.tier,
            "tokens_remaining": user.tokens_remaining,
            "searches_today": user.searches_today
        }
    }

@app.get("/api/auth/me")
async def get_me(current_user: User = Depends(get_current_user), db = Depends(get_db)):
    """Get current user info"""
    # Get tier limits
    limits = get_tier_limits(current_user.tier)
    
    return {
        "id": str(current_user.id),
        "email": current_user.email,
        "name": current_user.name,
        "tier": current_user.tier,
        "tokens_remaining": current_user.tokens_remaining,
        "searches_today": current_user.searches_today,
        "limits": limits
    }

# ==================== EXISTING ENDPOINTS ====================

@app.get("/api/v1/technical/{ticker}")
async def get_technical_analysis(ticker: str, interval: str = "15m"):
    """
    Get technical analysis for a cryptocurrency
    
    Args:
        ticker: Trading pair (e.g., BTCUSDT)
        interval: Timeframe (e.g., 15m, 1h, 4h, 1d)
    
    Returns:
        Technical analysis with beginner-friendly explanations
    """
    try:
        result = technical_service.analyze(ticker, interval)
        return result
    except Exception as e:
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
        # Remove client from service
        await websocket_service.remove_client(websocket)
        # Also try to remove from kline subscribers if they were subscribed
        # Note: This is a simplified cleanup, ideally we'd track both subscriptions
        await websocket_service.remove_client(websocket)  # This removes from general client list

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
        
        df = technical_service.analyzer.fetch_binance_ohlcv_with_fallback(
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

@app.post("/api/v1/chat")
async def chat(request: ChatRequest):
    """
    Chat with Nunno AI - The Empathetic Financial Educator
    
    This endpoint orchestrates tool calls and provides beginner-friendly responses
    """
    if not chat_service:
        raise HTTPException(status_code=503, detail="Chat service unavailable (initialization failed)")
    try:
        response = await chat_service.process_message(
            message=request.message,
            user_name=request.user_name,
            user_age=request.user_age,
            conversation_history=request.conversation_history
        )
        return response
    except Exception as e:
        print(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming chat endpoint for real-time responses
    """
    if not chat_service:
        raise HTTPException(status_code=503, detail="Chat service unavailable")
    try:
        return StreamingResponse(
            chat_service.stream_message(
                message=request.message,
                user_name=request.user_name,
                user_age=request.user_age,
                conversation_history=request.conversation_history
            ),
            media_type="text/event-stream"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== PATTERN RECOGNITION ENDPOINTS ====================

class PatternRequest(BaseModel):
    query: str
    base_price: Optional[float] = 50000
    num_points: Optional[int] = 50

@app.post("/api/v1/pattern/recognize")
async def recognize_pattern(request: PatternRequest):
    """
    Recognize chart pattern from user query and generate visualization data
    
    Args:
        query: User's pattern request (e.g., "show me a head and shoulders pattern")
        base_price: Starting price for pattern generation
        num_points: Number of data points to generate
    
    Returns:
        Pattern data structured for Recharts visualization
    """
    try:
        # Recognize pattern from query
        pattern_name = pattern_service.recognize_pattern(request.query)
        
        if not pattern_name:
            return {
                "success": False,
                "message": "I couldn't recognize a specific chart pattern in your request. Try asking for patterns like 'head and shoulders', 'double top', 'ascending triangle', etc.",
                "available_patterns": list(pattern_service.PATTERNS.keys())
            }
        
        # Generate pattern data
        pattern_data = pattern_service.generate_pattern_data(
            pattern_name=pattern_name,
            base_price=request.base_price,
            num_points=request.num_points
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

if __name__ == "__main__":
    import uvicorn
    # Use import string for reload to work correctly and avoid warnings
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
