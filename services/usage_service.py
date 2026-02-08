from datetime import datetime
from sqlalchemy.orm import Session
try:
    from database import User
except ImportError:
    from ..database import User

# Production-grade tier configuration with fallback chains
TIER_CONFIGS = {
    "free": {
        "daily_searches": 10,
        "daily_token_limit": 15000, # Approx 20-30 chat turns
        "model": "nvidia/nemotron-3-nano-30b-a3b:free", # Primary: Free Nemotron
        "fallback_model": "meta-llama/llama-3.1-8b-instruct", # Fallback 1: Cheap paid
        "fallback_model_2": "meta-llama/llama-3.2-3b-instruct:free", # Fallback 2: Free backup
        "precision": "low"
    },
    "pro": {
        "daily_searches": 100,
        "daily_token_limit": 100000,
        "model": "meta-llama/llama-3.1-70b-instruct", # Primary: High quality
        "fallback_model": "meta-llama/llama-3.1-8b-instruct", # Fallback 1: Fast & cheap
        "fallback_model_2": "nvidia/nemotron-3-nano-30b-a3b:free", # Fallback 2: Free
        "precision": "high"
    },
    "whale": {
        "daily_searches": 1000,
        "daily_token_limit": 1000000,
        "model": "openai/gpt-4o", # Primary: Premium
        "fallback_model": "meta-llama/llama-3.1-405b-instruct", # Fallback 1: Top open-source
        "fallback_model_2": "meta-llama/llama-3.1-70b-instruct", # Fallback 2: Still great
        "precision": "extreme"
    }
}

def get_tier_config(tier: str):
    return TIER_CONFIGS.get(tier, TIER_CONFIGS["free"])

def check_and_reset_usage(user: User, db: Session):
    """Resets daily usage if 24 hours have passed"""
    now = datetime.utcnow()
    # Check if a new day has started (simplified to 24h delta)
    if (now - user.last_reset).days >= 1:
        user.searches_today = 0
        user.tokens_used_today = 0
        user.last_reset = now
        db.commit()

def can_user_chat(user: User, db: Session) -> bool:
    if not user: return False
    
    check_and_reset_usage(user, db)
    config = get_tier_config(user.tier)
    
    # Block if either limit is hit
    if user.tokens_used_today >= config["daily_token_limit"]:
        return False
        
    return True

def can_user_search(user: User, db: Session) -> bool:
    if not user: return False
    
    check_and_reset_usage(user, db)
    config = get_tier_config(user.tier)
    
    if user.searches_today >= config["daily_searches"]:
        return False
        
    return True

def log_search(user_id: int, db: Session):
    user = db.query(User).filter(User.id == user_id).first()
    if user:
        user.searches_today += 1
        db.commit()

def log_token_usage(user_id: int, tokens: int, db: Session):
    user = db.query(User).filter(User.id == user_id).first()
    if user:
        user.tokens_used_today += tokens
        # Deduct from total balance if applicable
        if user.tokens_remaining > 0:
            user.tokens_remaining -= tokens
        db.commit()
