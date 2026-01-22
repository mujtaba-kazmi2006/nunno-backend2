from datetime import datetime
from sqlalchemy.orm import Session
# Use absolute import assuming backend is in python path or relative import if needed
# But since main.py runs from backend/, 'database' is a top-level module.
try:
    from database import User
except ImportError:
    # Fallback for relative import if run differently
    from ..database import User

TIER_LIMITS = {
    "free": {"daily_searches": 5, "tokens_per_day": 1000},
    "pro": {"daily_searches": 50, "tokens_per_day": 10000},
    "whale": {"daily_searches": 9999, "tokens_per_day": 100000}
}

def get_tier_limits(tier: str):
    return TIER_LIMITS.get(tier, TIER_LIMITS["free"])

def can_user_search(user: User) -> bool:
    if not user:
        return False
    limits = get_tier_limits(user.tier)
    return user.searches_today < limits["daily_searches"]

def log_usage(user_id: int, usage_type: str, db: Session):
    user = db.query(User).filter(User.id == user_id).first()
    if user:
        if usage_type == "search":
            user.searches_today += 1
        db.commit()
