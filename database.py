from sqlalchemy import create_engine, Column, Integer, String, Boolean, JSON, Float, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import os
from datetime import datetime

# Database setup
# SQLite for local development/fallback, PostgreSQL for production
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./nunno.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Configuration for different database types
engine_args = {}
if DATABASE_URL.startswith("sqlite"):
    engine_args["connect_args"] = {"check_same_thread": False}
else:
    # For Postgres, add pool settings and reasonable timeouts
    engine_args.update({
        "pool_pre_ping": True,
        "pool_recycle": 3600,
        "connect_args": {
            "connect_timeout": 10
        }
    })

try:
    print(f"DEBUG: Attempting to connect to database...")
    # Hide password in logs
    masked_url = DATABASE_URL
    if "@" in masked_url:
        start = masked_url.find("://") + 3
        end = masked_url.find("@")
        masked_url = masked_url[:start] + "****" + masked_url[end:]
    print(f"DEBUG: Using URL: {masked_url}")
    
    engine = create_engine(DATABASE_URL, **engine_args)
    # Test connection immediately
    with engine.connect() as conn:
        print("[OK] Database connection verified successfully!")
except Exception as e:
    print(f"[ERROR] DATABASE CONNECTION ERROR: {str(e)}")
    print("[WARNING] Could not connect to the specified DATABASE_URL.")
    
    # If connection fails and it's not already SQLite, try falling back to SQLite
    # This ensures the app can at least start on platforms like Hugging Face
    if not DATABASE_URL.startswith("sqlite"):
        print("[FALLBACK] Falling back to local SQLite (nunno.db) to allow app startup...")
        DATABASE_URL = "sqlite:///./nunno.db"
        engine_args = {"connect_args": {"check_same_thread": False}}
        engine = create_engine(DATABASE_URL, **engine_args)
    else:
        # If even SQLite fails (unlikely), re-raise
        raise e

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    password_hash = Column(String)
    name = Column(String)
    tier = Column(String, default="free") # free, pro, whale
    experience_level = Column(String, default="pro") # beginner, pro
    
    # Usage Tracking
    tokens_remaining = Column(Integer, default=10000) # Total tokens allowed in current cycle
    tokens_used_today = Column(Integer, default=0)
    searches_today = Column(Integer, default=0)
    last_reset = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    conversations = relationship("Conversation", back_populates="user")

class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(String, primary_key=True, index=True) # UUID
    user_id = Column(Integer, ForeignKey("users.id"))
    title = Column(String, default="New Chat")
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")

class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(String, ForeignKey("conversations.id"))
    role = Column(String) # user, assistant, system
    content = Column(String)
    tokens_used = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    conversation = relationship("Conversation", back_populates="messages")

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer) 
    ticker = Column(String)
    prediction_data = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
