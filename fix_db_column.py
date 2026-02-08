import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./nunno.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

print(f"Connecting to: {DATABASE_URL}")
engine = create_engine(DATABASE_URL)

def add_column():
    column_name = "experience_level"
    table_name = "users"
    
    with engine.connect() as conn:
        try:
            # Check if column exists
            if "sqlite" in DATABASE_URL:
                # SQLite check
                result = conn.execute(text(f"PRAGMA table_info({table_name})"))
                columns = [row[1] for row in result]
            else:
                # Postgres check
                result = conn.execute(text(f"SELECT column_name FROM information_schema.columns WHERE table_name='{table_name}' AND column_name='{column_name}'"))
                columns = [row[0] for row in result]
            
            if column_name not in columns:
                print(f"Adding '{column_name}' to '{table_name}'...")
                # Add column with default value 'pro'
                if "sqlite" in DATABASE_URL:
                    conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} TEXT DEFAULT 'pro'"))
                else:
                    conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} VARCHAR DEFAULT 'pro'"))
                conn.commit()
                print("✅ Column added successfully!")
            else:
                print(f"✅ Column '{column_name}' already exists.")
        except Exception as e:
            print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    add_column()
