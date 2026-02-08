import os
import time
import threading
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Set up a test database
DB_URL = "sqlite:///test_stress.db"
engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class TestUser(Base):
    __tablename__ = "test_users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)

Base.metadata.create_all(bind=engine)

def worker(worker_id, num_iterations):
    db = SessionLocal()
    try:
        for i in range(num_iterations):
            user = TestUser(name=f"Worker {worker_id} - Iter {i}")
            db.add(user)
            db.commit()
            # print(f"Worker {worker_id} committed {i}")
    except Exception as e:
        print(f"Worker {worker_id} failed: {e}")
    finally:
        db.close()

def run_stress_test(num_workers, iterations_per_worker):
    print(f"Starting stress test with {num_workers} workers, {iterations_per_worker} iterations each...")
    start_time = time.time()
    threads = []
    for i in range(num_workers):
        t = threading.Thread(target=worker, args=(i, iterations_per_worker))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    end_time = time.time()
    print(f"Finished in {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    # Test with 10 concurrent workers
    run_stress_test(10, 50)
    # Test with 50 concurrent workers (more likely to hit locking issues)
    run_stress_test(50, 20)
    
    # Cleanup
    if os.path.exists("test_stress.db"):
        os.remove("test_stress.db")
