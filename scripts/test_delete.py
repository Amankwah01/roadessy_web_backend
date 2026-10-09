import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, text
from app.database.session import DATABASE_URL

engine = create_engine(DATABASE_URL)
with engine.connect() as conn:
    # Start transaction
    trans = conn.begin()
    try:
        print("Before delete:")
        result = conn.execute(text("SELECT COUNT(*) FROM projects"))
        print(f"  projects: {result.scalar()}")
        
        result = conn.execute(text("DELETE FROM survey_runs"))
        print(f"Deleted {result.rowcount} survey_runs")
        
        result = conn.execute(text("DELETE FROM projects"))
        print(f"Deleted {result.rowcount} projects")
        
        result = conn.execute(text("SELECT COUNT(*) FROM projects"))
        print(f"After delete projects: {result.scalar()}")
        
        trans.commit()
        print("COMMITTED")
    except Exception as e:
        trans.rollback()
        print(f"ERROR: {e}")
        raise