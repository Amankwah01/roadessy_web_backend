import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, text
from app.database.session import DATABASE_URL

engine = create_engine(DATABASE_URL)
with engine.connect() as conn:
    for table in ['projects', 'roads', 'organisations']:
        result = conn.execute(text("SELECT COUNT(*) FROM " + table))
        print(table + ":", result.scalar())
    
    result = conn.execute(text("SELECT COUNT(*) FROM survey_runs WHERE project_id IS NOT NULL OR road_id IS NOT NULL OR organisation_id IS NOT NULL"))
    print("survey_runs with refs:", result.scalar())
    
    result = conn.execute(text("SELECT COUNT(*) FROM organisation_memberships"))
    print("organisation_memberships:", result.scalar())