import sqlite3
import os

db_path = r"F:\Adubofour\roadessy_mphil\roadessy_web_backend\roadessy.db"
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT sql FROM sqlite_master WHERE type='table'")
    for r in c.fetchall(): print(r[0])
    print()
    c.execute("SELECT COUNT(*) FROM survey_runs")
    print("survey_runs count:", c.fetchone())
    c.execute("SELECT COUNT(*) FROM iri_segments")
    print("iri_segments count:", c.fetchone())
    c.execute("SELECT COUNT(*) FROM road_centerlines")
    print("road_centerlines count:", c.fetchone())
    conn.close()
else:
    print("DB not found:", db_path)
