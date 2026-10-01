import sqlite3
import os

db_path = r"F:\Adubofour\roadessy_mphil\roadessy_web_backend\roadessy.db"
conn = sqlite3.connect(db_path)
c = conn.cursor()

c.execute("PRAGMA table_info(survey_runs)")
print("=== survey_runs columns ===")
for r in c.fetchall(): print(r)

c.execute("PRAGMA table_info(iri_segments)")
print("\n=== iri_segments columns ===")
for r in c.fetchall(): print(r)

c.execute("SELECT name FROM sqlite_master WHERE type='table'")
print("\n=== All tables ===")
for r in c.fetchall(): print(r)

c.execute("SELECT COUNT(*) FROM survey_runs")
print("\n=== survey_runs count ===", c.fetchone())

c.execute("SELECT COUNT(*) FROM iri_segments")
print("=== iri_segments count ===", c.fetchone())

c.execute("SELECT COUNT(*) FROM road_centerlines")
print("=== road_centerlines count ===", c.fetchone())

c.execute("SELECT COUNT(*) FROM gh_roads")
print("=== gh_roads count ===", c.fetchone())

c.execute("SELECT run_id, name, computed_iri, final_pci FROM survey_runs")
print("\n=== survey_runs data ===")
for r in c.fetchall(): print(r)

conn.close()
