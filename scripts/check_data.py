import sqlite3
import sys

db_path = r"F:\Adubofour\roadessy_mphil\roadessy_web_backend\roadessy.db"
conn = sqlite3.connect(db_path)
c = conn.cursor()

c.execute("SELECT run_id, name, computed_iri, final_pci, astm_pci, astm_qcm_iri, simple_iri, roadessy_pci, matched_road_id FROM survey_runs")
print("=== survey_runs data ===")
for r in c.fetchall():
    print(r)

c.execute("SELECT id, run_id, road_id, segment_start_m, segment_end_m, iri_value, simple_iri, astm_qcm_iri, astm_pci_segment, roadessy_pci_segment, classification FROM iri_segments LIMIT 10")
print("\n=== iri_segments data ===")
for r in c.fetchall():
    print(r)

c.execute("SELECT road_id, road_name, road_width_meters FROM road_centerlines")
print("\n=== road_centerlines ===")
for r in c.fetchall():
    print(r)

conn.close()
