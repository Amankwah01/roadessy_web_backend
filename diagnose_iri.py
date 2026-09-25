import psycopg2
import sys

conn = psycopg2.connect(
    host="127.0.0.1", port=5000, dbname="roadessy",
    user="postgres", password="#PassLetter#"
)
cur = conn.cursor()

print("=== Run IRI/PCI distribution ===")
cur.execute("""
    SELECT 
        run_id,
        name,
        computed_iri,
        final_pci,
        astm_pci,
        astm_qcm_iri,
        simple_iri,
        roadessy_pci,
        CASE 
            WHEN simple_iri > 100 THEN 'BAD_IRI (>100)'
            WHEN simple_iri < 0 THEN 'BAD_IRI (<0)'
            ELSE 'OK'
        END as iri_status
    FROM survey_runs
    ORDER BY simple_iri DESC
    LIMIT 20
""")
print("Runs with highest simple_iri (BAD):")
for r in cur.fetchall(): print(r)

print("\n=== Runs with good IRI (< 20) ===")
cur.execute("""
    SELECT run_id, name, computed_iri, final_pci, astm_pci, astm_qcm_iri, simple_iri, roadessy_pci
    FROM survey_runs
    WHERE simple_iri <= 20
    ORDER BY simple_iri DESC
""")
for r in cur.fetchall(): print(r)

print("\n=== IRI segment stats ===")
cur.execute("""
    SELECT 
        COUNT(*) as total,
        MIN(iri_value) as min_iri,
        MAX(iri_value) as max_iri,
        AVG(iri_value) as avg_iri,
        COUNT(CASE WHEN iri_value > 100 THEN 1 END) as bad_iri_count,
        COUNT(CASE WHEN simple_iri > 100 THEN 1 END) as bad_simple_iri_count,
        COUNT(CASE WHEN astm_qcm_iri > 10 THEN 1 END) as bad_astm_iri_count
    FROM iri_segments
""")
print(cur.fetchone())

print("\n=== Segment distribution by run ===")
cur.execute("""
    SELECT run_id, COUNT(*) as seg_count,
        MIN(iri_value) as min_iri, MAX(iri_value) as max_iri,
        MIN(simple_iri) as min_simple, MAX(simple_iri) as max_simple,
        MIN(astm_qcm_iri) as min_astm, MAX(astm_qcm_iri) as max_astm,
        MIN(astm_pci_segment) as min_astm_pci, MAX(astm_pci_segment) as max_astm_pci,
        MIN(roadessy_pci_segment) as min_road_pci, MAX(roadessy_pci_segment) as max_road_pci
    FROM iri_segments
    GROUP BY run_id
    ORDER BY max_simple DESC
    LIMIT 15
""")
for r in cur.fetchall(): print(r)

print("\n=== Runs with PCI < 100 ===")
cur.execute("""
    SELECT run_id, name, final_pci, astm_pci, roadessy_pci, simple_iri, astm_qcm_iri
    FROM survey_runs
    WHERE final_pci < 100 OR astm_pci < 100 OR roadessy_pci < 100
    ORDER BY final_pci DESC
""")
for r in cur.fetchall(): print(r)

print("\n=== Defects per run ===")
cur.execute("""
    SELECT sr.run_id, sr.name, COUNT(td.id) as defect_count,
        SUM(td.deduct_value) as total_dv
    FROM survey_runs sr
    LEFT JOIN tracked_defects td ON sr.run_id = td.run_id
    WHERE sr.final_pci < 100
    GROUP BY sr.run_id, sr.name
    ORDER BY sr.final_pci DESC
    LIMIT 10
""")
print("Runs with PCI < 100 and their defects:")
for r in cur.fetchall(): print(r)

print("\n=== Defect stats ===")
cur.execute("SELECT COUNT(*) FROM tracked_defects")
print("Total defects:", cur.fetchone())
cur.execute("SELECT run_id, COUNT(*) FROM tracked_defects GROUP BY run_id HAVING COUNT(*) > 0 ORDER BY COUNT(*) DESC LIMIT 10")
print("Defects per run:")
for r in cur.fetchall(): print(r)

cur.close()
conn.close()
