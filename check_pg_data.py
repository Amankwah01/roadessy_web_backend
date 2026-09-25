import psycopg2
import sys

try:
    conn = psycopg2.connect(
        host="127.0.0.1",
        port=5000,
        dbname="roadessy",
        user="postgres",
        password="#PassLetter#"
    )
    cur = conn.cursor()
    
    print("=== PostgreSQL Connected ===")
    
    # Check tables
    cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public' ORDER BY table_name")
    tables = [r[0] for r in cur.fetchall()]
    print("\nTables:", tables)
    
    # Check survey_runs columns
    cur.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_name='survey_runs' ORDER BY ordinal_position")
    print("\n=== survey_runs columns ===")
    for r in cur.fetchall(): print(r)
    
    # Check iri_segments columns
    cur.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_name='iri_segments' ORDER BY ordinal_position")
    print("\n=== iri_segments columns ===")
    for r in cur.fetchall(): print(r)
    
    # Check data counts
    cur.execute("SELECT COUNT(*) FROM survey_runs")
    print("\n=== Data counts ===")
    print("survey_runs:", cur.fetchone())
    cur.execute("SELECT COUNT(*) FROM iri_segments")
    print("iri_segments:", cur.fetchone())
    cur.execute("SELECT COUNT(*) FROM road_centerlines")
    print("road_centerlines:", cur.fetchone())
    cur.execute("SELECT COUNT(*) FROM gh_roads")
    print("gh_roads:", cur.fetchone())
    cur.execute("SELECT COUNT(*) FROM tracked_defects")
    print("tracked_defects:", cur.fetchone())
    
    # Check survey_runs data
    cur.execute("SELECT run_id, name, computed_iri, final_pci, astm_pci, astm_qcm_iri, simple_iri, roadessy_pci FROM survey_runs")
    print("\n=== survey_runs data ===")
    for r in cur.fetchall(): print(r)
    
    # Check iri_segments data
    cur.execute("SELECT id, run_id, segment_start_m, segment_end_m, iri_value, simple_iri, astm_qcm_iri, astm_pci_segment, roadessy_pci_segment, classification FROM iri_segments LIMIT 10")
    print("\n=== iri_segments data ===")
    for r in cur.fetchall(): print(r)
    
    # Check migration version
    cur.execute("SELECT version_num FROM alembic_version")
    print("\n=== Alembic version ===")
    for r in cur.fetchall(): print(r)
    
    cur.close()
    conn.close()
    
except Exception as e:
    print("Error:", e)
    sys.exit(1)
