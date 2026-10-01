import os
import csv
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus

DB_USER = os.getenv('DB_USER', 'postgres')
DB_PASSWORD = os.getenv('DB_PASSWORD', '#PassLetter#')
DB_HOST = os.getenv('DB_HOST', '127.0.0.1')
DB_PORT = os.getenv('DB_PORT', '5000')
DB_NAME = os.getenv('DB_NAME', 'roadessy')

encoded_password = quote_plus(DB_PASSWORD)
DATABASE_URL = 'postgresql+psycopg2://{}:{}@{}:{}/{}'.format(DB_USER, encoded_password, DB_HOST, DB_PORT, DB_NAME)

engine = create_engine(DATABASE_URL)

backup_dir = 'db_backup'

# Create gh_roads table with PostGIS geometry
with engine.connect() as conn:
    print('Creating gh_roads table...')
    conn.execute(text('''
        CREATE TABLE IF NOT EXISTS gh_roads (
            gid INTEGER,
            ogc_fid INTEGER,
            osm_id VARCHAR,
            code VARCHAR,
            fclass VARCHAR,
            name VARCHAR,
            ref VARCHAR,
            oneway VARCHAR,
            maxspeed VARCHAR,
            layer VARCHAR,
            bridge VARCHAR,
            tunnel VARCHAR,
            district VARCHAR,
            region VARCHAR,
            geom GEOMETRY(MULTILINESTRING, 4326)
        )
    '''))
    conn.execute(text('CREATE INDEX IF NOT EXISTS gh_roads_geom_idx ON gh_roads USING GIST (geom)'))
    conn.commit()
    print('  gh_roads table created')

# Create topology and layer tables (PostGIS topology)
with engine.connect() as conn:
    print('Creating topology tables...')
    conn.execute(text('''
        CREATE TABLE IF NOT EXISTS topology.topology (
            id SERIAL PRIMARY KEY,
            name VARCHAR NOT NULL UNIQUE,
            srid INTEGER NOT NULL,
            precision DOUBLE PRECISION NOT NULL,
            hasz BOOLEAN NOT NULL DEFAULT false
        )
    '''))
    conn.execute(text('''
        CREATE TABLE IF NOT EXISTS topology.layer (
            topology_id INTEGER NOT NULL REFERENCES topology.topology(id),
            layer_id INTEGER NOT NULL,
            feature_type INTEGER NOT NULL,
            level INTEGER NOT NULL DEFAULT 0,
            child_id INTEGER,
            PRIMARY KEY (topology_id, layer_id)
        )
    '''))
    conn.commit()
    print('  topology tables created')

print('Tables created, now restoring data...')

# Restore gh_roads
csv_path = os.path.join(backup_dir, 'gh_roads.csv')
if os.path.exists(csv_path):
    print(f'Restoring gh_roads from {csv_path}...')
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        columns = next(reader)
        rows = list(reader)
    
    placeholders = ', '.join([':' + col for col in columns])
    columns_str = ', '.join(['"' + col + '"' for col in columns])
    insert_sql = 'INSERT INTO gh_roads ({}) VALUES ({})'.format(columns_str, placeholders)
    
    with engine.connect() as conn:
        for i, row in enumerate(rows):
            row_dict = dict(zip(columns, row))
            for k, v in row_dict.items():
                if v == '':
                    row_dict[k] = None
            conn.execute(text(insert_sql), row_dict)
            
            if i % 50000 == 0 and i > 0:
                conn.commit()
                print(f'  Committed {i} rows...')
        conn.commit()
        print(f'  -> {len(rows)} rows restored for gh_roads')

# Restore calibration_profiles
csv_path = os.path.join(backup_dir, 'calibration_profiles.csv')
if os.path.exists(csv_path):
    print(f'Restoring calibration_profiles from {csv_path}...')
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        columns = next(reader)
        rows = list(reader)
    
    if rows:
        placeholders = ', '.join([':' + col for col in columns])
        columns_str = ', '.join(['"' + col + '"' for col in columns])
        insert_sql = 'INSERT INTO calibration_profiles ({}) VALUES ({})'.format(columns_str, placeholders)
        
        with engine.connect() as conn:
            for i, row in enumerate(rows):
                row_dict = dict(zip(columns, row))
                for k, v in row_dict.items():
                    if v == '':
                        row_dict[k] = None
                conn.execute(text(insert_sql), row_dict)
            conn.commit()
            print(f'  -> {len(rows)} rows restored for calibration_profiles')

# topology and layer typically have no data (they were empty)
print('Restore complete!')