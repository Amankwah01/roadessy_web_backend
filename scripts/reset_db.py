from sqlalchemy import create_engine, text
from urllib.parse import quote_plus
import os

DB_USER = os.getenv('DB_USER', 'postgres')
DB_PASSWORD = os.getenv('DB_PASSWORD', '#PassLetter#')
DB_HOST = os.getenv('DB_HOST', '127.0.0.1')
DB_PORT = os.getenv('DB_PORT', '5000')
DB_NAME = os.getenv('DB_NAME', 'roadessy')

encoded_password = quote_plus(DB_PASSWORD)
DATABASE_URL = 'postgresql+psycopg2://{}:{}@{}:{}/{}'.format(DB_USER, encoded_password, DB_HOST, DB_PORT, DB_NAME)

engine = create_engine(DATABASE_URL)

with engine.connect() as conn:
    conn.execute(text('DROP SCHEMA public CASCADE'))
    conn.execute(text('CREATE SCHEMA public'))
    conn.execute(text('GRANT ALL ON SCHEMA public TO postgres'))
    conn.execute(text('GRANT ALL ON SCHEMA public TO public'))
    conn.execute(text('CREATE EXTENSION IF NOT EXISTS postgis'))
    conn.execute(text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"'))
    conn.commit()
    print('Database schema reset complete!')