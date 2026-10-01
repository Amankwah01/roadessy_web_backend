import subprocess
import sys

# Check if PostgreSQL is running
try:
    result = subprocess.run(
        ["pg_isready", "-h", "127.0.0.1", "-p", "5000"],
        capture_output=True, text=True, timeout=5
    )
    print("PostgreSQL 5000:", result.stdout.strip() or result.stderr.strip() or "OK", "RC:", result.returncode)
except Exception as e:
    print("PostgreSQL 5000 check failed:", e)

try:
    result = subprocess.run(
        ["pg_isready", "-h", "localhost", "-p", "5432"],
        capture_output=True, text=True, timeout=5
    )
    print("PostgreSQL 5432:", result.stdout.strip() or result.stderr.strip() or "OK", "RC:", result.returncode)
except Exception as e:
    print("PostgreSQL 5432 check failed:", e)

# Check psql
try:
    result = subprocess.run(
        ["psql", "-h", "127.0.0.1", "-p", "5000", "-U", "postgres", "-d", "roadessy", "-c", "SELECT 1"],
        capture_output=True, text=True, timeout=10,
        env={**__import__('os').environ, "PGPASSWORD": "#PassLetter#"}
    )
    print("psql connection:", result.stdout.strip() or "OK")
    if result.stderr:
        print("psql stderr:", result.stderr.strip())
except Exception as e:
    print("psql check failed:", e)
