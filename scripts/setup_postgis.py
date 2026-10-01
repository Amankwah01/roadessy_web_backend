"""Create PostGIS extension and spatial indexes for the Roadessy DB.

Run as:

    python -m tools.setup_postgis

This will attempt to enable PostGIS on the connected database and create GIST
indexes for `tracked_defects.geom_location` and `survey_runs.geom_trajectory`.
"""

from sqlalchemy import text
from app.database.session import engine


def main() -> int:
    try:
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis;"))
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS idx_tracked_defects_geom_location ON tracked_defects USING GIST (geom_location);"
                )
            )
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS idx_survey_runs_geom_trajectory ON survey_runs USING GIST (geom_trajectory);"
                )
            )
            print("PostGIS extension and spatial indexes ensured.")
        return 0
    except Exception as exc:
        print("Failed to ensure PostGIS or create indexes:", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
