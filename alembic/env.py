"""Alembic environment, configured to use the application's SQLAlchemy engine."""
from logging.config import fileConfig
import os
import sys

from alembic import context

# ensure app package is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from app.database.session import engine as app_engine, Base, IS_POSTGRES

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging (best-effort).
try:
    if config.config_file_name is not None:
        fileConfig(config.config_file_name)
except Exception:
    # If logging sections are not present or malformed, continue without failing
    pass

# Add your model's MetaData object here for 'autogenerate' support
target_metadata = Base.metadata


def run_migrations_online():
    connectable = app_engine

    with connectable.connect() as connection:
        if IS_POSTGRES:
            # Geometry migrations must not run before PostGIS exists.
            connection.exec_driver_sql('CREATE EXTENSION IF NOT EXISTS "postgis"')
            connection.exec_driver_sql('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
            connection.commit()
        context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    raise RuntimeError("Offline mode is not supported by this env.py; run migrations online")
else:
    run_migrations_online()
