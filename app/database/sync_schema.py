"""Create any tables the models declare that do not exist yet, and add any
columns that were added to a model after its table was first created.

There is no migration tool in this project. `Base.metadata.create_all` handles
new tables, and the column loop handles the other case -- a column added to an
existing model -- so that neither needs a hand-written migration to be applied.

Safe to run repeatedly: every statement is guarded by a check first.
"""

import logging

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

from app.database.session import Base

# Imported for their side effect of registering the tables on Base.metadata.
# Collected into a tuple so the dependency is explicit rather than looking
# like an unused import to a linter.
import app.models.admin  # noqa: F401
import app.models.calibration  # noqa: F401
import app.models.organisation  # noqa: F401
import app.models.survey  # noqa: F401

_REGISTERED_MODELS = (app.models.survey, app.models.calibration, app.models.organisation, app.models.admin)

logger = logging.getLogger(__name__)


def sync_schema(engine: Engine) -> None:
    # Touching the tuple is what makes the model imports above load; without it
    # they would be removed as unused and the tables would stop being created.
    assert _REGISTERED_MODELS is not None

    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())

    Base.metadata.create_all(engine, checkfirst=True)

    created = set(inspector.get_table_names()) - existing_tables
    if created:
        logger.info("Created tables: %s", ", ".join(sorted(created)))

    # create_all never alters a table that already exists, so a column added to
    # a model afterwards has to be added here or every read of it fails.
    for table in Base.metadata.sorted_tables:
        if table.name not in existing_tables:
            # Newly created above, so it already has every column.
            continue
        present = {column["name"] for column in inspector.get_columns(table.name)}
        for column in table.columns:
            if column.name in present:
                continue
            ddl = f'ALTER TABLE {table.name} ADD COLUMN "{column.name}" {column.type.compile(dialect=engine.dialect)}'
            logger.info("Adding column %s.%s", table.name, column.name)
            with engine.begin() as connection:
                connection.execute(text(ddl))
