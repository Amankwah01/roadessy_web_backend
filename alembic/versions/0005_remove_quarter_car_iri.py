"""Remove legacy quarter_car_iri column from survey_runs.

The quarter_car_iri column was used for the legacy Track 2 QCM IRI.
It is now superseded by astm_qcm_iri which is part of the ASTM Pipeline 1.
"""
from alembic import op
import sqlalchemy as sa

revision = "0005_remove_quarter_car_iri"
down_revision = "0004_add_quarter_car_iri"
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "survey_runs" not in inspector.get_table_names():
        return
    existing = {c["name"] for c in inspector.get_columns("survey_runs")}
    with op.batch_alter_table("survey_runs") as batch_op:
        if "quarter_car_iri" in existing:
            batch_op.drop_column("quarter_car_iri")


def downgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "survey_runs" not in inspector.get_table_names():
        return
    existing = {c["name"] for c in inspector.get_columns("survey_runs")}
    with op.batch_alter_table("survey_runs") as batch_op:
        if "quarter_car_iri" not in existing:
            batch_op.add_column(
                sa.Column("quarter_car_iri", sa.Float(), nullable=True)
            )
