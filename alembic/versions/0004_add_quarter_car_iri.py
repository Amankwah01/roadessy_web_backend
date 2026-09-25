"""Add quarter_car_iri column to survey_runs for IRI comparison

Revision ID: 0004_add_quarter_car_iri
Revises: 0003_add_calibration_mounting_parameters
"""
from alembic import op
import sqlalchemy as sa

revision = "0004_add_quarter_car_iri"
down_revision = "0003_add_calibration_mounting_parameters"
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "survey_runs" not in inspector.get_table_names():
        return
    existing = {c["name"] for c in inspector.get_columns("survey_runs")}
    with op.batch_alter_table("survey_runs") as batch_op:
        if "quarter_car_iri" not in existing:
            batch_op.add_column(sa.Column("quarter_car_iri", sa.Float(), nullable=True))


def downgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "survey_runs" not in inspector.get_table_names():
        return
    existing = {c["name"] for c in inspector.get_columns("survey_runs")}
    with op.batch_alter_table("survey_runs") as batch_op:
        if "quarter_car_iri" in existing:
            batch_op.drop_column("quarter_car_iri")