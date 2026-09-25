"""Add road_centerlines, iri_segments tables and extend tracked_defects

Revision ID: 0002_add_road_centerlines_and_iri_segments
Revises: 0001_initial
Create Date: 2026-08-04 00:00:00.000000
"""
from alembic import op
import sqlalchemy as sa
from geoalchemy2 import Geometry

# revision identifiers, used by Alembic.
revision = "0002_add_road_centerlines_and_iri_segments"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if "road_centerlines" not in inspector.get_table_names():
        op.create_table(
            "road_centerlines",
            sa.Column("road_id", sa.String(), primary_key=True, nullable=False),
            sa.Column("user_id", sa.String(), nullable=False),
            sa.Column("road_name", sa.String()),
            sa.Column("geom", Geometry(geometry_type="LINESTRING", srid=4326)),
            sa.Column("road_width_meters", sa.Float()),
            sa.Column("buffer_geom", Geometry(geometry_type="POLYGON", srid=4326)),
            sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()")),
        )
        op.create_index("ix_road_centerlines_user_id", "road_centerlines", ["user_id"])

    if "iri_segments" not in inspector.get_table_names():
        op.create_table(
            "iri_segments",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("run_id", sa.dialects.postgresql.UUID(as_uuid=True), nullable=False),
            sa.Column("user_id", sa.String(), nullable=False),
            sa.Column("road_id", sa.String(), nullable=True),
            sa.Column("segment_start_m", sa.Float(), nullable=False),
            sa.Column("segment_end_m", sa.Float(), nullable=False),
            sa.Column("iri_value", sa.Float(), nullable=False),
            sa.Column("classification", sa.String()),
            sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()")),
        )
        op.create_index("ix_iri_segments_run_id", "iri_segments", ["run_id"])
        op.create_index("ix_iri_segments_user_id", "iri_segments", ["user_id"])
        op.create_index("ix_iri_segments_road_id", "iri_segments", ["road_id"])

    tracked_columns = [c["name"] for c in inspector.get_columns("tracked_defects")]
    with op.batch_alter_table("tracked_defects", schema=None) as batch_op:
        if "user_id" not in tracked_columns:
            batch_op.add_column(sa.Column("user_id", sa.String(), nullable=False))
            batch_op.create_index("ix_tracked_defects_user_id", ["user_id"])
        if "road_id" not in tracked_columns:
            batch_op.add_column(sa.Column("road_id", sa.String(), nullable=True))
            batch_op.create_index("ix_tracked_defects_road_id", ["road_id"])


def downgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    tracked_columns = [c["name"] for c in inspector.get_columns("tracked_defects")]
    with op.batch_alter_table("tracked_defects", schema=None) as batch_op:
        if "road_id" in tracked_columns:
            batch_op.drop_index("ix_tracked_defects_road_id")
            batch_op.drop_column("road_id")
        if "user_id" in tracked_columns:
            batch_op.drop_index("ix_tracked_defects_user_id")
            batch_op.drop_column("user_id")

    if "iri_segments" in inspector.get_table_names():
        op.drop_index("ix_iri_segments_road_id", table_name="iri_segments")
        op.drop_index("ix_iri_segments_user_id", table_name="iri_segments")
        op.drop_index("ix_iri_segments_run_id", table_name="iri_segments")
        op.drop_table("iri_segments")

    if "road_centerlines" in inspector.get_table_names():
        op.drop_index("ix_road_centerlines_user_id", table_name="road_centerlines")
        op.drop_table("road_centerlines")
