"""remove_road_centerlines

Revision ID: 0010_remove_road_centerlines
Revises: 0009_remove_height_tilt
Create Date: 2026-09-23
"""
from alembic import op
import sqlalchemy as sa

revision = "0010_remove_road_centerlines"
down_revision = "0009_remove_height_tilt"
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "road_centerlines" in inspector.get_table_names():
        op.drop_table("road_centerlines")


def downgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "road_centerlines" not in inspector.get_table_names():
        op.create_table(
            "road_centerlines",
            sa.Column("road_id", sa.String(), primary_key=True, nullable=False),
            sa.Column("user_id", sa.String(), nullable=False),
            sa.Column("road_name", sa.String()),
            sa.Column("geom", sa.Text()),
            sa.Column("road_width_meters", sa.Float()),
            sa.Column("buffer_geom", sa.Text()),
            sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()")),
        )
        op.create_index("ix_road_centerlines_user_id", "road_centerlines", ["user_id"])
