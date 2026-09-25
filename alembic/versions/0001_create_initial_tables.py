"""Create initial survey and tracked defect tables

Revision ID: 0001_initial
Revises: 
Create Date: 2026-06-02 00:00:00.000000
"""
from alembic import op
import sqlalchemy as sa
from geoalchemy2 import Geometry

# revision identifiers, used by Alembic.
revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "survey_runs",
        sa.Column("run_id", sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("organisation_id", sa.String(), nullable=False),
        sa.Column("computed_iri", sa.Float()),
        sa.Column("final_pci", sa.Float()),
        sa.Column("matched_road_id", sa.String()),
        sa.Column("geom_trajectory", Geometry(geometry_type="LINESTRING", srid=4326)),
        sa.Column("workspace_dir", sa.String()),
        sa.Column("vibration_file_path", sa.String()),
        sa.Column("video_file_path", sa.String()),
        sa.Column("gps_file_path", sa.String()),
        sa.Column("model_path", sa.String()),
        sa.Column("processing_metadata", sa.Text()),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()")),
    )

    op.create_table(
        "tracked_defects",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("run_id", sa.dialects.postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("track_id", sa.Integer()),
        sa.Column("failure_mode", sa.String()),
        sa.Column("max_area_m2", sa.Float()),
        sa.Column("max_width_mm", sa.Float()),
        sa.Column("max_depth_mm", sa.Float()),
        sa.Column("mean_depth_mm", sa.Float()),
        sa.Column("depth_confidence", sa.Float()),
        sa.Column("depth_method", sa.String()),
        sa.Column("depth_message", sa.Text()),
        sa.Column("deduct_value", sa.Float()),
        sa.Column("geom_location", Geometry(geometry_type="POINT", srid=4326)),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()")),
    )


def downgrade():
    op.drop_table("tracked_defects")
    op.drop_table("survey_runs")
