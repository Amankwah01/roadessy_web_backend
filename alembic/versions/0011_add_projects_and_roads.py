"""add projects and roads

Revision ID: 0011_add_projects_and_roads
Revises: 0010_remove_road_centerlines
Create Date: 2026-09-29
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0011_add_projects_and_roads"
down_revision = "0010_remove_road_centerlines"
branch_labels = None
depends_on = None


def upgrade():
    # Create projects table
    op.create_table(
        "projects",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("org_id", sa.String(), nullable=False),
        sa.Column("org_name", sa.String(), nullable=False),
        sa.Column("visibility", sa.String(), default="PERSONAL"),
        sa.Column("team_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("team_name", sa.String(), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("created_by", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("now()")),
    )
    op.create_index("ix_projects_org_id", "projects", ["org_id"])
    op.create_index("ix_projects_team_id", "projects", ["team_id"])

    # Create roads table
    op.create_table(
        "roads",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("project_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("project_name", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("start_node", sa.String(), nullable=True),
        sa.Column("end_node", sa.String(), nullable=True),
        sa.Column("surface_type", sa.String(), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("road_geom", sa.Text(), nullable=True),  # Geometry column
        sa.Column("created_by", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("now()")),
    )
    op.create_index("ix_roads_project_id", "roads", ["project_id"])
    op.create_unique_constraint("uq_road_name_per_project", "roads", ["project_id", "name"])

    # Add project_id and road_id columns to survey_runs
    op.add_column("survey_runs", sa.Column("project_id", postgresql.UUID(as_uuid=True), nullable=True))
    op.add_column("survey_runs", sa.Column("road_id", postgresql.UUID(as_uuid=True), nullable=True))
    op.create_index("ix_survey_runs_project_id", "survey_runs", ["project_id"])
    op.create_index("ix_survey_runs_road_id", "survey_runs", ["road_id"])
    op.create_foreign_key("fk_survey_runs_project_id", "survey_runs", "projects", ["project_id"], ["id"])
    op.create_foreign_key("fk_survey_runs_road_id", "survey_runs", "roads", ["road_id"], ["id"])


def downgrade():
    # Remove foreign keys and columns from survey_runs
    op.drop_constraint("fk_survey_runs_road_id", "survey_runs", type_="foreignkey")
    op.drop_constraint("fk_survey_runs_project_id", "survey_runs", type_="foreignkey")
    op.drop_index("ix_survey_runs_road_id", table_name="survey_runs")
    op.drop_index("ix_survey_runs_project_id", table_name="survey_runs")
    op.drop_column("survey_runs", "road_id")
    op.drop_column("survey_runs", "project_id")

    # Drop roads table
    op.drop_constraint("uq_road_name_per_project", "roads", type_="unique")
    op.drop_index("ix_roads_project_id", table_name="roads")
    op.drop_table("roads")

    # Drop projects table
    op.drop_index("ix_projects_team_id", table_name="projects")
    op.drop_index("ix_projects_org_id", table_name="projects")
    op.drop_table("projects")