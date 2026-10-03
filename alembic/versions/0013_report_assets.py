"""add report assets

Revision ID: 0013_report_assets
Revises: 0012_defect_road_station
Create Date: 2026-10-03
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

revision = "0013_report_assets"
down_revision = "0012_defect_road_station"
branch_labels = None
depends_on = None


def upgrade():
    # A generated report is a file on disk plus this row. The row is the part
    # that is queryable: the dashboard lists an organization's reports newest
    # first, and a filesystem walk cannot answer which organization asked for a
    # PDF or which year it covers.
    op.create_table(
        "report_assets",
        sa.Column("id", PG_UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("organisation_id", sa.String(), nullable=False),
        sa.Column("year", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(), nullable=True),
        sa.Column("kind", sa.String(), nullable=False, server_default="organisation"),
        sa.Column("run_id", sa.String(), nullable=True),
        sa.Column("file_name", sa.String(), nullable=False),
        sa.Column("original_filename", sa.String(), nullable=True),
        sa.Column("content_type", sa.String(), nullable=False, server_default="application/pdf"),
        sa.Column("size_bytes", sa.Integer(), nullable=True),
        sa.Column("sha256", sa.String(), nullable=True),
        sa.Column("storage_path", sa.Text(), nullable=False),
        sa.Column("bucket", sa.String(), nullable=False, server_default="reports"),
        sa.Column("key", sa.String(), nullable=True),
        sa.Column("created_by", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_report_assets_organisation_id", "report_assets", ["organisation_id"])
    op.create_index("ix_report_assets_year", "report_assets", ["year"])
    op.create_index("ix_report_assets_run_id", "report_assets", ["run_id"])
    op.create_index("ix_report_assets_created_by", "report_assets", ["created_by"])
    op.create_index("ix_report_assets_created_at", "report_assets", ["created_at"])
    # The one listing query the dashboard issues: this organization's reports,
    # newest first, optionally narrowed to a year.
    op.create_index(
        "ix_report_org_year_created",
        "report_assets",
        ["organisation_id", "year", "created_at"],
    )


def downgrade():
    op.drop_index("ix_report_org_year_created", table_name="report_assets")
    op.drop_index("ix_report_assets_created_at", table_name="report_assets")
    op.drop_index("ix_report_assets_created_by", table_name="report_assets")
    op.drop_index("ix_report_assets_run_id", table_name="report_assets")
    op.drop_index("ix_report_assets_year", table_name="report_assets")
    op.drop_index("ix_report_assets_organisation_id", table_name="report_assets")
    op.drop_table("report_assets")