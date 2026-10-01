"""add absolute road station to defects

Revision ID: 0012_defect_road_station
Revises: 0011_add_projects_and_roads
Create Date: 2026-09-30
"""
from alembic import op
import sqlalchemy as sa

revision = "0012_defect_road_station"
down_revision = "0011_add_projects_and_roads"
branch_labels = None
depends_on = None


def upgrade():
    # `tracked_defects.chainage_m` is distance since the survey started, so it is
    # zero wherever the vehicle was put on the road and cannot be compared with
    # another survey of the same stretch. `station_m` is the same defect measured
    # from the beginning of the matched road's centerline, which can.
    #
    # `chainage_m` is kept and not replaced: the 50 m PCI table and the IRI
    # segments are cut from it and begin at the start of the run.
    op.add_column("tracked_defects", sa.Column("station_m", sa.Float(), nullable=True))
    op.add_column("tracked_defects", sa.Column("station_road_id", sa.String(), nullable=True))
    op.add_column("tracked_defects", sa.Column("station_source", sa.String(), nullable=True))
    op.add_column("tracked_defects", sa.Column("chainage_offset_m", sa.Float(), nullable=True))
    op.create_index(
        "ix_tracked_defects_station_m", "tracked_defects", ["station_m"], unique=False
    )


def downgrade():
    op.drop_index("ix_tracked_defects_station_m", table_name="tracked_defects")
    op.drop_column("tracked_defects", "chainage_offset_m")
    op.drop_column("tracked_defects", "station_source")
    op.drop_column("tracked_defects", "station_road_id")
    op.drop_column("tracked_defects", "station_m")
