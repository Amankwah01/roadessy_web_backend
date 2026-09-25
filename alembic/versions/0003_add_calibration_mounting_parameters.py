"""Add camera mounting parameters to calibration profiles

Revision ID: 0003_add_calibration_mounting_parameters
Revises: 0002_add_road_centerlines_and_iri_segments
"""
from alembic import op
import sqlalchemy as sa

revision = "0003_add_calibration_mounting_parameters"
down_revision = "0002_add_road_centerlines_and_iri_segments"
branch_labels = None
depends_on = None

EXPECTED_COLUMNS = {
    "camera_height_meters": sa.Float(),
    "camera_tilt_degrees": sa.Float(),
    "checkerboard_inner_corners": sa.Text(),
    "square_size_meters": sa.Float(),
    "fx": sa.Float(),
    "fy": sa.Float(),
    "cx": sa.Float(),
    "cy": sa.Float(),
    "distortion_coefficients": sa.Text(),
    "reprojection_error": sa.Float(),
    "used_images": sa.Integer(),
    "uploaded_images": sa.Integer(),
}


def upgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = inspector.get_table_names()

    if "calibration_profiles" not in tables:
        op.create_table(
            "calibration_profiles",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("user_id", sa.String(), nullable=False),
            sa.Column("device_name", sa.String(), nullable=False),
            sa.Column("camera_height_meters", sa.Float()),
            sa.Column("camera_tilt_degrees", sa.Float()),
            sa.Column("checkerboard_inner_corners", sa.Text()),
            sa.Column("square_size_meters", sa.Float()),
            sa.Column("fx", sa.Float()),
            sa.Column("fy", sa.Float()),
            sa.Column("cx", sa.Float()),
            sa.Column("cy", sa.Float()),
            sa.Column("distortion_coefficients", sa.Text()),
            sa.Column("reprojection_error", sa.Float()),
            sa.Column("used_images", sa.Integer()),
            sa.Column("uploaded_images", sa.Integer()),
            sa.Column("created_at", sa.DateTime()),
            sa.Column("updated_at", sa.DateTime()),
        )
        op.create_index("ix_calibration_profiles_user_id", "calibration_profiles", ["user_id"])
        return

    existing = {column["name"] for column in inspector.get_columns("calibration_profiles")}
    with op.batch_alter_table("calibration_profiles") as batch_op:
        for column_name, column_type in EXPECTED_COLUMNS.items():
            if column_name not in existing:
                batch_op.add_column(sa.Column(column_name, column_type))


def downgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "calibration_profiles" not in inspector.get_table_names():
        return
    existing = {column["name"] for column in inspector.get_columns("calibration_profiles")}
    with op.batch_alter_table("calibration_profiles") as batch_op:
        if "uploaded_images" in existing:
            batch_op.drop_column("uploaded_images")
        if "used_images" in existing:
            batch_op.drop_column("used_images")
        if "reprojection_error" in existing:
            batch_op.drop_column("reprojection_error")
        if "distortion_coefficients" in existing:
            batch_op.drop_column("distortion_coefficients")
        if "cy" in existing:
            batch_op.drop_column("cy")
        if "cx" in existing:
            batch_op.drop_column("cx")
        if "fy" in existing:
            batch_op.drop_column("fy")
        if "fx" in existing:
            batch_op.drop_column("fx")
        if "square_size_meters" in existing:
            batch_op.drop_column("square_size_meters")
        if "checkerboard_inner_corners" in existing:
            batch_op.drop_column("checkerboard_inner_corners")
        if "camera_tilt_degrees" in existing:
            batch_op.drop_column("camera_tilt_degrees")
        if "camera_height_meters" in existing:
            batch_op.drop_column("camera_height_meters")