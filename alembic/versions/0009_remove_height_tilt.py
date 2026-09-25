"""Remove camera_height_meters and camera_tilt_degrees from calibration_profiles

These parameters come from the payload (CameraCalibrationMetadata), not the database.
Only intrinsics (fx, fy, cx, cy) and distortion_coefficients are stored.

Revision ID: 0009_remove_height_tilt
Revises: 0008_restore_camera_intrinsics
Create Date: 2026-09-17
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '0009_remove_height_tilt'
down_revision = '0008_restore_camera_intrinsics'
branch_labels = None
depends_on = None


def upgrade():
    # Drop camera_height_meters and camera_tilt_degrees from calibration_profiles
    with op.batch_alter_table('calibration_profiles') as batch_op:
        batch_op.drop_column('camera_height_meters')
        batch_op.drop_column('camera_tilt_degrees')


def downgrade():
    # Re-add camera_height_meters and camera_tilt_degrees
    with op.batch_alter_table('calibration_profiles') as batch_op:
        batch_op.add_column(sa.Column('camera_height_meters', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('camera_tilt_degrees', sa.Float(), nullable=True))