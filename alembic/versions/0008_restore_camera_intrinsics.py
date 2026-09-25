"""Restore camera intrinsics to calibration_profiles

The camera intrinsics (fx, fy, cx, cy) are stored in the database
and used for undistortion. Camera height and tilt come from the payload.

Revision ID: 0008_restore_camera_intrinsics
Revises: 0007_remove_camera_params
Create Date: 2026-09-17
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '0008_restore_camera_intrinsics'
down_revision = '0007_remove_camera_params'
branch_labels = None
depends_on = None


def upgrade():
    # Add camera intrinsics columns back to calibration_profiles
    with op.batch_alter_table('calibration_profiles') as batch_op:
        batch_op.add_column(sa.Column('fx', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('fy', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('cx', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('cy', sa.Float(), nullable=True))
        # Also restore height/tilt for completeness
        batch_op.add_column(sa.Column('camera_height_meters', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('camera_tilt_degrees', sa.Float(), nullable=True))


def downgrade():
    # Drop camera parameter columns from calibration_profiles
    with op.batch_alter_table('calibration_profiles') as batch_op:
        batch_op.drop_column('fx')
        batch_op.drop_column('fy')
        batch_op.drop_column('cx')
        batch_op.drop_column('cy')
        batch_op.drop_column('camera_height_meters')
        batch_op.drop_column('camera_tilt_degrees')