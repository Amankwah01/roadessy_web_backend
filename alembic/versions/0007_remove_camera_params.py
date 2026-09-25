"""Remove camera parameters from calibration_profiles

These parameters are now provided via the payload (CameraCalibrationMetadata)
instead of being stored in the calibration profile.

Revision ID: 0007_remove_camera_params
Revises: 0006_roadessy_pci_map
Create Date: 2026-09-17
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '0007_remove_camera_params'
down_revision = '0006_roadessy_pci_map'
branch_labels = None
depends_on = None


def upgrade():
    # Drop camera parameter columns from calibration_profiles
    with op.batch_alter_table('calibration_profiles') as batch_op:
        batch_op.drop_column('camera_height_meters')
        batch_op.drop_column('camera_tilt_degrees')
        batch_op.drop_column('fx')
        batch_op.drop_column('fy')
        batch_op.drop_column('cx')
        batch_op.drop_column('cy')


def downgrade():
    # Re-add camera parameter columns
    with op.batch_alter_table('calibration_profiles') as batch_op:
        batch_op.add_column(sa.Column('camera_height_meters', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('camera_tilt_degrees', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('fx', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('fy', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('cx', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('cy', sa.Float(), nullable=True))