"""Add Roadessy PCI columns, rename legacy Ghana PCI, and ensure map matching schema.

Revision ID: 0006_roadessy_pci_map
Revises: 0005_remove_quarter_car_iri
Create Date: 2026-09-17
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy import text

# revision identifiers, used by Alembic.
revision = "0006_roadessy_pci_map"
down_revision = "0005_remove_quarter_car_iri"
branch_labels = None
depends_on = None


def _columns(inspector, table_name):
    """Return set of column names for table."""
    try:
        return {c["name"] for c in inspector.get_columns(table_name)}
    except Exception:
        return set()


def _indexes(inspector, table_name):
    """Return set of index names for table."""
    try:
        return {idx["name"] for idx in inspector.get_indexes(table_name)}
    except Exception:
        return set()


def _unique_constraints(inspector, table_name):
    """Return set of unique constraint names for table."""
    try:
        return {uc["name"] for uc in inspector.get_unique_constraints(table_name)}
    except Exception:
        return set()


def _drop_null_column(bind, table, column, inspector):
    """Drop column only if all values are NULL; otherwise raise RuntimeError."""
    existing = _columns(inspector, table)
    if column not in existing:
        return
    # Check non-null count
    cnt = bind.execute(text(f'SELECT COUNT(*) FROM "{table}" WHERE "{column}" IS NOT NULL')).scalar()
    if cnt and cnt > 0:
        raise RuntimeError(
            f"Cannot drop column {table}.{column}: {cnt} non-null values present"
        )
    with op.batch_alter_table(table) as batch_op:
        batch_op.drop_column(column)


def _add_missing_columns(table, columns_def, inspector):
    """Add columns that are missing."""
    existing = _columns(inspector, table)
    with op.batch_alter_table(table) as batch_op:
        for col_name, col_type, kwargs in columns_def:
            if col_name not in existing:
                batch_op.add_column(sa.Column(col_name, col_type, **kwargs))


def _rename_column_if_needed(bind, table, old, new, inspector):
    """Rename old to new if old exists and new does not.
    If both exist, ensure old is null then drop old; if old has data and new is null, copy then drop old.
    If both have data, raise RuntimeError.
    """
    existing = _columns(inspector, table)
    if old not in existing:
        return
    if new not in existing:
        # Simple rename
        with op.batch_alter_table(table) as batch_op:
            batch_op.alter_column(old, new_column_name=new)
        return
    # Both exist - check data
    cnt_old = bind.execute(text(f'SELECT COUNT(*) FROM "{table}" WHERE "{old}" IS NOT NULL')).scalar()
    cnt_new = bind.execute(text(f'SELECT COUNT(*) FROM "{table}" WHERE "{new}" IS NOT NULL')).scalar()
    if cnt_old and cnt_old > 0:
        if cnt_new and cnt_new > 0:
            raise RuntimeError(
                f"Both {old} and {new} have non-null data in {table}; cannot resolve"
            )
        # new is null, copy old to new
        bind.execute(text(f'UPDATE "{table}" SET "{new}" = "{old}" WHERE "{new}" IS NULL AND "{old}" IS NOT NULL'))
    # Now old can be dropped (all null or copied)
    _drop_null_column(bind, table, old, sa.inspect(bind))


def _handle_legacy_rename(bind, table, old, new, inspector):
    """Handle renaming legacy column to new column.
    If new already exists (from SQLAlchemy model), copy data from old and drop old.
    """
    existing = _columns(inspector, table)
    if old not in existing:
        return
    if new not in existing:
        # Simple rename
        with op.batch_alter_table(table) as batch_op:
            batch_op.alter_column(old, new_column_name=new)
        return
    # Both exist - copy data from old to new where new is null, then drop old
    cnt_old = bind.execute(text(f'SELECT COUNT(*) FROM "{table}" WHERE "{old}" IS NOT NULL')).scalar()
    cnt_new = bind.execute(text(f'SELECT COUNT(*) FROM "{table}" WHERE "{new}" IS NOT NULL')).scalar()
    if cnt_old and cnt_old > 0:
        if cnt_new == 0:
            # Copy data from old to new
            bind.execute(text(f'UPDATE "{table}" SET "{new}" = "{old}" WHERE "{new}" IS NULL AND "{old}" IS NOT NULL'))
        # If both have data, keep new (from model), drop old
    # Drop old column
    _drop_null_column(bind, table, old, sa.inspect(bind))


def upgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    # ----- survey_runs -----
    survey_cols = _columns(inspector, "survey_runs")

    # Add missing columns for Track 1 (ASTM) and Track 2 (Roadessy)
    # Note: roadessy_pci may already exist from SQLAlchemy model creation
    survey_new_cols = [
        ("astm_pci", sa.Float(), {"nullable": True}),
        ("astm_qcm_iri", sa.Float(), {"nullable": True}),
        ("astm_quality_flags", sa.JSON(), {"nullable": True}),
        ("astm_upsampling_info", sa.JSON(), {"nullable": True}),
        ("simple_iri", sa.Float(), {"nullable": True}),
    ]
    _add_missing_columns("survey_runs", survey_new_cols, inspector)

    # Handle ghana_pci -> roadessy_pci rename (legacy to new)
    if "ghana_pci" in survey_cols:
        _handle_legacy_rename(bind, "survey_runs", "ghana_pci", "roadessy_pci", inspector)

    # Drop legacy columns only if null
    legacy_survey = ["rppci", "pci_qcm", "qcm_iri", "quarter_car_iri"]
    for col in legacy_survey:
        _drop_null_column(bind, "survey_runs", col, inspector)

    # Ensure processing_metadata is JSON (if it's Text, convert)
    # We check the column type via information_schema
    try:
        col_type = bind.execute(
            text("""
                SELECT data_type FROM information_schema.columns
                WHERE table_name = 'survey_runs' AND column_name = 'processing_metadata'
            """)
        ).scalar()
        if col_type and col_type.lower() == "text":
            # Convert Text to JSON using casting; assume valid JSON
            with op.batch_alter_table("survey_runs") as batch_op:
                batch_op.alter_column(
                    "processing_metadata",
                    type_=sa.JSON(),
                    postgresql_using="processing_metadata::json"
                )
    except Exception:
        # If anything goes wrong, skip to avoid breaking migration
        pass

    # ----- iri_segments -----
    segment_cols = _columns(inspector, "iri_segments")

    segment_new_cols = [
        ("simple_iri", sa.Float(), {"nullable": True}),
        ("astm_qcm_iri", sa.Float(), {"nullable": True}),
        ("astm_pci_segment", sa.Float(), {"nullable": True}),
    ]
    _add_missing_columns("iri_segments", segment_new_cols, inspector)

    # Handle ghana_pci_segment -> roadessy_pci_segment rename (legacy to new)
    if "ghana_pci_segment" in segment_cols:
        _handle_legacy_rename(bind, "iri_segments", "ghana_pci_segment", "roadessy_pci_segment", inspector)

    # Drop legacy segment columns if null
    legacy_segment = ["rppci_segment", "pci_qcm_segment", "qcm_iri"]
    for col in legacy_segment:
        _drop_null_column(bind, "iri_segments", col, inspector)

    # ----- map_matching_runs -----
    map_match_cols = _columns(inspector, "map_matching_runs")
    if not map_match_cols:
        # Create table
        op.create_table(
            "map_matching_runs",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("run_id", UUID(as_uuid=True), nullable=False, index=True),
            sa.Column("user_id", sa.String(), nullable=False, index=True),
            sa.Column("matched_road_id", sa.String(), nullable=True, index=True),
            sa.Column("match_status", sa.String(), nullable=True),
            sa.Column("match_method", sa.String(), nullable=True),
            sa.Column("centerline_wkt", sa.Text(), nullable=True),
            sa.Column("centerline_geojson", sa.JSON(), nullable=True),
            sa.Column("snapped_trajectory_wkt", sa.Text(), nullable=True),
            sa.Column("chainage_min_m", sa.Float(), nullable=True),
            sa.Column("chainage_max_m", sa.Float(), nullable=True),
            sa.Column("matched_distance_m", sa.Float(), nullable=True),
            sa.Column("road_length_m", sa.Float(), nullable=True),
            sa.Column("max_distance_m", sa.Float(), nullable=True),
            sa.Column("matched_points", sa.Integer(), nullable=True),
            sa.Column("total_points", sa.Integer(), nullable=True),
            sa.Column("unmatched_points", sa.Integer(), nullable=True),
            sa.Column("snapped_coords", sa.JSON(), nullable=True),
            sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()")),
            sa.UniqueConstraint("run_id", name="uq_map_matching_runs_run_id"),
        )
        # Indexes are created via index=True on columns; unique constraint added.
    else:
        # Table exists, add missing columns
        map_new_cols = [
            ("run_id", UUID(as_uuid=True), {"nullable": False, "index": True}),
            ("user_id", sa.String(), {"nullable": False, "index": True}),
            ("matched_road_id", sa.String(), {"nullable": True, "index": True}),
            ("match_status", sa.String(), {"nullable": True}),
            ("match_method", sa.String(), {"nullable": True}),
            ("centerline_wkt", sa.Text(), {"nullable": True}),
            ("centerline_geojson", sa.JSON(), {"nullable": True}),
            ("snapped_trajectory_wkt", sa.Text(), {"nullable": True}),
            ("chainage_min_m", sa.Float(), {"nullable": True}),
            ("chainage_max_m", sa.Float(), {"nullable": True}),
            ("matched_distance_m", sa.Float(), {"nullable": True}),
            ("road_length_m", sa.Float(), {"nullable": True}),
            ("max_distance_m", sa.Float(), {"nullable": True}),
            ("matched_points", sa.Integer(), {"nullable": True}),
            ("total_points", sa.Integer(), {"nullable": True}),
            ("unmatched_points", sa.Integer(), {"nullable": True}),
            ("snapped_coords", sa.JSON(), {"nullable": True}),
            ("created_at", sa.DateTime(), {"nullable": True, "server_default": sa.text("now()")}),
        ]
        _add_missing_columns("map_matching_runs", map_new_cols, inspector)

        # Ensure unique constraint on run_id
        unique_constraints = _unique_constraints(inspector, "map_matching_runs")
        if "uq_map_matching_runs_run_id" not in unique_constraints:
            op.create_unique_constraint(
                "uq_map_matching_runs_run_id", "map_matching_runs", ["run_id"]
            )

        # Ensure indexes
        indexes = _indexes(inspector, "map_matching_runs")
        for col in ["run_id", "user_id", "matched_road_id"]:
            idx_name = f"ix_map_matching_runs_{col}"
            if idx_name not in indexes:
                op.create_index(idx_name, "map_matching_runs", [col])


def downgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    # ----- survey_runs -----
    survey_cols = _columns(inspector, "survey_runs")

    # Rename roadessy_pci back to ghana_pci if roadessy exists and ghana absent
    if "roadessy_pci" in survey_cols:
        _rename_column_if_needed(bind, "survey_runs", "roadessy_pci", "ghana_pci", inspector)

    # Re-add legacy columns if they were dropped (nullable)
    legacy_survey = [
        ("rppci", sa.Float(), {"nullable": True}),
        ("pci_qcm", sa.Float(), {"nullable": True}),
        ("qcm_iri", sa.Float(), {"nullable": True}),
        ("quarter_car_iri", sa.Float(), {"nullable": True}),
    ]
    with op.batch_alter_table("survey_runs") as batch_op:
        for col_name, col_type, kwargs in legacy_survey:
            if col_name not in survey_cols:
                batch_op.add_column(sa.Column(col_name, col_type, **kwargs))

    # Remove the Track 1/2 columns we added (if they are null)
    added_survey = ["astm_pci", "astm_qcm_iri", "astm_quality_flags", "astm_upsampling_info", "simple_iri", "roadessy_pci"]
    for col in added_survey:
        _drop_null_column(bind, "survey_runs", col, inspector)

    # Revert processing_metadata to Text if it was converted
    try:
        col_type = bind.execute(
            text("""
                SELECT data_type FROM information_schema.columns
                WHERE table_name = 'survey_runs' AND column_name = 'processing_metadata'
            """)
        ).scalar()
        if col_type and col_type.lower() == "json":
            with op.batch_alter_table("survey_runs") as batch_op:
                batch_op.alter_column(
                    "processing_metadata",
                    type_=sa.Text(),
                    postgresql_using="processing_metadata::text"
                )
    except Exception:
        pass

    # ----- iri_segments -----
    segment_cols = _columns(inspector, "iri_segments")

    # Rename roadessy_pci_segment back to ghana_pci_segment
    if "roadessy_pci_segment" in segment_cols:
        _rename_column_if_needed(bind, "iri_segments", "roadessy_pci_segment", "ghana_pci_segment", inspector)

    # Re-add legacy segment columns
    legacy_segment = [
        ("rppci_segment", sa.Float(), {"nullable": True}),
        ("pci_qcm_segment", sa.Float(), {"nullable": True}),
        ("qcm_iri", sa.Float(), {"nullable": True}),
    ]
    with op.batch_alter_table("iri_segments") as batch_op:
        for col_name, col_type, kwargs in legacy_segment:
            if col_name not in segment_cols:
                batch_op.add_column(sa.Column(col_name, col_type, **kwargs))

    # Drop added segment columns if null
    added_segment = ["simple_iri", "astm_qcm_iri", "astm_pci_segment", "roadessy_pci_segment"]
    for col in added_segment:
        _drop_null_column(bind, "iri_segments", col, inspector)

    # ----- map_matching_runs -----
    # We don't drop the table because it might have been created by Base.metadata.create_all
    # Instead, we drop the unique constraint and indexes we added, and drop columns we added if null.
    if "map_matching_runs" in inspector.get_table_names():
        # Drop unique constraint if we added it
        unique_constraints = _unique_constraints(inspector, "map_matching_runs")
        if "uq_map_matching_runs_run_id" in unique_constraints:
            op.drop_constraint("uq_map_matching_runs_run_id", "map_matching_runs", type_="unique")

        # Drop indexes we added
        indexes = _indexes(inspector, "map_matching_runs")
        for col in ["run_id", "user_id", "matched_road_id"]:
            idx_name = f"ix_map_matching_runs_{col}"
            if idx_name in indexes:
                op.drop_index(idx_name, table_name="map_matching_runs")

        # Drop columns we added if null
        map_cols = _columns(inspector, "map_matching_runs")
        added_map = [
            "run_id", "user_id", "matched_road_id", "match_status", "match_method",
            "centerline_wkt", "centerline_geojson", "snapped_trajectory_wkt",
            "chainage_min_m", "chainage_max_m", "matched_distance_m",
            "road_length_m", "max_distance_m", "matched_points",
            "total_points", "unmatched_points", "snapped_coords", "created_at"
        ]
        for col in added_map:
            if col in map_cols:
                _drop_null_column(bind, "map_matching_runs", col, inspector)