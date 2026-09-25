# app/models/survey.py
import datetime
import uuid

from geoalchemy2 import Geometry
from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, Text, JSON, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID

from app.database.session import Base, DATABASE_URL


def _geometry_column(geometry_type: str, srid: int):
    if DATABASE_URL.startswith("sqlite"):
        return Column(String)
    return Column(Geometry(geometry_type=geometry_type, srid=srid))


class SurveyRun(Base):
    __tablename__ = "survey_runs"

    run_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    user_id = Column(String, nullable=False, index=True)
    organisation_id = Column(String, nullable=False)
    
    # Track 1: ASTM
    astm_pci = Column(Float, nullable=True)
    astm_qcm_iri = Column(Float, nullable=True)
    astm_quality_flags = Column(JSON, nullable=True)
    astm_upsampling_info = Column(JSON, nullable=True)
    
    # Track 2: Roadessy (Simple IRI + Ghana-aligned PCI)
    simple_iri = Column(Float, nullable=True)
    
    # Track 3: Roadessy
    roadessy_pci = Column(Float, nullable=True)
    
    # Legacy
    computed_iri = Column(Float, nullable=True)
    final_pci = Column(Float, nullable=True)
    
    matched_road_id = Column(String, nullable=True, index=True)
    geom_trajectory = _geometry_column("LINESTRING", 4326)
    workspace_dir = Column(String)
    vibration_file_path = Column(String)
    video_file_path = Column(String)
    gps_file_path = Column(String)
    model_path = Column(String)
    processing_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class MapMatchingRun(Base):
    __tablename__ = "map_matching_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    user_id = Column(String, nullable=False, index=True)
    matched_road_id = Column(String, nullable=True, index=True)
    match_status = Column(String, nullable=True)
    match_method = Column(String, nullable=True)
    centerline_wkt = Column(Text, nullable=True)
    centerline_geojson = Column(JSON, nullable=True)
    snapped_trajectory_wkt = Column(Text, nullable=True)
    chainage_min_m = Column(Float, nullable=True)
    chainage_max_m = Column(Float, nullable=True)
    matched_distance_m = Column(Float, nullable=True)
    road_length_m = Column(Float, nullable=True)
    max_distance_m = Column(Float, nullable=True)
    matched_points = Column(Integer, nullable=True)
    total_points = Column(Integer, nullable=True)
    unmatched_points = Column(Integer, nullable=True)
    snapped_coords = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    __table_args__ = (UniqueConstraint("run_id", name="uq_map_matching_runs_run_id"),)


class IRISegment(Base):
    __tablename__ = "iri_segments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    user_id = Column(String, nullable=False, index=True)
    road_id = Column(String, nullable=True, index=True)
    segment_start_m = Column(Float, nullable=False)
    segment_end_m = Column(Float, nullable=False)
    iri_value = Column(Float, nullable=False)
    
    # Track-specific IRI values
    simple_iri = Column(Float, nullable=True)
    astm_qcm_iri = Column(Float, nullable=True)
    
    # Track-specific PCI values
    astm_pci_segment = Column(Float, nullable=True)
    roadessy_pci_segment = Column(Float, nullable=True)
    
    classification = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class TrackedDefect(Base):
    __tablename__ = "tracked_defects"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(UUID(as_uuid=True), nullable=False)
    user_id = Column(String, nullable=False, index=True)
    road_id = Column(String, nullable=True, index=True)
    track_id = Column(Integer)
    failure_mode = Column(String)
    max_area_m2 = Column(Float)
    max_width_mm = Column(Float)
    max_depth_mm = Column(Float)
    mean_depth_mm = Column(Float)
    depth_confidence = Column(Float)
    depth_method = Column(String)
    depth_message = Column(Text)
    deduct_value = Column(Float)
    image_path = Column(String, nullable=True)
    
    # Track-specific fields
    astm_distress_type = Column(String, nullable=True)
    ghana_distress_type = Column(String, nullable=True)
    ghana_rating_1to9 = Column(Integer, nullable=True)
    yolo_class_id = Column(Integer, nullable=True)
    confidence = Column(Float, nullable=True)
    chainage_m = Column(Float, nullable=True)
    geom_location = _geometry_column("POINT", 4326)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
