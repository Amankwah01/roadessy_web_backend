# app/models/calibration.py
import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import declarative_base

from app.database.session import Base


class CalibrationProfile(Base):
    __tablename__ = "calibration_profiles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, nullable=False, index=True)
    device_name = Column(String, nullable=False)
    checkerboard_inner_corners = Column(Text)
    square_size_meters = Column(Float)
    fx = Column(Float)
    fy = Column(Float)
    cx = Column(Float)
    cy = Column(Float)
    distortion_coefficients = Column(Text)
    reprojection_error = Column(Float)
    used_images = Column(Integer)
    uploaded_images = Column(Integer)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
