# app/schemas/calibration.py
from pydantic import BaseModel, Field


class CalibrationProfileResponse(BaseModel):
    id: int
    user_id: str
    device_name: str
    checkerboard_inner_corners: dict | None = None
    square_size_meters: float | None = None
    fx: float | None = None
    fy: float | None = None
    cx: float | None = None
    cy: float | None = None
    distortion_coefficients: list[float] | None = None
    camera_height_meters: float | None = None
    camera_tilt_degrees: float | None = None
    reprojection_error: float | None = None
    used_images: int | None = None
    uploaded_images: int | None = None
    created_at: str | None = None
    updated_at: str | None = None

    model_config = {"from_attributes": True}


class CalibrationProfileSaveRequest(BaseModel):
    user_id: str = Field(..., description="Unique user identifier")
    device_name: str = Field(..., description="Human-readable device name")
    metadata: dict = Field(..., description="Camera calibration metadata")
    summary: dict | None = Field(None, description="Calibration summary with reprojection error and image counts")


class CalibrationProfileListResponse(BaseModel):
    status: str
    profiles: list[CalibrationProfileResponse]
