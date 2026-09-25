# app/schemas/payloads.py
from pydantic import BaseModel, Field, model_validator


class CameraIntrinsics(BaseModel):
    fx: float = Field(..., description="Horizontal focal length in pixels")
    fy: float = Field(..., description="Vertical focal length in pixels")
    cx: float = Field(..., description="Optical center X coordinate")
    cy: float = Field(..., description="Optical center Y coordinate")
    distortion_coefficients: list[float] = Field(default_factory=list, description="Lens distortion coefficients [k1, k2, p1, p2, k3, ...]")


class CameraCalibrationMetadata(BaseModel):
    camera_height_meters: float = Field(..., description="Mounting height above pavement", gt=0)
    camera_tilt_degrees: float = Field(..., description="Vertical pitch angle from horizon")
    intrinsics: CameraIntrinsics | None = Field(None, description="Camera intrinsic parameters")
    already_undistorted: bool = Field(False, description="True when the source video/images have already been undistorted")

    @model_validator(mode="before")
    @classmethod
    def normalize_flat_intrinsics(cls, values):
        if isinstance(values, dict) and values.get("intrinsics") is None:
            flat_fields = ("fx", "fy", "cx", "cy")
            if all(field in values for field in flat_fields):
                values = dict(values)
                values["intrinsics"] = {
                    field: values[field]
                    for field in (*flat_fields, "distortion_coefficients")
                    if field in values
                }
        return values
