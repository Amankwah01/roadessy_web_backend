import json
from typing import Any, cast
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.models.calibration import CalibrationProfile
from app.schemas.calibration import (
    CalibrationProfileSaveRequest,
    CalibrationProfileResponse,
    CalibrationProfileListResponse,
)

router = APIRouter(prefix="/calibration", tags=["calibration"])


def _parse_json_field(raw: Any) -> Any:
    if raw is None:
        return None
    if isinstance(raw, (dict, list)):
        return raw
    try:
        return json.loads(str(raw))
    except (ValueError, TypeError):
        return None


def _format_datetime(dt: Any) -> str | None:
    dt_val = cast(datetime | None, dt)
    return dt_val.isoformat() if dt_val is not None else None


@router.post("/save", response_model=dict)
def save_calibration_profile(
    request: CalibrationProfileSaveRequest,
    db: Session = Depends(get_db),
):
    try:
        profile = (
            db.query(CalibrationProfile)
            .filter(
                CalibrationProfile.user_id == request.user_id,
                CalibrationProfile.device_name == request.device_name,
            )
            .first()
        )

        raw_metadata = request.metadata or {}
        metadata = raw_metadata.get("camera_metadata", raw_metadata)
        summary = request.summary or raw_metadata.get("calibration_summary") or {}
        intrinsics = metadata.get("intrinsics") or {}

        if profile is None:
            profile = CalibrationProfile(
                user_id=request.user_id,
                device_name=request.device_name,
            )
            db.add(profile)

        checkerboard_inner_corners = metadata.get("checkerboard_inner_corners")
        if checkerboard_inner_corners is not None:
            setattr(
                profile,
                "checkerboard_inner_corners",
                json.dumps(checkerboard_inner_corners),
            )

        values = {
            "square_size_meters": metadata.get("square_size_meters"),
            "fx": metadata.get("fx", intrinsics.get("fx")),
            "fy": metadata.get("fy", intrinsics.get("fy")),
            "cx": metadata.get("cx", intrinsics.get("cx")),
            "cy": metadata.get("cy", intrinsics.get("cy")),
            "reprojection_error": summary.get("reprojection_error"),
            "used_images": summary.get("used_images", raw_metadata.get("used_images")),
            "uploaded_images": summary.get("uploaded_images", raw_metadata.get("uploaded_images")),
        }
        for field, value in values.items():
            if value is not None:
                setattr(profile, field, value)

        distortion_coefficients = metadata.get(
            "distortion_coefficients", intrinsics.get("distortion_coefficients")
        )
        if distortion_coefficients is not None:
            setattr(
                profile,
                "distortion_coefficients",
                json.dumps(distortion_coefficients),
            )

        db.commit()
        db.refresh(profile)

        return {"status": "success", "profile_id": getattr(profile, "id")}
    except HTTPException:
        raise
    except Exception as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save calibration profile: {str(exc)}",
        )


@router.get("/devices/{user_id}", response_model=CalibrationProfileListResponse)
def list_calibration_devices(user_id: str, db: Session = Depends(get_db)):
    try:
        profiles = (
            db.query(CalibrationProfile)
            .filter(CalibrationProfile.user_id == user_id)
            .order_by(CalibrationProfile.updated_at.desc())
            .all()
        )

        results = []
        for p in profiles:
            results.append(
                {
                    "id": getattr(p, "id"),
                    "user_id": getattr(p, "user_id"),
                    "device_name": getattr(p, "device_name"),
                    "checkerboard_inner_corners": _parse_json_field(
                        getattr(p, "checkerboard_inner_corners", None)
                    ),
                    "square_size_meters": getattr(p, "square_size_meters", None),
                    "fx": getattr(p, "fx", None),
                    "fy": getattr(p, "fy", None),
                    "cx": getattr(p, "cx", None),
                    "cy": getattr(p, "cy", None),
                    "distortion_coefficients": _parse_json_field(
                        getattr(p, "distortion_coefficients", None)
                    ),
                    "reprojection_error": getattr(p, "reprojection_error", None),
                    "used_images": getattr(p, "used_images", None),
                    "uploaded_images": getattr(p, "uploaded_images", None),
                    "created_at": _format_datetime(getattr(p, "created_at", None)),
                    "updated_at": _format_datetime(getattr(p, "updated_at", None)),
                }
            )

        return {"status": "success", "profiles": results}
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list calibration devices: {str(exc)}",
        )


@router.get("/{user_id}/{device_name}", response_model=dict)
def get_calibration_profile(
    user_id: str,
    device_name: str,
    db: Session = Depends(get_db),
):
    try:
        profile = (
            db.query(CalibrationProfile)
            .filter(
                CalibrationProfile.user_id == user_id,
                CalibrationProfile.device_name == device_name,
            )
            .first()
        )

        if profile is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Calibration profile not found",
            )

        response = {
            "status": "success",
            "profile": {
                "id": getattr(profile, "id"),
                "user_id": getattr(profile, "user_id"),
                "device_name": getattr(profile, "device_name"),
                "camera_height_meters": getattr(profile, "camera_height_meters", None),
                "camera_tilt_degrees": getattr(profile, "camera_tilt_degrees", None),
                "checkerboard_inner_corners": _parse_json_field(
                    getattr(profile, "checkerboard_inner_corners", None)
                ),
                "square_size_meters": getattr(profile, "square_size_meters", None),
                "fx": getattr(profile, "fx", None),
                "fy": getattr(profile, "fy", None),
                "cx": getattr(profile, "cx", None),
                "cy": getattr(profile, "cy", None),
                "distortion_coefficients": _parse_json_field(
                    getattr(profile, "distortion_coefficients", None)
                ),
                "reprojection_error": getattr(profile, "reprojection_error", None),
                "used_images": getattr(profile, "used_images", None),
                "uploaded_images": getattr(profile, "uploaded_images", None),
                "created_at": _format_datetime(getattr(profile, "created_at", None)),
                "updated_at": _format_datetime(getattr(profile, "updated_at", None)),
            },
        }

        return response
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch calibration profile: {str(exc)}",
        )