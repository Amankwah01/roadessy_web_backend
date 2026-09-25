from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database.session import Base
from app.api.routes.calibration import list_calibration_devices, save_calibration_profile
from app.schemas.calibration import CalibrationProfileSaveRequest


def test_save_nested_calibration_response_populates_profile():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()

    request = CalibrationProfileSaveRequest(
        user_id="user-1",
        device_name="front-camera",
        metadata={
            "uploaded_images": 5,
            "used_images": 4,
            "camera_metadata": {
                "camera_height_meters": 1.6,
                "camera_tilt_degrees": 5.0,
                "checkerboard_inner_corners": {"x": 7, "y": 5},
                "square_size_meters": 0.025,
                "intrinsics": {
                    "fx": 800.0,
                    "fy": 801.0,
                    "cx": 320.0,
                    "cy": 240.0,
                    "distortion_coefficients": [0.1, -0.02, 0.0, 0.0, 0.0],
                },
            },
            "calibration_summary": {"reprojection_error": 0.42},
        },
    )

    assert save_calibration_profile(request, session)["status"] == "success"

    response = list_calibration_devices("user-1", session)
    profile = response["profiles"][0]
    assert profile["square_size_meters"] == 0.025
    assert profile["camera_height_meters"] == 1.6
    assert profile["camera_tilt_degrees"] == 5.0
    assert profile["fx"] == 800.0
    assert profile["fy"] == 801.0
    assert profile["cx"] == 320.0
    assert profile["cy"] == 240.0
    assert profile["distortion_coefficients"] == [0.1, -0.02, 0.0, 0.0, 0.0]
    assert profile["reprojection_error"] == 0.42
    assert profile["used_images"] == 4
    assert profile["uploaded_images"] == 5
