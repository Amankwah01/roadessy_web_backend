from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models.calibration import CalibrationProfile
from app.api.routes.calibration import list_calibration_devices, save_calibration_profile
from app.schemas.calibration import CalibrationProfileSaveRequest


def _session():
    """A session over just the calibration table.

    `Base.metadata.create_all` would build every model in the project, and
    several of those carry PostGIS geometry columns. Creating those on SQLite
    fails on the `RecoverGeometryColumn` spatial function, which SQLite has no
    such thing for — so the whole fixture dies before the test runs, for a
    reason that has nothing to do with calibration. Building only the table this
    test touches keeps the failure where it belongs.
    """
    engine = create_engine("sqlite:///:memory:")
    CalibrationProfile.__table__.create(bind=engine)
    return sessionmaker(bind=engine)()


def test_save_nested_calibration_response_populates_profile():
    session = _session()

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

    # No principal, so the handler falls back to the submitted user id — the
    # same path an unauthenticated development request takes.
    assert save_calibration_profile(request, session, {})["status"] == "success"

    response = list_calibration_devices("user-1", session, {})
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


def test_verified_principal_overrides_submitted_user_id():
    """A verified token decides whose profile this is, not the request body.

    Without this, one signed-in user could write a calibration under another
    user's id, and the read path — which filters on the verified id — would
    leave the profile invisible to the person who saved it.
    """
    session = _session()

    request = CalibrationProfileSaveRequest(
        user_id="someone-else",
        device_name="front-camera",
        metadata={"camera_metadata": {"square_size_meters": 0.025}},
    )
    assert save_calibration_profile(request, session, {"user_id": "real-user"})["status"] == "success"

    # Stored against the verified identity, not the one in the body.
    assert list_calibration_devices("someone-else", session, {"user_id": "real-user"})["profiles"]
    assert list_calibration_devices("real-user", session, {"user_id": "real-user"})["profiles"]
    # And not readable as though it belonged to the submitted id.
    assert not list_calibration_devices("someone-else", session, {})["profiles"]


def test_saving_twice_updates_rather_than_duplicates():
    session = _session()

    for square_size in (0.025, 0.05):
        save_calibration_profile(
            CalibrationProfileSaveRequest(
                user_id="user-1",
                device_name="front-camera",
                metadata={"camera_metadata": {"square_size_meters": square_size}},
            ),
            session,
            {},
        )

    profiles = list_calibration_devices("user-1", session, {})["profiles"]
    assert len(profiles) == 1
    assert profiles[0]["square_size_meters"] == 0.05
