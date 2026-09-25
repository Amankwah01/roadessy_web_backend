from app.services.motion_stability_service import MotionStabilityService


def test_constant_history():
    svc = MotionStabilityService()
    for _ in range(10):
        svc.add_motion_score(5.0)
    # constant values => std=0 => stability -> 1.0
    assert svc.get_stability() == 1.0


def test_varying_history():
    svc = MotionStabilityService()
    values = [0.0, 1.0, 2.0, 3.0, 4.0]
    for v in values:
        svc.add_motion_score(v)
    s = svc.get_stability()
    assert 0.0 < s < 1.0
