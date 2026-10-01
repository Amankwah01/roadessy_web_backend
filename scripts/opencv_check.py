"""Simple runtime check for OpenCV features required by the project."""

import importlib
import sys


def main() -> int:
    try:
        cv2 = importlib.import_module("cv2")
    except Exception as exc:
        print("ERROR: OpenCV (cv2) is not installed:", exc)
        return 2

    ok = True
    orb_ctor = getattr(cv2, "ORB_create", None)
    if orb_ctor is None:
        orb_class = getattr(cv2, "ORB", None)
        if orb_class is None or not hasattr(orb_class, "create"):
            print("WARNING: ORB feature detector not available in this OpenCV build.")
            print("         Install 'opencv-contrib-python' or a build with nonfree/contrib modules.")
            ok = False

    print("OpenCV detected. ORB available:", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
