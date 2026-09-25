OpenCV requirements

This repository relies on OpenCV with the ORB feature detector and other contrib modules.

Recommended package (CPU builds):

- opencv-contrib-python==4.13.0.92

If you install a headless environment, you may choose:

- opencv-contrib-python-headless==4.13.0.92

Runtime check

A simple runtime probe is available at `tools/opencv_check.py` which prints whether ORB is available in the current OpenCV build. Run:

```bash
python -m tools.opencv_check
```

If the check reports ORB is unavailable, install the contrib package:

```bash
pip install opencv-contrib-python==4.13.0.92
```
