import pandas as pd

from tools.yolo_defect_detector import align_detection_to_csv, infer_time_series


def test_infer_time_series_prefers_numeric_column():
    df = pd.DataFrame({"time_s": [0.0, 2.0, 4.0], "vibration": [1.0, 3.0, 5.0]})

    result = infer_time_series(df, "time_s")

    assert result.tolist() == [0.0, 2.0, 4.0]


def test_align_detection_to_csv_returns_nearest_row():
    df = pd.DataFrame({"time_s": [0.0, 2.0, 4.0], "vibration": [1.0, 2.0, 3.0]})
    time_series = infer_time_series(df, "time_s")

    row_index = align_detection_to_csv(3.0, df, time_series)

    assert row_index == 2
