"""Tests for the IoU tracker used by the YOLO video harness.

The tracker is the part of the harness with real logic in it, and the failure
modes that matter are all counting errors: a drifting box counted as a hundred
defects, or two nearby defects welded into one. Those are what these cover.

The detector itself is not exercised here — running YOLO needs weights and a
video, and a test that depends on both is a test that fails for reasons that
have nothing to do with the tracker.
"""

import pytest

from scripts.vision_tracking import IoUTracker, build_tracking_registry, iou, merge_duplicates


def det(x1, y1, x2, y2, cls_id=4, conf=0.5, area=1.0, width=0.5, length=0.5, dv=5.0,
        chainage=0.0, severity=""):
    return {
        "bbox": [x1, y1, x2, y2],
        "yolo_class_id": cls_id,
        "confidence": conf,
        "area": area,
        "width": width,
        "length": length,
        "deduct_value": dv,
        "chainage_m": chainage,
        "severity": severity,
    }


class TestIou:
    def test_identical_boxes_overlap_fully(self):
        assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)

    def test_disjoint_boxes_do_not_overlap(self):
        assert iou([0, 0, 10, 10], [50, 50, 60, 60]) == 0.0

    def test_touching_boxes_do_not_overlap(self):
        """Edge contact has zero area, so it is not an overlap."""
        assert iou([0, 0, 10, 10], [10, 0, 20, 10]) == 0.0

    def test_half_overlap(self):
        # 5x10 = 50 shared, union = 100 + 100 - 50 = 150, so 1/3.
        assert iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1 / 3)

    def test_half_overlap_vertically(self):
        assert iou([0, 0, 10, 10], [0, 5, 10, 15]) == pytest.approx(1 / 3)

    def test_degenerate_zero_area_box(self):
        assert iou([0, 0, 0, 0], [0, 0, 0, 0]) == 0.0


class TestTrackerIdentity:
    def test_first_detection_creates_a_track(self):
        tracker = IoUTracker()
        pairs = tracker.update([det(0, 0, 10, 10)], frame_idx=0)
        assert len(pairs) == 1
        assert pairs[0][0].track_id == 1
        assert pairs[0][0].hits == 1

    def test_a_box_across_frames_is_one_defect_not_many(self):
        """The whole point of tracking: a pothole is not recounted per frame."""
        tracker = IoUTracker(iou_threshold=0.3)
        for frame in range(10):
            tracker.update([det(100 + frame, 100, 110 + frame, 110)], frame_idx=frame)
        tracks = tracker.finish()
        assert len(tracks) == 1
        assert tracks[0].hits == 10
        assert tracks[0].first_frame == 0
        assert tracks[0].last_frame == 9

    def test_a_defect_moving_with_the_camera_stays_one_track(self):
        """Boxes shift as the vehicle moves; IoU has to absorb that."""
        tracker = IoUTracker(iou_threshold=0.3)
        for frame in range(6):
            offset = frame * 5
            tracker.update([det(200 + offset, 200, 260 + offset, 260)], frame_idx=frame)
        assert len(tracker.finish()) == 1

    def test_two_defects_stay_separate(self):
        tracker = IoUTracker(iou_threshold=0.3)
        for frame in range(5):
            tracker.update(
                [det(0, 0, 40, 40), det(500, 500, 540, 540)], frame_idx=frame
            )
        assert len(tracker.finish()) == 2

    def test_ids_are_unique_across_tracks(self):
        tracker = IoUTracker(iou_threshold=0.3)
        for frame in range(4):
            tracker.update(
                [det(0, 0, 40, 40), det(300, 0, 340, 40), det(600, 0, 640, 40)],
                frame_idx=frame,
            )
        ids = [t.track_id for t in tracker.finish()]
        assert len(ids) == 3
        assert len(set(ids)) == 3

    def test_no_detections_keeps_existing_tracks(self):
        tracker = IoUTracker(iou_threshold=0.3)
        tracker.update([det(0, 0, 40, 40)], frame_idx=0)
        assert tracker.update([], frame_idx=1) == []
        tracker.update([det(0, 0, 40, 40)], frame_idx=2)
        assert len(tracker.finish()) == 1


class TestTrackerCounting:
    def test_a_defect_that_leaves_and_comes_back_resumes_its_track(self):
        """A brief occlusion should not split one pothole into two."""
        tracker = IoUTracker(iou_threshold=0.3, max_missed=5)
        tracker.update([det(0, 0, 40, 40)], frame_idx=0)
        for frame in (1, 2):
            tracker.update([], frame_idx=frame)
        tracker.update([det(0, 0, 40, 40)], frame_idx=3)
        assert len(tracker.finish()) == 1

    def test_a_defect_seen_again_too_late_is_a_new_one(self):
        """Past the miss window it is a different defect, not a returning one.

        Checked with merging off: the boxes are identical, so the duplicate
        merge pass would legitimately fold these back into one track. This is
        about the association window, not the merge.
        """
        tracker = IoUTracker(iou_threshold=0.3, max_missed=2)
        tracker.update([det(0, 0, 40, 40)], frame_idx=0)
        for frame in range(1, 8):
            tracker.update([], frame_idx=frame)
        tracker.update([det(0, 0, 40, 40)], frame_idx=8)
        assert len(tracker.finish(merge=False)) == 2

    def test_low_overlap_starts_a_new_track(self):
        """The same place but a different size is not the same detection."""
        tracker = IoUTracker(iou_threshold=0.5)
        tracker.update([det(0, 0, 20, 20)], frame_idx=0)
        tracker.update([det(100, 100, 400, 400)], frame_idx=1)
        assert len(tracker.finish()) == 2

    def test_one_detection_never_claims_two_tracks(self):
        """A single box must not merge two separate tracks into one."""
        tracker = IoUTracker(iou_threshold=0.3)
        tracker.update([det(0, 0, 40, 40)], frame_idx=0)
        tracker.update([det(0, 0, 40, 40)], frame_idx=1)
        # One box overlapping both existing tracks: only one may be extended.
        pairs = tracker.update([det(0, 0, 40, 40)], frame_idx=2)
        assert len(pairs) == 1
        assert len(tracker.finish()) == 1


class TestTrackAggregates:
    def test_best_confidence_is_kept_not_the_last(self):
        tracker = IoUTracker(iou_threshold=0.3)
        tracker.update([det(0, 0, 40, 40, conf=0.9)], frame_idx=0)
        tracker.update([det(0, 0, 40, 40, conf=0.1)], frame_idx=1)
        track = tracker.finish()[0]
        assert track.best_confidence == pytest.approx(0.9)
        assert track.mean_confidence == pytest.approx(0.5)

    def test_best_frame_points_at_the_clears_detection(self):
        """The crop is written from the best frame, so a weak frame never wins."""
        tracker = IoUTracker(iou_threshold=0.3)
        tracker.update([det(0, 0, 40, 40, conf=0.2)], frame_idx=0)
        tracker.update([det(1, 1, 41, 41, conf=0.95)], frame_idx=1)
        tracker.update([det(2, 2, 42, 42, conf=0.3)], frame_idx=2)
        track = tracker.finish()[0]
        assert track.best_frame_idx == 1
        assert track.best_bbox == [1, 1, 41, 41]

    def test_maxima_use_the_worst_reading_not_the_average(self):
        """A defect seen clearly once must not be averaged down to mild."""
        tracker = IoUTracker(iou_threshold=0.3)
        tracker.update([det(0, 0, 40, 40, area=0.1, dv=1.0)], frame_idx=0)
        tracker.update([det(0, 0, 40, 40, area=4.0, dv=20.0)], frame_idx=1)
        track = tracker.finish()[0]
        assert track.max_area == pytest.approx(4.0)
        assert track.max_deduct_value == pytest.approx(20.0)
        assert track.total_deduct_value == pytest.approx(21.0)

    def test_duration_spans_first_to_last_frame(self):
        tracker = IoUTracker(iou_threshold=0.3)
        for frame in (10, 11, 12):
            tracker.update([det(0, 0, 40, 40)], frame_idx=frame)
        assert tracker.finish()[0].duration_frames == 3


class TestMergeDuplicates:
    def test_a_flickering_defect_becomes_one_track(self):
        """A weak detection that drops out and returns is still one defect.

        This is the case frame-to-frame association cannot catch on its own: the
        gap is longer than any usable `max_missed`, so each appearance gets its
        own track and the defect is counted repeatedly.
        """
        tracker = IoUTracker(iou_threshold=0.3, max_missed=2)
        tracker.update([det(0, 0, 40, 40, chainage=0.0)], frame_idx=0)
        for frame in range(1, 10):
            tracker.update([], frame_idx=frame)
        tracker.update([det(0, 0, 40, 40, chainage=1.0)], frame_idx=10)
        assert len(tracker.finish()) == 1

    def test_merge_uses_chainage_when_the_camera_has_moved(self):
        """Same place along the road, different pixels, still one defect."""
        tracker = IoUTracker(iou_threshold=0.3, max_missed=1)
        tracker.update([det(0, 0, 40, 40, chainage=100.0)], frame_idx=0)
        tracker.update([], frame_idx=1)
        tracker.update([det(800, 400, 840, 440, chainage=105.0)], frame_idx=5)
        assert len(tracker.finish()) == 1

    def test_different_places_along_the_road_stay_separate(self):
        """Chainage has to actually separate, or every defect merges into one.

        The boxes are far apart in the image too, so this isolates the
        chainage rule rather than the pixel-overlap fallback.
        """
        tracker = IoUTracker(iou_threshold=0.3, max_missed=1)
        tracker.update([det(0, 0, 40, 40, chainage=0.0)], frame_idx=0)
        tracker.update([], frame_idx=1)
        tracker.update([det(1500, 900, 1540, 940, chainage=900.0)], frame_idx=5)
        assert len(tracker.finish()) == 2

    def test_absent_chainage_does_not_merge_everything(self):
        """With no GPS and no assumed speed, chainage is all zeros.

        Merging on "both at distance 0" would collapse every same-class pair in
        the video into a single track, so unknown position must not be read as
        the same position.
        """
        tracker = IoUTracker(iou_threshold=0.3, max_missed=1)
        tracker.update([det(0, 0, 40, 40, chainage=0.0)], frame_idx=0)
        tracker.update([], frame_idx=1)
        tracker.update([det(900, 700, 940, 740, chainage=0.0)], frame_idx=5)
        assert len(tracker.finish()) == 2

    def test_different_classes_are_never_merged(self):
        """A crack and a pothole at the same spot are two defects."""
        tracker = IoUTracker(iou_threshold=0.3, max_missed=1)
        tracker.update([det(0, 0, 40, 40, cls_id=2, chainage=0.0)], frame_idx=0)
        tracker.update([], frame_idx=1)
        tracker.update([det(0, 0, 40, 40, cls_id=4, chainage=1.0)], frame_idx=5)
        assert len(tracker.finish()) == 2

    def test_two_defects_present_in_the_same_frame_are_never_merged(self):
        """Overlapping boxes in one frame are simultaneous, so both are real.

        This is what leftover NMS duplicates look like, and merging them here
        would hide a genuine double detection rather than remove a repeat. The
        two arrive in the same `update`, which is what "same frame" means.
        """
        tracker = IoUTracker(iou_threshold=0.3, max_missed=5)
        pairs = tracker.update(
            [det(0, 0, 40, 40, chainage=0.0), det(2, 2, 42, 42, chainage=0.5)], frame_idx=0
        )
        assert len(pairs) == 2
        assert len(tracker.finish()) == 2

    def test_merging_keeps_the_strongest_reading(self):
        """A defect seen clearly once is not averaged down by the faint frames."""
        tracker = IoUTracker(iou_threshold=0.3, max_missed=1)
        tracker.update([det(0, 0, 40, 40, conf=0.2, area=0.5, dv=3.0, chainage=0.0)], frame_idx=0)
        tracker.update([], frame_idx=1)
        tracker.update([det(0, 0, 40, 40, conf=0.9, area=4.0, dv=18.0, chainage=1.0)], frame_idx=3)
        merged = tracker.finish()[0]
        assert merged.hits == 2
        assert merged.best_confidence == pytest.approx(0.9)
        assert merged.max_area == pytest.approx(4.0)
        assert merged.max_deduct_value == pytest.approx(18.0)

    def test_merged_track_spans_the_whole_gap(self):
        tracker = IoUTracker(iou_threshold=0.3, max_missed=1)
        tracker.update([det(0, 0, 40, 40, chainage=0.0)], frame_idx=2)
        tracker.update([], frame_idx=3)
        tracker.update([det(0, 0, 40, 40, chainage=1.0)], frame_idx=40)
        merged = tracker.finish()[0]
        assert merged.first_frame == 2
        assert merged.last_frame == 40

    def test_merge_can_be_turned_off(self):
        tracker = IoUTracker(iou_threshold=0.3, max_missed=1)
        tracker.update([det(0, 0, 40, 40, chainage=0.0)], frame_idx=0)
        tracker.update([], frame_idx=1)
        tracker.update([det(0, 0, 40, 40, chainage=1.0)], frame_idx=5)
        assert len(tracker.finish(merge=False)) == 2

    def test_empty_input_is_safe(self):
        assert merge_duplicates([]) == []


class TestSeverity:
    def test_highest_severity_seen_wins(self):
        """A faint frame must not downgrade a defect that was clearly severe."""
        tracker = IoUTracker(iou_threshold=0.3)
        tracker.update([det(0, 0, 40, 40, chainage=0.0)], frame_idx=0)
        tracker.update([det(0, 0, 40, 40, severity="HIGH", chainage=1.0)], frame_idx=1)
        tracker.update([det(0, 0, 40, 40, severity="LOW", chainage=2.0)], frame_idx=2)
        assert tracker.finish()[0].severity == "HIGH"

    def test_severity_survives_a_merge(self):
        tracker = IoUTracker(iou_threshold=0.3, max_missed=1)
        tracker.update([det(0, 0, 40, 40, severity="LOW", chainage=0.0)], frame_idx=0)
        tracker.update([], frame_idx=1)
        tracker.update([det(0, 0, 40, 40, severity="HIGH", chainage=1.0)], frame_idx=4)
        assert tracker.finish()[0].severity == "HIGH"

    def test_severity_reaches_the_registry(self):
        tracker = IoUTracker(iou_threshold=0.3)
        tracker.update([det(0, 0, 40, 40, severity="MEDIUM", chainage=0.0)], frame_idx=0)
        registry = build_tracking_registry(tracker.finish(), {4: "Pothole"})
        assert next(iter(registry.values()))["severity"] == "MEDIUM"
    def test_registry_is_keyed_by_track_id(self):
        tracker = IoUTracker(iou_threshold=0.3)
        tracker.update([det(0, 0, 40, 40, cls_id=4)], frame_idx=0)
        tracker.update([det(300, 300, 340, 340, cls_id=2)], frame_idx=1)
        registry = build_tracking_registry(tracker.finish(), {4: "Pothole", 2: "Crack"})
        assert set(registry) == {1, 2}
        assert registry[1]["defect_type"] == "Pothole"
        assert registry[2]["defect_type"] == "Crack"

    def test_registry_carries_the_keys_fusion_reads(self):
        """`_compute_vision_pci` needs exactly these two to be present."""
        tracker = IoUTracker(iou_threshold=0.3)
        tracker.update([det(0, 0, 40, 40, dv=12.0)], frame_idx=0)
        registry = build_tracking_registry(tracker.finish(), {4: "Pothole"})
        entry = next(iter(registry.values()))
        assert "deduct_value" in entry
        assert "defect_type" in entry

    def test_empty_input_gives_an_empty_registry(self):
        assert build_tracking_registry([], {}) == {}
