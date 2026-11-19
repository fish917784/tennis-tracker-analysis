import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("review_bounce", ROOT / "working/web/bounce_detector.py")
bounce = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bounce)


class FeatureModel:
    def predict(self, features):
        scores = np.zeros(len(features))
        scores[:2] = [0.9, 0.8]
        return scores


class SegmentDetector:
    def __init__(self):
        self.lengths = []

    def is_ready(self):
        return True

    def predict_bounces(self, xs, ys, smooth=True):
        self.lengths.append(len(xs))
        return {2: 0.9}


def state(segment=0, matrix=None, valid=True):
    matrix = np.eye(3) if matrix is None else matrix
    return dict(H=matrix, H_inv=np.linalg.inv(matrix), calibrated=valid, segment_id=segment)


class BounceTests(unittest.TestCase):
    def setUp(self):
        self.detector = bounce.CatBoostBounceDetector.__new__(bounce.CatBoostBounceDetector)
        self.detector.threshold = 0.45
        self.detector.model = FeatureModel()

    def test_contiguous_group_keeps_global_peak(self):
        selected = self.detector.postprocess(np.array([0, 1, 2]), np.array([0.9, 0.6, 0.7]))
        self.assertEqual(selected, [0])

    def test_missing_frames_do_not_merge_separate_peaks(self):
        xs = [float(i + 1) for i in range(40)]
        ys = xs.copy()
        for index in range(5, 31):
            xs[index] = ys[index] = None
        result = self.detector.predict_bounces(xs, ys, smooth=False)
        self.assertEqual(result, {2: 0.9, 33: 0.8})

    def test_real_zero_coordinate_is_not_extrapolated(self):
        xs = [10.0, 9.0, 8.0, 7.0, 6.0, 0.0, -1.0]
        smoothed, _ = self.detector.smooth_predictions(xs, [1.0] * len(xs))
        self.assertEqual(smoothed[5], 0.0)

    def test_all_invisible_window_returns_no_predictions(self):
        result = self.detector.predict_bounces([None] * 9, [None] * 9)
        self.assertEqual(result, {})

    def test_missing_y_invalidates_entire_feature_window(self):
        xs = list(range(9))
        ys = xs.copy()
        ys[4] = None
        features, frames = self.detector.prepare_features(xs, ys)
        self.assertEqual(frames, [])
        self.assertTrue(features.empty)

    def test_camera_segments_never_share_features_or_debounce(self):
        points = [dict(x=5.0, y=10.0, visible=True) for _ in range(16)]
        detector = SegmentDetector()
        states = [state(segment=i // 8) for i in range(16)]
        with patch.object(bounce, "get_bounce_detector", return_value=detector):
            result = bounce.detect_court_bounces_catboost(points, None, None, court_states=states)
        self.assertEqual(detector.lengths, [8, 8])
        self.assertEqual([item["frame"] for item in result], [2, 10])
        self.assertEqual([item["segment_id"] for item in result], [0, 1])

    def test_uses_bounce_frames_matrix_not_first_matrix(self):
        points = [dict(x=5.0, y=10.0, visible=True) for _ in range(8)]
        states = [state() for _ in points]
        moved = np.eye(3)
        moved[0, 2] = 1.0
        states[2] = state(matrix=moved)
        with patch.object(bounce, "get_bounce_detector", return_value=SegmentDetector()):
            result = bounce.detect_court_bounces_catboost(points, None, None, court_states=states)
        self.assertEqual(result[0]["court_x"], 6.0)

    def test_unverified_court_produces_no_predictions(self):
        points = [dict(x=5.0, y=10.0, visible=True) for _ in range(12)]
        detector = SegmentDetector()
        with patch.object(bounce, "get_bounce_detector", return_value=detector):
            result = bounce.detect_court_bounces_catboost(
                points, np.eye(3), np.eye(3), court_states=[state(valid=False) for _ in points]
            )
        self.assertEqual(result, [])
        self.assertEqual(detector.lengths, [])

    def test_invalid_court_splits_small_windows(self):
        points = [dict(x=5.0, y=10.0, visible=True) for _ in range(9)]
        states = [state() for _ in points]
        states[4] = state(valid=False)
        detector = SegmentDetector()
        with patch.object(bounce, "get_bounce_detector", return_value=detector):
            result = bounce.detect_court_bounces_catboost(points, None, None, court_states=states)
        self.assertEqual(result, [])
        self.assertEqual(detector.lengths, [])

    def test_invalid_matrices_and_bad_inputs(self):
        points = [dict(x=5.0, y=10.0, visible=True) for _ in range(10)]
        self.assertEqual(bounce.detect_court_bounces_catboost(points, None, None), [])
        with self.assertRaises(ValueError):
            bounce.detect_court_bounces_catboost(points, None, None, fps=0)
        with self.assertRaises(ValueError):
            bounce.detect_court_bounces_catboost(points, None, None, court_states=[])


if __name__ == "__main__":
    unittest.main()
