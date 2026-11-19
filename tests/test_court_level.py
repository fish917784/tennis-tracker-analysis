import importlib
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
package = types.ModuleType('court_level_test_core')
package.__path__ = [str(ROOT / 'working/court')]
sys.modules[package.__name__] = package
court = importlib.import_module('court_level_test_core.court_tracking')


def detection(dx=0.):
    matrix = np.array([[15., 0., 60. + dx], [0., 9., 30.], [0., 0., 1.]])
    quad = cv2.perspectiveTransform(court.COURT_CORNERS[None], matrix)[0]
    return court.CourtDetection(np.linalg.inv(matrix), matrix, quad, .86, .82)


class CourtLevelTests(unittest.TestCase):
    def setUp(self):
        self.tracker = court.CourtTracker(60.)
        self.tracker._shape = (360, 640)
        self.frame = np.zeros((360, 640, 3), np.uint8)
        self.distance = np.zeros((360, 640), np.float32)
        self.score = patch.object(self.tracker, '_line_scores', return_value=(np.array([.88]), np.array([.70]), np.array([True])))
        self.score.start()
        self.lines = patch.object(court, '_long_transverse_lines', return_value=[])
        self.lines.start()

    def tearDown(self):
        self.lines.stop()
        self.score.stop()

    def confirm(self, frame_idx):
        self.tracker._frame_number = frame_idx
        return self.tracker._confirm_partial_court(self.frame, self.distance, np.eye(3))

    def test_low_view_requires_multiple_scales_and_frames(self):
        with patch.object(court, '_detect_at_width', return_value=detection()):
            self.assertIsNone(self.confirm(10))
            self.assertIsNotNone(self.confirm(20))
        self.assertEqual(len(self.tracker._recent), 1)

    def test_single_scale_requires_three_currently_verified_frames(self):
        with patch.object(court, '_detect_at_width', side_effect=lambda image, size: detection() if size == 320 else None):
            self.assertIsNone(self.confirm(10))
            self.assertIsNone(self.confirm(20))
            self.assertIsNotNone(self.confirm(30))

    def test_disagreeing_half_courts_cannot_vote_together(self):
        with patch.object(court, '_detect_at_width', side_effect=lambda image, size: detection({320: 0., 512: 100., 640: 200.}[size])):
            self.assertIsNone(self.confirm(10))
            self.assertIsNone(self.confirm(20))
            self.assertIsNone(self.confirm(30))

    def test_one_missing_sample_does_not_erase_unpublished_evidence(self):
        with patch.object(court, '_detect_at_width', return_value=detection()):
            self.assertIsNone(self.confirm(10))
        with patch.object(court, '_detect_at_width', return_value=None):
            self.assertIsNone(self.confirm(20))
        with patch.object(court, '_detect_at_width', return_value=detection()):
            self.assertIsNotNone(self.confirm(30))

    def test_reset_discards_pending_calibration(self):
        with patch.object(court, '_detect_at_width', return_value=detection()):
            self.assertIsNone(self.confirm(10))
            self.tracker._reset_tracking()
            self.assertIsNone(self.confirm(20))

    def test_existing_pose_tolerance_still_requires_strong_full_pattern(self):
        self.assertFalse(court._tracking_support(.85, .50, True, established=False))
        self.assertTrue(court._tracking_support(.85, .50, True, established=True))
        self.assertFalse(court._tracking_support(.79, .48, True, established=True))
        self.assertFalse(court._tracking_support(.95, .80, False, established=True))

    def test_unexplained_near_baseline_rejects_service_line_fit(self):
        matrix = detection().court_to_image
        self.assertTrue(court._baseline_beyond(matrix, [[0, 300, 639, 300]], (360, 640)))
        self.assertFalse(court._baseline_beyond(matrix, [[0, 180, 639, 180]], (360, 640)))

    def test_tracking_loss_retains_matching_last_verified_view_for_recovery(self):
        known = detection()
        previous = np.full((342, 608), 70, np.uint8)
        self.tracker._result = known
        self.tracker._track_gray = previous
        with patch.object(self.tracker, '_track', return_value=None), patch.object(self.tracker, '_recover', return_value=None):
            self.tracker.update(self.frame, frame_idx=10)
        saved, image = self.tracker._recent_views[0]
        self.assertIs(saved, known)
        np.testing.assert_array_equal(image, previous)
        self.assertIsNone(self.tracker._result)

    def test_currently_valid_refinement_replaces_invalid_pose_despite_lower_average(self):
        self.tracker._result = detection()
        gray = np.zeros_like(self.distance, dtype=np.uint8)
        self.tracker._track_gray = gray
        adjusted = self.tracker._result.court_to_image.copy()
        adjusted[1, 2] += 2.
        with patch.object(self.tracker, '_line_scores', return_value=(np.array([.835]), np.array([.42]), np.array([True]))), \
             patch.object(court, '_refine_matrix', return_value=(adjusted, .78, .56)):
            result = self.tracker._track(gray, self.distance, np.eye(3))
        self.assertIsNotNone(result)
        np.testing.assert_allclose(result.court_to_image, adjusted)


class CourtViewChangeTests(unittest.TestCase):
    def test_similar_lower_court_does_not_hide_an_edit_in_upper_view(self):
        previous = np.zeros((54, 96), np.uint8)
        current = previous.copy()
        current[:27, :48] = 100
        self.assertLess(np.count_nonzero(current > 29) / current.size, .31)
        self.assertTrue(court.court_view_changed(previous, current))

    def test_local_player_motion_is_not_a_view_change(self):
        previous = np.zeros((54, 96), np.uint8)
        current = previous.copy()
        current[10:45, 43:53] = 255
        self.assertFalse(court.court_view_changed(previous, current))


if __name__ == '__main__':
    unittest.main()
