import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

import numpy as np
import cv2


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "fixed_court_test", ROOT / "working/court/court_tracking.py"
)
# A private package proves the web wrapper does not import its own detector.
import types
PACKAGE = types.ModuleType("fixed_court_package")
PACKAGE.__path__ = [str(ROOT / "working/court")]
sys.modules[PACKAGE.__name__] = PACKAGE
import importlib
court = importlib.import_module("fixed_court_package.court_tracking")


class CourtSchedulingTests(unittest.TestCase):
    def setUp(self):
        self.tracker = court.CourtTracker(fps=30)
        self.result = Mock()
        self.tracker._recover = Mock(return_value=self.result)
        self.tracker._track = Mock(return_value=self.result)
        self.frame = np.zeros((72, 128, 3), dtype=np.uint8)

    def test_sparse_source_frames_each_verify_current_geometry(self):
        for frame_idx in [15, 30, 45, 60]:
            self.assertIs(self.tracker.update(self.frame, frame_idx=frame_idx), self.result)
        self.assertEqual(self.tracker._recover.call_count, 1)
        self.assertEqual(self.tracker._track.call_count, 3)
        self.assertEqual(self.tracker._frame_number, 60)

    def test_omitted_frame_index_keeps_cli_cadence(self):
        for _ in range(4):
            self.tracker.update(self.frame)
        self.assertEqual(self.tracker._recover.call_count, 1)
        self.assertEqual(self.tracker._track.call_count, 1)

    def test_source_rewind_discards_previous_geometry(self):
        self.tracker.update(self.frame, frame_idx=30)
        self.tracker._recent.append(self.result)
        self.tracker.update(self.frame, frame_idx=0)
        self.assertEqual(self.tracker._recover.call_count, 2)
        self.assertEqual(len(self.tracker._recent), 0)
        self.assertEqual(self.tracker._frame_number, 0)

    def test_rejects_negative_source_index(self):
        with self.assertRaises(ValueError):
            self.tracker.update(self.frame, frame_idx=-1)

    def test_current_frame_geometry_verification_requires_court_lines(self):
        matrix = np.array([[15., 0., 60.], [0., 9., 30.], [0., 0., 1.]])
        frame = np.zeros((320, 320, 3), np.uint8)
        for start, end in cv2.perspectiveTransform(court.COURT_SEGMENTS, matrix):
            cv2.line(frame, tuple(np.rint(start).astype(int)), tuple(np.rint(end).astype(int)), (255, 255, 255), 2)
        self.assertTrue(court.court_geometry_is_visible(frame, matrix))
        self.assertFalse(court.court_geometry_is_visible(np.zeros_like(frame), matrix))
        self.assertFalse(court.court_geometry_is_visible(frame, None))


if __name__ == "__main__":
    unittest.main()
