"""Automatic court tracking when the near corners or baseline are offscreen."""

import importlib
from pathlib import Path
import sys
import types
import unittest

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
package = types.ModuleType('cropped_court_test_core')
package.__path__ = [str(ROOT / 'working/court')]
sys.modules[package.__name__] = package
court = importlib.import_module('cropped_court_test_core.court_tracking')


def render_court(near_y, *, offset=0, omit=None):
    corners = np.float32([[295, 290], [555, 293], [1040, near_y + 4], [-215, near_y]])
    corners[:, 0] += offset
    return render_markings(corners, (464, 848), omit=omit)


def render_markings(corners, shape, *, omit=None):
    matrix = cv2.getPerspectiveTransform(court.COURT_CORNERS, corners)
    frame = np.full((*shape, 3), (65, 76, 46), np.uint8)
    cv2.fillConvexPoly(frame, np.rint(corners).astype(np.int32), (132, 88, 55))
    projected = cv2.perspectiveTransform(court.COURT_SEGMENTS, matrix)
    for index, (start, end) in enumerate(np.rint(projected).astype(int)):
        if index != omit:
            cv2.line(frame, tuple(start), tuple(end), (235, 235, 235), 2, cv2.LINE_AA)
    return frame, matrix


class CroppedCourtTests(unittest.TestCase):
    def assert_visible_markings_match(self, result, expected, shape):
        points = court._SAMPLES[None, :, :2]
        target = cv2.perspectiveTransform(points, expected)[0]
        valid = ((target >= 0) & (target < np.asarray(shape[:2][::-1]))).all(axis=1)
        segments = cv2.perspectiveTransform(court.COURT_SEGMENTS, result.court_to_image)
        starts, ends = segments[court._SEGMENT_IDS[valid]].transpose(1, 0, 2)
        directions = ends - starts
        delta = target[valid] - starts
        # Distance across the marking measures whether it overlays the real
        # line. Offscreen endpoints can extrapolate along the same line.
        area = directions[:, 0] * delta[:, 1] - directions[:, 1] * delta[:, 0]
        errors = np.abs(area) / np.linalg.norm(directions, axis=1)
        self.assertLess(np.percentile(errors, 95), 4.)

    def test_automatic_confirmation_with_cropped_corners_and_baseline(self):
        for near_y in (430, 488):
            with self.subTest(near_y=near_y):
                frame, expected = render_court(near_y)
                tracker = court.CourtTracker(30.)
                self.assertIsNone(tracker.update(frame, 0))
                result = tracker.update(frame, 6)
                self.assertIsNotNone(result)
                self.assertTrue(result.partial)
                self.assert_visible_markings_match(result, expected, frame.shape)
                moved, expected_moved = render_court(near_y, offset=3)
                verified = court.refine_court_geometry(moved, result.court_to_image,
                                                       established=True, partial=True)
                self.assertIsNotNone(verified)
                self.assert_visible_markings_match(verified, expected_moved, moved.shape)

    def test_missing_near_service_line_and_blank_view_cannot_calibrate(self):
        incomplete, _ = render_court(488, omit=6)
        tracker = court.CourtTracker(30.)
        for index in (0, 6, 12):
            self.assertIsNone(tracker.update(incomplete, index))

        frame, _ = render_court(488)
        tracker = court.CourtTracker(30.)
        self.assertIsNone(tracker.update(frame, 0))
        result = tracker.update(frame, 6)
        self.assertIsNotNone(result)
        blank = np.full_like(frame, 180)
        self.assertIsNone(court.refine_court_geometry(blank, result.court_to_image,
                                                      established=True, partial=True))
        self.assertIsNone(tracker.update(blank, 7))
        self.assertGreater(tracker.cuts, 0)

    def test_oblique_near_baseline_and_nearly_vertical_sideline(self):
        corners = np.float32([[403, 116], [606, 128], [722, 396], [74, 244]])
        frame, expected = render_markings(corners, (342, 608))
        tracker = court.CourtTracker(30.)
        self.assertIsNone(tracker.update(frame, 0))
        result = tracker.update(frame, 6)
        self.assertIsNotNone(result)
        self.assertTrue(result.partial)
        self.assert_visible_markings_match(result, expected, frame.shape)


if __name__ == '__main__':
    unittest.main()
