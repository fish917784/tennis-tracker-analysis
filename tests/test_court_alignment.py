"""Current-line checks for small perspective changes of an established court."""
import importlib
from pathlib import Path
import sys
import types
import unittest

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
package = types.ModuleType('court_alignment_test_core')
package.__path__ = [str(ROOT / 'working/court')]
sys.modules[package.__name__] = package
court = importlib.import_module('court_alignment_test_core.court_tracking')


class CourtAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.shape = (342, 608)
        self.border = np.float32([[0, 0], [608, 0], [608, 342], [0, 342]])
        self.known = cv2.getPerspectiveTransform(
            court.COURT_CORNERS,
            np.float32([[160, 50], [450, 50], [570, 310], [40, 310]]),
        )

    def markings_distance(self, matrix, segments=None):
        mask = np.zeros(self.shape, np.uint8)
        segments = court.COURT_SEGMENTS if segments is None else segments
        for start, end in cv2.perspectiveTransform(segments, matrix):
            cv2.line(mask, tuple(np.rint(start).astype(int)),
                     tuple(np.rint(end).astype(int)), 255, 1)
        return cv2.distanceTransform(cv2.bitwise_not(mask), cv2.DIST_L2, cv2.DIST_MASK_3)

    def test_small_camera_tilt_recovers_current_lines_when_translation_cannot(self):
        # Opposite vertical movement at the top corners cannot be represented
        # by one x/y shift. Render the target independently of the refiner.
        camera_adjustment = cv2.getPerspectiveTransform(
            self.border,
            self.border + np.float32([[0, 4], [0, -4], [0, 0], [0, 0]]),
        )
        target = camera_adjustment @ self.known
        distance = self.markings_distance(target)
        self.assertIsNone(court._refine_matrix(self.known, distance, established=True))
        refined = court._refine_projective(self.known, distance)
        self.assertIsNotNone(refined)
        matrix, score, close_score = refined
        scores, close, pattern = court.CourtTracker._line_scores(matrix[None], distance)
        self.assertTrue(court._tracking_support(scores[0], close[0], pattern[0], established=True))
        samples = court._SAMPLES[None, :, :2]
        expected = cv2.perspectiveTransform(samples, target)
        actual = cv2.perspectiveTransform(samples, matrix)
        self.assertLess(np.percentile(np.linalg.norm(expected - actual, axis=2), 95), 2.0)
        correction = matrix @ np.linalg.inv(self.known)
        moved_border = cv2.perspectiveTransform(self.border[None], correction)[0]
        self.assertLessEqual(np.max(np.abs(moved_border - self.border)), 6.001)

    def test_blank_frame_cannot_reactivate_established_court(self):
        distance = np.full(self.shape, 100., np.float32)
        self.assertIsNone(court._refine_projective(self.known, distance))

    def test_high_support_without_complete_marking_pattern_is_rejected(self):
        # Removing two sidelines leaves high average support, but no complete
        # court pattern. Refinement must preserve that independent condition.
        distance = self.markings_distance(self.known, court.COURT_SEGMENTS[2:])
        score, close, pattern = court.CourtTracker._line_scores(self.known[None], distance)
        self.assertGreater(score[0], .77)
        self.assertGreater(close[0], .52)
        self.assertFalse(pattern[0])
        self.assertIsNone(court._refine_projective(self.known, distance))


if __name__ == '__main__':
    unittest.main()
