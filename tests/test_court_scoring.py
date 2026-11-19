"""CPU-only exact parity checks for shared court-line projection."""
import importlib
from pathlib import Path
import sys
import types
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = types.ModuleType("court_scoring_parity")
PACKAGE.__path__ = [str(ROOT / "working/court")]
sys.modules[PACKAGE.__name__] = PACKAGE
geometry = importlib.import_module(PACKAGE.__name__ + ".court_geometry")
tracking = importlib.import_module(PACKAGE.__name__ + ".court_tracking")


class SharedCourtScoringTests(unittest.TestCase):
    def assert_matches_reference(self, matrices, distance, points, ids, tolerances):
        with np.errstate(over="ignore", invalid="ignore"):
            supports, visibility = geometry.score_templates_multi(
                matrices, distance, points, ids, tolerances
            )
            for index, tolerance in enumerate(tolerances):
                expected_support, expected_visibility = geometry.score_templates(
                    matrices, distance, points, ids, tolerance
                )
                np.testing.assert_array_equal(supports[index], expected_support)
                np.testing.assert_array_equal(visibility, expected_visibility)
        self.assertEqual(supports.dtype, np.dtype(np.float32))
        self.assertEqual(visibility.dtype, np.dtype(np.float32))

    def test_random_homographies_match_original_for_both_float_types(self):
        rng = np.random.default_rng(7941)
        points, ids = geometry.sample_segments(0.55)
        for dtype in (np.float32, np.float64):
            for count in (1, 5, 32):
                with self.subTest(dtype=dtype, count=count):
                    matrices = np.repeat(np.eye(3, dtype=dtype)[None], count, axis=0)
                    matrices[:, 0, 0] = rng.uniform(0.5, 4.0, count)
                    matrices[:, 1, 1] = rng.uniform(0.5, 3.0, count)
                    matrices[:, :2, 2] = rng.uniform(-30, 70, (count, 2))
                    matrices[:, 2, :2] = rng.uniform(-0.025, 0.025, (count, 2))
                    distance = rng.uniform(0, 4.0, (72, 96)).astype(dtype)
                    self.assert_matches_reference(
                        matrices, distance, points, ids, (2.0, 0.75)
                    )

    def test_threshold_boundaries_and_nonconsecutive_segment_ids(self):
        points = np.array([
            [0., 0., 1.], [1., 0., 1.], [2., 0., 1.], [3., 0., 1.],
            [0., 1., 1.], [1., 1., 1.], [2., 1., 1.], [3., 1., 1.],
        ], dtype=np.float32)
        ids = np.array([0, 0, 2, 2, 5, 5, 9, 9], dtype=np.intp)
        distance = np.array([
            [.75, np.nextafter(np.float32(.75), np.float32(0)),
             np.nextafter(np.float32(.75), np.float32(1)), np.nan],
            [2., np.nextafter(np.float32(2), np.float32(0)),
             np.nextafter(np.float32(2), np.float32(3)), np.inf],
        ], dtype=np.float32)
        self.assert_matches_reference(
            np.eye(3, dtype=np.float32)[None], distance, points, ids,
            (2.0, 0.75, 0.74999999, -1.0, np.nan),
        )

    def test_invalid_and_offscreen_projections_match_original(self):
        points, ids = geometry.sample_segments(0.55)
        matrices = np.repeat(np.eye(3, dtype=np.float64)[None], 5, axis=0)
        matrices[0, 2] = 0
        matrices[1, 0, 0] = np.nan
        matrices[2, 1, 1] = np.inf
        matrices[3, :2, 2] = -1000
        matrices[4, :2, 2] = 1000
        self.assert_matches_reference(matrices, np.ones((30, 40), np.float32), points, ids, (2., .75))

    def test_empty_candidates_and_samples_match_original(self):
        points, ids = geometry.sample_segments(0.55)
        self.assert_matches_reference(np.empty((0, 3, 3), np.float32), np.zeros((4, 5), np.float32),
                                      points, ids, (2., .75))
        self.assert_matches_reference(np.eye(3, dtype=np.float32)[None], np.zeros((4, 5), np.float32),
                                      np.empty((0, 3), np.float32), np.empty(0, np.intp), (2., .75))

    def test_no_tolerances_still_returns_visibility(self):
        points, ids = geometry.sample_segments(0.55)
        matrices = np.eye(3, dtype=np.float32)[None]
        supports, visibility = geometry.score_templates_multi(
            matrices, np.zeros((40, 40), np.float32), points, ids, ()
        )
        _, expected = geometry.score_templates(matrices, np.zeros((40, 40), np.float32), points, ids, 2.)
        self.assertEqual(supports.shape, (0, 1, geometry.SEGMENT_COUNT))
        np.testing.assert_array_equal(visibility, expected)

    def test_tracking_scores_and_acceptance_pattern_are_exactly_equal(self):
        rng = np.random.default_rng(816)
        matrices = np.repeat(np.eye(3)[None], 20, axis=0)
        matrices[:, :2, :2] += rng.uniform(-.1, .1, (20, 2, 2))
        matrices[:, :2, 2] = rng.uniform(-10, 30, (20, 2))
        distance = rng.uniform(0, 3, (72, 96)).astype(np.float32)
        support, visibility = geometry.score_templates(matrices, distance, tracking._SAMPLES, tracking._SEGMENT_IDS, 2.)
        close_support, _ = geometry.score_templates(matrices, distance, tracking._SAMPLES, tracking._SEGMENT_IDS, .75)
        coverage = support * visibility
        expected = (
            coverage.mean(axis=1),
            (close_support * visibility).mean(axis=1),
            ((coverage[:, :4] >= .60).sum(axis=1) >= 3)
            & ((coverage[:, 4:8] >= .60).sum(axis=1) >= 3)
            & (coverage[:, 8:10].max(axis=1) >= .63),
        )
        actual = tracking.CourtTracker._line_scores(matrices, distance)
        for value, reference in zip(actual, expected):
            np.testing.assert_array_equal(value, reference)


if __name__ == "__main__":
    unittest.main()
