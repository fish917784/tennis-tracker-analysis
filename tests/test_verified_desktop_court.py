import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "working/desktop"))
import tennis_tacker_live as desktop


SPEC = importlib.util.spec_from_file_location(
    "desktop_verified_test_fixture", ROOT / "working/web/verified_court.py"
)
verified = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verified)


class VerifiedDesktopCourtTests(unittest.TestCase):
    def setUp(self):
        self.worker = Mock()
        with patch.object(desktop, "CourtWorker", return_value=self.worker), \
             patch.object(desktop.LiveTrackerRun, "_load_catboost"):
            self.run = desktop.LiveTrackerRun("verified.mp4", None)
        self.frame = np.zeros((180, 320, 3), dtype=np.uint8)
        self.heat = np.zeros((288, 512), dtype=np.float32)
        self.run.show_minicourt = False

    def geometry(self, offsets, segments, displayed):
        matrices, polygons = [], []
        corners = np.array([[0., 0.], [11., 0.], [11., 24.], [0., 24.]], dtype=np.float32)
        for offset in offsets:
            if offset is None:
                matrices.append(np.full((3, 3), np.nan))
                polygons.append(np.full((4, 2), np.nan, dtype=np.float32))
            else:
                matrices.append(np.array([[8., 0., offset], [0., 4., 40.], [0., 0., 1.]]))
                polygons.append(corners * np.array([8., 4.]) + np.array([offset, 40.]))
        count = len(offsets)
        geometry = verified.VerifiedCourtGeometry(
            dict(fps=30., width=320, height=180, frame_count=count,
                 calibrated_frames=sum(offset is not None for offset in offsets)),
            dict(court_to_image=np.stack(matrices), court_quad=np.stack(polygons),
                 segment_id=np.array(segments, dtype=np.int32), displayed=np.array(displayed)),
        )
        self.run._source_court = geometry
        return geometry

    def draw_frame(self, frame_idx):
        self.run.process_and_draw(self.frame, self.heat, frame_idx, 30.)

    def test_verified_playback_uses_zero_based_geometry_and_skips_auto_detection(self):
        geometry = self.geometry([70, 90], [1, 1], [True, True])
        with patch.object(geometry, "update", wraps=geometry.update) as update:
            self.draw_frame(1)
            np.testing.assert_array_equal(self.run.H_inv, geometry.to_image[0])
            self.draw_frame(2)
            np.testing.assert_array_equal(self.run.H_inv, geometry.to_image[1])
        self.assertEqual([call.args[1] for call in update.call_args_list], [0, 1])
        self.assertTrue(self.run.court_calibrated)
        self.worker.submit.assert_not_called()

    def test_verified_occlusion_disables_projection_before_the_outline_expires(self):
        self.geometry([70, None, None], [1, 2, 2], [True, True, False])
        self.draw_frame(1)
        self.assertTrue(self.run.project_point(100., 80.)[2])
        self.draw_frame(2)
        self.assertTrue(self.run.court_displayed)
        self.assertFalse(self.run.court_calibrated)
        self.assertIsNone(self.run.H)
        self.assertIsNone(self.run.H_inv)
        self.assertEqual(self.run.project_point(100., 80.), (0., 0., False))
        self.draw_frame(3)
        self.assertFalse(self.run.court_displayed)
        self.assertFalse(self.run.court_calibrated)

    def test_cut_between_two_verified_courts_clears_previous_shot_history(self):
        geometry = self.geometry([70, 90], [1, 2], [True, True])
        self.draw_frame(1)
        self.run.bounces.append({"frame": 1, "id": 1})
        self.run.ball_trail.append((100, 80))
        self.assertTrue(self.run.ball_history)
        self.draw_frame(2)
        self.assertTrue(self.run.court_calibrated)
        np.testing.assert_array_equal(self.run.H_inv, geometry.to_image[1])
        self.assertEqual(self.run.bounces, [])
        self.assertEqual(list(self.run.ball_trail), [])
        self.assertEqual([entry[0] for entry in self.run.ball_history], [2])


if __name__ == "__main__":
    unittest.main()
