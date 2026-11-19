from collections import deque
from pathlib import Path
from queue import SimpleQueue
from threading import Event, Thread
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'working/desktop'))
import tennis_tacker_live as desktop


class FakeWorker:
    def __init__(self, callback):
        self.callback = callback
        self.busy = False
        self.submit = Mock()
    def start(self):
        pass


class DesktopTests(unittest.TestCase):
    def setUp(self):
        with patch.object(desktop, 'CourtWorker', FakeWorker), patch.object(desktop.LiveTrackerRun, '_load_catboost'):
            self.run = desktop.LiveTrackerRun('unused.mp4', None)
        self.run.H = np.eye(3)
        self.run.H_inv = np.eye(3)
        self.run.court_calibrated = True
        self.run._last_court_result = 1
        self.run.show_court = False
        self.frame = np.zeros((180, 320, 3), dtype=np.uint8)
        self.heatmap = np.zeros((288, 512), dtype=np.float32)

    def test_worker_publishes_without_mutating_shared_matrix(self):
        thread = Thread(target=self.run.on_court_detected, args=(None, 2))
        thread.start()
        thread.join()
        np.testing.assert_array_equal(self.run.H, np.eye(3))
        self.assertTrue(self.run.court_calibrated)
        with patch.object(desktop, '_COURT_READY', False):
            self.run.process_and_draw(self.frame, self.heatmap, 2, 30)
        self.assertIsNone(self.run.H)
        self.assertFalse(self.run.court_calibrated)

    def test_prefetched_initial_court_frame_is_only_probed_once(self):
        with patch.object(desktop, '_COURT_READY', True), \
             patch.object(desktop, 'detect_court', return_value=None) as detect:
            self.run._probe_first_court(self.frame)
            self.run._update_live_court(self.frame, 1, 30.)
        detect.assert_called_once_with(self.frame, max_width=320)
        self.run.court_worker.submit.assert_called_once_with(self.frame, 1)

    def test_nonconsecutive_bounce_window_is_rejected(self):
        self.run.catboost_model = Mock()
        self.run.ball_history.extend((f, 5., 10., True, 5., 10., 0.) for f in [1, 2, 3, 20, 21])
        self.run.check_live_bounce(21, 30)
        self.run.catboost_model.predict.assert_not_called()
        self.assertEqual(self.run.bounces, [])

    def _seed_display(self):
        detection = SimpleNamespace(score=.95, validation_score=.9, image_to_court=np.eye(3),
                                    court_to_image=np.eye(3), quad=np.array([[0.,0.],[10.,0.],[10.,20.],[0.,20.]]))
        self.run._apply_court_result(detection, 10)
        return detection

    def test_short_occlusion_holds_only_display_and_suspends_line_calls(self):
        self._seed_display()
        self.run.bounces.append({'id': 1})
        self.run._apply_court_result(None, 11)
        self.assertIsNotNone(self.run._court_for_display(15, 60.))
        self.assertFalse(self.run.court_calibrated)
        self.assertIsNone(self.run.H)
        self.run.catboost_model = Mock()
        self.run.check_live_bounce(15, 60.)
        self.run.catboost_model.predict.assert_not_called()
        self.assertEqual(len(self.run.bounces), 1)

    def test_prolonged_loss_clears_display_and_history(self):
        self._seed_display()
        self.run.bounces.append({'id': 1})
        self.run._apply_court_result(None, 11)
        self.assertIsNone(self.run._court_for_display(38, 60.))
        self.assertEqual(self.run.bounces, [])

    def test_delayed_failure_starts_grace_at_current_observation(self):
        self._seed_display()
        self.run.on_court_detected(None, 20)
        with patch.object(desktop, '_COURT_READY', False), patch.object(self.run, '_verify_court_on_frame', return_value=False):
            self.run.process_and_draw(self.frame, self.heatmap, 60, 60.)
        self.assertEqual(self.run._last_court_result, 20)
        self.assertEqual(self.run._court_invalid_since, 60)
        self.assertIsNotNone(self.run._court_for_display(86, 60.))
        self.assertIsNone(self.run._court_for_display(87, 60.))

    def test_display_pose_is_only_reactivated_after_current_line_verification(self):
        detection = self._seed_display()
        self.run._apply_court_result(None, 11)
        with patch.object(desktop, 'refine_court_geometry', return_value=None) as verify:
            self.assertFalse(self.run._verify_court_on_frame(self.frame, allow_display=True))
            self.assertTrue(verify.call_args.kwargs['established'])
        self.assertIsNone(self.run.H)
        with patch.object(desktop, 'refine_court_geometry', return_value=detection):
            self.run.on_court_detected(None, 12)
            with patch.object(desktop, '_COURT_READY', False):
                self.run.process_and_draw(self.frame, self.heatmap, 12, 60.)
        self.assertTrue(self.run.court_calibrated)

    def test_real_cut_immediately_clears_display_without_grace(self):
        self._seed_display()
        self.run._apply_court_result(None, 11, hard=True)
        self.assertIsNone(self.run._court_for_display(11, 60.))

    def test_confirmed_recovery_keeps_history_from_short_occlusion(self):
        detection = self._seed_display()
        self.run.bounces.append({'id': 1})
        self.run._apply_court_result(None, 11)
        self.run._apply_court_result(detection, 20)
        self.assertTrue(self.run.court_calibrated)
        self.assertEqual(len(self.run.bounces), 1)

    def test_hud_displays_source_fps_separately_from_analysis(self):
        self.run.current_fps = 123.0
        with patch.object(desktop, '_COURT_READY', False), \
             patch.object(desktop.cv2, 'putText', wraps=desktop.cv2.putText) as draw_text:
            self.run.process_and_draw(self.frame, self.heatmap, 3, 25.0)
        labels = [call.args[1] for call in draw_text.call_args_list]
        self.assertIn('SOURCE FPS: 25.00', labels)
        self.assertIn('ANALYSIS: 123.0 FPS', labels)

    def test_pending_reply_does_not_expire_a_visible_court(self):
        self.run._court_verified_frame = 1
        with patch.object(desktop, '_COURT_READY', False), patch.object(self.run, '_verify_court_on_frame', return_value=True):
            self.run.process_and_draw(self.frame, self.heatmap, 17, 25)
        self.assertTrue(self.run.court_calibrated)
        self.assertEqual(self.run._court_verified_frame, 17)

    def test_file_analysis_is_unlimited_by_default_and_limiting_is_explicit(self):
        self.assertTrue(self.run.fast_analysis.is_set())
        with patch.object(desktop, 'CourtWorker', FakeWorker), patch.object(desktop.LiveTrackerRun, '_load_catboost'):
            run = desktop.LiveTrackerRun('unused.mp4', None, fast_analysis=False)
        self.assertFalse(run.fast_analysis.is_set())

    def test_fast_processing_accepts_late_reply_only_after_current_geometry_check(self):
        detection = SimpleNamespace(score=.99, validation_score=.9, image_to_court=np.eye(3),
                                    court_to_image=np.eye(3), quad=np.zeros((4, 2)))
        self.run.on_court_detected(detection, 2)
        with patch.object(desktop, '_COURT_READY', False), patch.object(self.run, '_verify_court_on_frame', return_value=True):
            self.run.process_and_draw(self.frame, self.heatmap, 40, 25)
        self.assertTrue(self.run.court_calibrated)
        self.assertEqual(self.run._last_court_result, 2)
        self.assertEqual(self.run._court_verified_frame, 40)

    def test_expired_geometry_is_removed_when_current_lines_do_not_match(self):
        self.run._court_verified_frame = 1
        with patch.object(desktop, '_COURT_READY', False), patch.object(self.run, '_verify_court_on_frame', return_value=False):
            self.run.process_and_draw(self.frame, self.heatmap, 17, 25)
        self.assertFalse(self.run.court_calibrated)
        self.assertIsNone(self.run.H)

    def test_busy_worker_keeps_newest_pending_frame_at_25fps(self):
        self.run.court_worker.busy = True
        with patch.object(desktop, '_COURT_READY', False):
            self.run.process_and_draw(self.frame, self.heatmap, 3, 25)
        self.run.court_worker.submit.assert_called_once()
        self.assertEqual(self.run.court_worker.submit.call_args.args[1], 3)

    def test_single_failed_reply_keeps_current_verified_geometry(self):
        self.run.on_court_detected(None, 2)
        with patch.object(desktop, '_COURT_READY', False), patch.object(self.run, '_verify_court_on_frame', return_value=True):
            self.run.process_and_draw(self.frame, self.heatmap, 2, 25)
        self.assertTrue(self.run.court_calibrated)
        self.assertEqual(self.run._last_court_result, 2)

    def test_core_accepted_tracking_score_does_not_drop_live_court(self):
        detection = SimpleNamespace(score=.79, validation_score=.6, image_to_court=np.eye(3),
                                    court_to_image=np.eye(3), quad=np.zeros((4, 2)))
        self.run._apply_court_result(detection, 2)
        self.assertTrue(self.run.court_calibrated)

    def test_verified_tracking_score_can_recover_after_occlusion(self):
        self.run.court_calibrated = False
        detection = SimpleNamespace(score=.79, validation_score=.6, image_to_court=np.eye(3),
                                    court_to_image=np.eye(3), quad=np.zeros((4, 2)))
        self.run._apply_court_result(detection, 2)
        self.assertTrue(self.run.court_calibrated)

    def test_five_consecutive_visible_frames_can_produce_bounce(self):
        self.run.catboost_model = Mock()
        self.run.catboost_model.predict.return_value = [0.9]
        self.run.ball_history.extend((f, 5., 10., True, 5., 10., 42.) for f in range(5))
        self.run.check_live_bounce(4, 30)
        self.assertEqual(self.run.bounces[0]['frame'], 2)
        self.assertEqual(self.run.bounces[0]['speed_kmh'], 42.)
        self.assertEqual(self.run.bounces[0]['speed_type'], 'ground_projection_estimate')

    def test_source_gap_resets_motion_and_small_frame_renders(self):
        self.run._last_processed_frame = 3
        self.run.ball_history.extend((f, 5., 10., True, 5., 10., 42.) for f in range(4))
        self.run.current_speed_kmh = 99.
        with patch.object(desktop, '_COURT_READY', False):
            rendered, _ = self.run.process_and_draw(self.frame, self.heatmap, 20, 30)
        self.assertEqual(rendered.shape, self.frame.shape)
        self.assertEqual(len(self.run.ball_history), 1)
        self.assertEqual(self.run.current_speed_kmh, 0.)

    def test_outdated_worker_result_cannot_undo_new_state(self):
        self.run._last_court_result = 10
        self.run._court_verified_frame = 10
        self.run.on_court_detected(None, 5)
        with patch.object(desktop, '_COURT_READY', False):
            self.run.process_and_draw(self.frame, self.heatmap, 11, 30)
        self.assertTrue(self.run.court_calibrated)

    def test_camera_cut_immediately_invalidates_court(self):
        with patch.object(desktop, '_COURT_READY', False):
            self.run.process_and_draw(self.frame, self.heatmap, 1, 30)
            self.run.process_and_draw(np.full_like(self.frame, 255), self.heatmap, 2, 30)
        self.assertFalse(self.run.court_calibrated)

    def test_gui_reports_error_and_preserves_failure_status(self):
        app = desktop.TennisTrackerLiveApp.__new__(desktop.TennisTrackerLiveApp)
        event_queue = SimpleQueue()
        event_queue.put(('error', 'test missing weights'))
        run = SimpleNamespace(stop=Event(), events=event_queue, last_error=None,
                              is_alive=lambda: False, previews=deque(), input_fps=None)
        run.stop.set()
        app.run = run
        app.pending_preview = None
        app.last_preview = None
        app.preview_times = deque()
        app.next_fps_update = float('inf')
        app.closing = False
        app.status = Mock()
        app.preview_fps = Mock()
        app.root = Mock()
        app.set_running = Mock()
        with patch.object(desktop.messagebox, 'showerror') as showerror:
            app.poll()
        showerror.assert_called_once()
        self.assertIn('test missing weights', app.status.set.call_args.args[0])


if __name__ == '__main__':
    unittest.main()
