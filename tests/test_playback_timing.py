"""Verify source-frame playback timing independently of analysis throughput."""
from collections import deque
from pathlib import Path
from queue import SimpleQueue
import sys
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'working/desktop'))
import tennis_tacker_live as desktop


class PlaybackScheduleTests(unittest.TestCase):
    def test_source_fps_sets_display_spacing(self):
        for fps in (25.0, 29.97, 60.0):
            with self.subTest(fps=fps):
                timestamps, following = desktop.preview_schedule(fps, 3, ready_at=10.0)
                self.assertEqual(len(timestamps), 3)
                np.testing.assert_allclose(np.diff(timestamps), [1.0 / fps] * 2, rtol=0, atol=1e-12)
                self.assertAlmostEqual(timestamps[0], 10.0)
                self.assertAlmostEqual(following, 10.0 + 3.0 / fps)

    def test_fast_analysis_waits_for_next_source_frame_deadline(self):
        first, following = desktop.preview_schedule(25.0, 3, ready_at=10.0)
        second, next_following = desktop.preview_schedule(25.0, 3, ready_at=10.01, next_deadline=following)
        self.assertAlmostEqual(second[0] - first[-1], 1.0 / 25.0)
        self.assertAlmostEqual(second[0], 10.12)
        self.assertAlmostEqual(next_following, 10.24)

    def test_tail_keeps_only_actual_frames_and_duration(self):
        for count in (1, 2):
            with self.subTest(count=count):
                timestamps, following = desktop.preview_schedule(25.0, count, ready_at=10.0)
                self.assertEqual(len(timestamps), count)
                self.assertAlmostEqual(following, 10.0 + count / 25.0)

    def test_slow_analysis_rebases_instead_of_bursting_expired_frames(self):
        _, following = desktop.preview_schedule(25.0, 3, ready_at=10.0)
        timestamps, next_following = desktop.preview_schedule(25.0, 3, ready_at=11.0, next_deadline=following)
        np.testing.assert_allclose(timestamps, [11.0, 11.04, 11.08], rtol=0, atol=1e-12)
        self.assertAlmostEqual(next_following, 11.12)

    def test_fractional_source_fps_does_not_round_to_integer_rate(self):
        timestamps, following = desktop.preview_schedule(29.97, 300, ready_at=0.0)
        self.assertAlmostEqual(timestamps[-1], 299.0 / 29.97)
        self.assertAlmostEqual(following, 300.0 / 29.97)
        self.assertNotAlmostEqual(following, 300.0 / 30.0, places=5)

    def test_invalid_source_rates_and_counts_are_rejected(self):
        for fps in (0.0, -1.0, float('nan'), float('inf'), float('-inf')):
            with self.subTest(fps=fps), self.assertRaises(ValueError):
                desktop.preview_schedule(fps, 3, ready_at=0.0)
        for count in (0, -1, 1.5):
            with self.subTest(count=count), self.assertRaises(ValueError):
                desktop.preview_schedule(25.0, count, ready_at=0.0)


class FilePreviewIntegrationTests(unittest.TestCase):
    def test_file_preview_waits_through_model_loading_and_allows_a_storage_error(self):
        app = desktop.TennisTrackerLiveApp.__new__(desktop.TennisTrackerLiveApp)
        run = SimpleNamespace(source_fps=59.94, analyzed_through=0, draw_frame=Mock(), is_alive=lambda: True)
        playback = Mock()
        self.assertIsNone(app._read_preview(run, playback, 10.))
        playback.take.assert_not_called()
        run.analyzed_through = 3
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        playback.take.return_value = (3, frame, None)
        index, model, displayed, point, latency = app._read_preview(run, playback, 11.)
        self.assertEqual((index, model, point, latency), (3, None, None, None))
        playback.take.assert_called_once_with(11., 59.94, 3, analysis_finished=False)
        self.assertIs(displayed, frame)
        run.draw_frame.assert_not_called()

    def test_file_preview_draws_only_metadata_for_its_source_index(self):
        app = desktop.TennisTrackerLiveApp.__new__(desktop.TennisTrackerLiveApp)
        overlay = SimpleNamespace(point=(3, 7))
        run = SimpleNamespace(source_fps=25., analyzed_through=500, is_alive=lambda: True,
                              draw_frame=Mock(return_value="drawn"))
        playback = Mock()
        playback.take.return_value = (12, "decoded frame", ("EffB0", overlay, 8.0))
        self.assertEqual(app._read_preview(run, playback, 21.), (12, "EffB0", "drawn", (3, 7), 8.0))
        run.draw_frame.assert_called_once_with("decoded frame", overlay, 12, 25.)

    def test_background_eof_keeps_playback_and_its_final_fps_visible(self):
        app = desktop.TennisTrackerLiveApp.__new__(desktop.TennisTrackerLiveApp)
        playback = SimpleNamespace(take=Mock(return_value=None), finished=False,
                                   next_at=None, error=None, close=Mock())
        run = SimpleNamespace(file_playback=playback, source_fps=25., input_fps=25.,
                              analyzed_through=500, analysis_fps=92., stop=Event(),
                              is_alive=lambda: False, last_error=None, events=SimpleQueue())
        app.run = run
        app.next_fps_update = None
        app.preview_times = deque()
        app.last_preview_fps = 24.9
        app.playback_finished = False
        app.closing = False
        app.root = Mock()
        app.preview_fps = Mock()
        app.set_running = Mock()
        app.status = Mock()

        app.poll()
        self.assertIs(app.run, run)
        self.assertIn("分析: 92.0", app.preview_fps.set.call_args.args[0])
        playback.close.assert_not_called()

        playback.finished = True
        app.poll()
        self.assertIsNone(app.run)
        playback.close.assert_called_once_with()
        self.assertIn("分析: 92.0", app.preview_fps.set.call_args.args[0])


class AnalysisThroughputTests(unittest.TestCase):
    def make_run(self, window=30):
        # The metric is independent of models, camera hardware and GUI state.
        run = desktop.LiveTrackerRun.__new__(desktop.LiveTrackerRun)
        run.current_fps = 0.0
        run._fps_timestamps = deque(maxlen=window)
        run._analysis_frame_count = 0
        return run

    def test_first_completion_has_no_elapsed_interval(self):
        run = self.make_run()
        run._record_analysis_fps(3, timestamp=10.0)
        self.assertEqual(run.current_fps, 0.0)

    def test_partial_tail_is_not_counted_as_a_complete_triplet(self):
        run = self.make_run()
        run._record_analysis_fps(3, timestamp=10.0)
        run._record_analysis_fps(3, timestamp=11.0)
        self.assertAlmostEqual(run.current_fps, 3.0)
        run._record_analysis_fps(1, timestamp=12.0)
        self.assertAlmostEqual(run.current_fps, 2.0)

    def test_rolling_window_uses_actual_frame_count_deltas(self):
        run = self.make_run(window=2)
        run._record_analysis_fps(3, timestamp=10.0)
        run._record_analysis_fps(3, timestamp=11.0)
        run._record_analysis_fps(1, timestamp=12.0)
        self.assertAlmostEqual(run.current_fps, 1.0)


class FileRateLimitTests(unittest.TestCase):
    def simulate(self, *, start_limited, enable_after_first_batch=False, release_during_wait=False, inference_time=.01):
        with patch.object(desktop.CourtWorker, 'start'), patch.object(desktop.LiveTrackerRun, '_load_catboost'):
            run = desktop.LiveTrackerRun('video.mp4', None, fast_analysis=not start_limited)
        run.court_worker = None
        now = 0.
        waits = []
        frames = iter(range(1, 11))
        analyzed = []
        capture = Mock()
        capture.get.return_value = 25.

        def read_frame():
            frame = next(frames, None)
            return frame is not None, frame

        capture.read.side_effect = read_frame

        def predict(batch, realtime):
            nonlocal now
            now += inference_time
            if enable_after_first_batch and batch[0] == 4:
                run.fast_analysis.clear()
            return batch

        def wait(seconds):
            nonlocal now
            waits.append(seconds)
            now += seconds
            if release_during_wait:
                run.fast_analysis.set()
            return False

        def draw(frame, heatmap, index, fps):
            analyzed.append((frame, index))
            self.assertEqual(fps, 25.)
            return frame, None

        run.update_model = lambda detector, device: SimpleNamespace(predict=predict, choice=SimpleNamespace(name='Test'))
        run.process_and_draw = draw
        with patch.object(desktop.cv2, 'VideoCapture', return_value=capture), \
             patch.object(desktop.torch.cuda, 'is_available', return_value=False), \
             patch.object(desktop.torch, 'set_num_threads'), patch.object(desktop, 'perf_counter', side_effect=lambda: now), \
             patch.object(run.stop, 'wait', side_effect=wait):
            run.run()
        self.assertIsNone(run.last_error)
        self.assertEqual(analyzed, [(i, i) for i in range(1, 11)])
        return [display_at for display_at, _ in run.previews], waits

    def test_source_limit_paces_analysis_and_preview_even_when_inference_is_fast(self):
        deadlines, waits = self.simulate(start_limited=True)
        np.testing.assert_allclose(np.diff(deadlines), .04, atol=1e-12)
        self.assertGreater(sum(waits), .30)

    def test_disabling_limit_interrupts_wait_and_analyzes_remaining_frames_at_full_speed(self):
        deadlines, waits = self.simulate(start_limited=True, release_during_wait=True)
        self.assertEqual(len(waits), 1)
        self.assertLess(deadlines[3], .13)
        self.assertAlmostEqual(deadlines[6] - deadlines[3], .01)
        self.assertAlmostEqual(deadlines[-1] - deadlines[6], .01)

    def test_enabling_limit_during_unlimited_analysis_adopts_source_cadence(self):
        deadlines, waits = self.simulate(start_limited=False, enable_after_first_batch=True)
        self.assertEqual(deadlines[:3], [.01] * 3)
        np.testing.assert_allclose(np.diff(deadlines[3:]), .04, atol=1e-12)
        self.assertGreater(sum(waits), .20)

    def test_slow_inference_is_not_delayed_by_the_source_limit(self):
        deadlines, waits = self.simulate(start_limited=True, inference_time=.20)
        self.assertEqual(waits, [])
        self.assertAlmostEqual(deadlines[3] - deadlines[0], .20)


if __name__ == '__main__':
    unittest.main()
