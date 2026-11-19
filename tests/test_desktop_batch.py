"""Exercise desktop file batching without a camera, decoder, or GPU."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'working/desktop'))
import tennis_tacker_live as desktop


class DesktopBatchTests(unittest.TestCase):
    def simulate(self, frame_count, *, cuda=True, model_name=desktop.DEFAULT_MODEL,
                 fast_analysis=True, change_after_prediction=None, prediction_error=None):
        choice = desktop.ModelChoice.preset(model_name)
        with patch.object(desktop.CourtWorker, 'start'), patch.object(desktop.LiveTrackerRun, '_load_catboost'):
            run = desktop.LiveTrackerRun('video.mp4', choice, fast_analysis=fast_analysis)
        run.court_worker = None

        source_fps = 25.0
        input_frames = iter(range(1, frame_count + 1))
        capture = Mock()
        capture.get.return_value = source_fps

        def read_frame():
            frame = next(input_frames, None)
            return frame is not None, frame

        capture.read.side_effect = read_frame
        analyzed = []
        calls = []
        now = 0.0

        def draw(frame, heatmap, index, fps):
            analyzed.append((frame, int(heatmap), index))
            self.assertEqual(fps, source_fps)
            return frame, None

        def predict(frames, realtime=False):
            nonlocal now
            calls.append(tuple(frames))
            now += 0.01
            if change_after_prediction is not None:
                change_after_prediction(run, len(calls))
            if prediction_error is not None:
                raise prediction_error
            return np.asarray(frames) + 100

        detector = SimpleNamespace(
            choice=choice,
            predict=predict,
            close=Mock(),
        )
        run.process_and_draw = draw

        def wait(seconds):
            nonlocal now
            now += seconds
            return False

        with patch.object(desktop, 'Detector', return_value=detector), \
             patch.object(desktop.cv2, 'VideoCapture', return_value=capture), \
             patch.object(desktop.torch.cuda, 'is_available', return_value=cuda), \
             patch.object(desktop.torch.cuda, 'get_device_name', return_value='Test GPU'), \
             patch.object(desktop.torch.cuda, 'empty_cache'), \
             patch.object(desktop.torch, 'set_float32_matmul_precision'), \
             patch.object(desktop.torch, 'set_num_threads'), \
             patch.object(desktop, 'perf_counter', side_effect=lambda: now), \
             patch.object(run.stop, 'wait', side_effect=wait):
            run.run()

        capture.release.assert_called_once()
        detector.close.assert_called_once()
        if prediction_error is None:
            self.assertIsNone(run.last_error)
            self.assertEqual(analyzed, [(i, i + 100, i) for i in range(1, frame_count + 1)])
            self.assertEqual(run._analysis_frame_count, frame_count)

        return run, calls, analyzed

    def test_fixed_triplets_and_short_tails_keep_all_frames_in_order(self):
        for tail in ((), (10,), (10, 11)):
            total = 9 + len(tail)
            with self.subTest(frame_count=total):
                _, calls, _ = self.simulate(total)
                expected = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
                if tail:
                    expected.append(tail)
                self.assertEqual(calls, expected)

    def test_toggling_source_limit_paces_preview_and_keeps_batch_size(self):
        def limit(run, call_index):
            if call_index == 1:
                run.fast_analysis.clear()

        run, calls, _ = self.simulate(12, change_after_prediction=limit)
        self.assertEqual(calls, [(1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)])
        deadlines = [deadline for deadline, _ in run.previews]
        np.testing.assert_allclose(np.diff(deadlines), 1.0 / run.source_fps, atol=1e-12)

    def test_cpu_other_model_and_source_limit_stay_at_three_frames(self):
        for configuration in (
            {'cuda': False},
            {'model_name': 'TrackNetV2 MSPE + CSCA'},
            {'fast_analysis': False},
        ):
            with self.subTest(configuration=configuration):
                _, calls, _ = self.simulate(7, **configuration)
                self.assertEqual(calls, [(1, 2, 3), (4, 5, 6), (7,)])

    def test_prediction_failure_stops_analysis_and_reports_error(self):
        run, calls, analyzed = self.simulate(6, prediction_error=RuntimeError('Test detector failure'))
        self.assertEqual(run.last_error, 'Test detector failure')
        self.assertEqual(calls, [(1, 2, 3)])
        self.assertEqual(analyzed, [])

    def test_model_switch_closes_only_replaced_detector(self):
        original = Mock(choice=desktop.ModelChoice.preset(desktop.DEFAULT_MODEL))
        replacement = Mock(choice=desktop.ModelChoice.preset('TrackNetV2 MSPE + CSCA'))
        with patch.object(desktop.CourtWorker, 'start'), patch.object(desktop.LiveTrackerRun, '_load_catboost'):
            run = desktop.LiveTrackerRun('video.mp4', original.choice)

        with patch.object(desktop, 'Detector', side_effect=[original, OSError('missing weights'), replacement]):
            self.assertIs(run.update_model(None, 'cpu'), original)
            run.requests.put(replacement.choice)
            self.assertIs(run.update_model(original, 'cpu'), original)
            original.close.assert_not_called()

            run.requests.put(replacement.choice)
            self.assertIs(run.update_model(original, 'cpu'), replacement)
            original.close.assert_called_once()
            replacement.close.assert_not_called()


if __name__ == '__main__':
    unittest.main()
