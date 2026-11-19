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
                 fast_analysis=True, six_frame_batch=True, change_after_prediction=None,
                 grouped_error=None):
        with patch.object(desktop.CourtWorker, 'start'), patch.object(desktop.LiveTrackerRun, '_load_catboost'):
            run = desktop.LiveTrackerRun(
                'video.mp4', desktop.ModelChoice.preset(desktop.DEFAULT_MODEL),
                fast_analysis=fast_analysis, six_frame_batch=six_frame_batch,
            )
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

        def predict(groups, grouped):
            nonlocal now
            groups = tuple(tuple(group) for group in groups)
            calls.append((grouped, groups))
            now += 0.01
            if change_after_prediction is not None:
                change_after_prediction(run, len(calls))
            if grouped and grouped_error is not None:
                raise grouped_error
            return [np.asarray(group) + 100 for group in groups]

        detector = SimpleNamespace(
            choice=SimpleNamespace(name=model_name),
            predict=lambda frames, realtime=False: predict([frames], grouped=False)[0],
            predict_batch=lambda groups: predict(groups, grouped=True),
        )
        run.update_model = lambda previous, device: detector
        run.process_and_draw = draw

        def wait(seconds):
            nonlocal now
            now += seconds
            return False

        with patch.object(desktop.cv2, 'VideoCapture', return_value=capture), \
             patch.object(desktop.torch.cuda, 'is_available', return_value=cuda), \
             patch.object(desktop.torch.cuda, 'get_device_name', return_value='Test GPU'), \
             patch.object(desktop.torch.cuda, 'empty_cache'), \
             patch.object(desktop.torch, 'set_float32_matmul_precision'), \
             patch.object(desktop.torch, 'set_num_threads'), \
             patch.object(desktop, 'perf_counter', side_effect=lambda: now), \
             patch.object(run.stop, 'wait', side_effect=wait):
            run.run()

        capture.release.assert_called_once()
        if grouped_error is None or isinstance(grouped_error, desktop.torch.cuda.OutOfMemoryError):
            self.assertIsNone(run.last_error)
            self.assertEqual(analyzed, [(i, i + 100, i) for i in range(1, frame_count + 1)])
            self.assertEqual(run._analysis_frame_count, frame_count)

        return run, calls, analyzed

    def test_six_frames_are_two_ordered_triplets_and_short_tails_count_once(self):
        for total in (10, 11, 13):
            with self.subTest(frame_count=total):
                _, calls, _ = self.simulate(total)
                self.assertEqual(calls[0], (True, ((1, 2, 3), (4, 5, 6))))
                if total == 13:
                    self.assertEqual(calls[1:], [
                        (True, ((7, 8, 9), (10, 11, 12))),
                        (False, ((13,),)),
                    ])
                else:
                    self.assertEqual(calls[1:], [
                        (True, ((7, 8, 9), tuple(range(10, total + 1)))),
                    ])

    def test_toggling_source_limit_during_batch_changes_next_batch_size_and_paces_preview(self):
        def limit(run, call_index):
            if call_index == 1:
                run.fast_analysis.clear()

        run, calls, _ = self.simulate(12, change_after_prediction=limit)
        self.assertEqual(calls, [
            (True, ((1, 2, 3), (4, 5, 6))),
            (False, ((7, 8, 9),)),
            (False, ((10, 11, 12),)),
        ])
        deadlines = [deadline for deadline, _ in run.previews]
        np.testing.assert_allclose(np.diff(deadlines), 1.0 / run.source_fps, atol=1e-12)

    def test_rollback_can_disable_and_reenable_batching_midstream(self):
        def toggle(run, call_index):
            if call_index == 1:
                run.six_frame_batch.clear()
            elif call_index == 2:
                run.six_frame_batch.set()

        _, calls, _ = self.simulate(15, change_after_prediction=toggle)
        self.assertEqual(calls, [
            (True, ((1, 2, 3), (4, 5, 6))),
            (False, ((7, 8, 9),)),
            (True, ((10, 11, 12), (13, 14, 15))),
        ])

    def test_cpu_other_models_and_disabled_modes_stay_at_three_frames(self):
        for configuration in (
            {'cuda': False},
            {'model_name': 'TrackNetV2 MSPE + CSCA'},
            {'six_frame_batch': False},
            {'fast_analysis': False},
        ):
            with self.subTest(configuration=configuration):
                _, calls, _ = self.simulate(7, **configuration)
                self.assertEqual(calls, [
                    (False, ((1, 2, 3),)),
                    (False, ((4, 5, 6),)),
                    (False, ((7,),)),
                ])

    def test_cuda_oom_retries_the_same_frames_then_stays_with_three_frames(self):
        _, calls, _ = self.simulate(
            13, grouped_error=desktop.torch.cuda.OutOfMemoryError('Test memory exhaustion'),
        )
        self.assertEqual(calls, [
            (True, ((1, 2, 3), (4, 5, 6))),
            (False, ((1, 2, 3),)),
            (False, ((4, 5, 6),)),
            (False, ((7, 8, 9),)),
            (False, ((10, 11, 12),)),
            (False, ((13,),)),
        ])

    def test_unexpected_batch_error_is_not_retried_or_reported_as_success(self):
        run, calls, analyzed = self.simulate(6, grouped_error=RuntimeError('Test detector failure'))
        self.assertEqual(run.last_error, 'Test detector failure')
        self.assertEqual(calls, [(True, ((1, 2, 3), (4, 5, 6)))])
        self.assertEqual(analyzed, [])


if __name__ == '__main__':
    unittest.main()
