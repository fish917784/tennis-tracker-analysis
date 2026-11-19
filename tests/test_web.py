import importlib
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = types.ModuleType('fixed_web_tests')
PACKAGE.__path__ = [str(ROOT / 'working/web')]
sys.modules[PACKAGE.__name__] = PACKAGE
app = importlib.import_module('fixed_web_tests.app')
geometry = importlib.import_module('fixed_web_tests.court_detector')


class NoCourt:
    def __init__(self, fps):
        pass
    def update(self, frame, frame_idx):
        return dict(H=None, H_inv=None, court_poly=None, corners=None, method='none',
                    calibrated=False, segment_id=0)


class WebTests(unittest.TestCase):
    def test_unlimited_analysis_does_not_wait(self):
        clock, sleeper = Mock(), Mock()
        pacer = app.SourceFramePacer(59.94, clock=clock, sleeper=sleeper)
        for _ in range(10):
            pacer.wait()
        sleeper.assert_not_called()
        clock.assert_not_called()

    def test_source_pacing_has_no_catch_up_burst_when_behind(self):
        now = [0.0]
        sleeps = []
        def sleep(delay):
            sleeps.append(delay)
            now[0] += delay
        pacer = app.SourceFramePacer(59.94, lambda: True, lambda: now[0], sleep)
        pacer.wait()
        pacer.wait()
        self.assertAlmostEqual(now[0], 1 / 59.94)
        now[0] += 1.0
        before = len(sleeps)
        pacer.wait()
        self.assertEqual(len(sleeps), before)
        previous = now[0]
        pacer.wait()
        self.assertAlmostEqual(now[0] - previous, 1 / 59.94)

    def test_live_switch_releases_pacing_and_restarts_from_current_time(self):
        now, limited = [0.0], [True]
        def sleep(delay):
            now[0] += delay
            limited[0] = False
        pacer = app.SourceFramePacer(5.0, lambda: limited[0], lambda: now[0], sleep)
        pacer.wait()
        pacer.wait()
        self.assertLessEqual(now[0], .025)
        self.assertIsNone(pacer.next_frame)
        now[0] = 5.0
        limited[0] = True
        pacer.wait()
        self.assertAlmostEqual(pacer.next_frame, 5.2)

    def test_recent_analysis_fps_resets_after_toggle(self):
        now = [0.0]
        meter = app.AnalysisFrameRate(lambda: now[0])
        meter.record(False)
        for index in range(1, 101):
            now[0] = index * .001
            meter.record(False)
        self.assertAlmostEqual(meter.fps, 1000.)
        now[0] = .2
        self.assertEqual(meter.record(True), 0.)
        now[0] += 1 / 59.94
        self.assertAlmostEqual(meter.record(True), 59.94)
        for _ in range(100):
            now[0] += .04
            meter.record(True)
        self.assertAlmostEqual(meter.fps, 25.)
        self.assertLessEqual(len(meter.timestamps), 27)

    def test_imports_and_outputs_stay_in_copy(self):
        model_module = sys.modules[app.TrackNetV2B0MSPECSCA.__module__]
        model_source = Path(model_module.__file__).resolve()
        for path in [model_source, app.UPLOAD_DIR, app.OUTPUT_DIR, geometry.CODEX_COURT_DIR]:
            self.assertTrue(path.is_relative_to(ROOT / 'working'))
        self.assertIn('_tennis_court_core', geometry.CourtTracker.__module__)

    def test_failed_geometry_returns_no_homography(self):
        capture = Mock()
        capture.get.return_value = 1
        with patch.object(geometry, '_CODEX_AVAILABLE', False):
            H, H_inv, _, _, method = geometry.get_court_geometry(capture, 320, 180)
        self.assertIsNone(H)
        self.assertIsNone(H_inv)
        self.assertEqual(method, 'none')

    def test_court_loss_and_cut_start_new_segments(self):
        detector = Mock(image_to_court=np.eye(3), court_to_image=np.eye(3),
                        quad=np.array([[0.,0.],[10.,0.],[10.,20.],[0.,20.]]), score=0.99)
        tracker = Mock(cuts=0)
        tracker.update.side_effect = [detector, None, detector]
        live = geometry.LiveCourtGeometry(30)
        live.tracker = tracker
        frame = np.zeros((180, 320, 3), np.uint8)
        first = live.update(frame, 0)
        second = live.update(frame, 1)
        tracker.cuts = 1
        third = live.update(frame, 2)
        self.assertTrue(first['calibrated'])
        self.assertFalse(second['calibrated'])
        self.assertTrue(third['calibrated'])
        self.assertEqual(len({s['segment_id'] for s in [first, second, third]}), 3)

    def test_small_videos_render_and_unverified_court_has_no_bounces(self):
        for width, height in [(320, 180), (400, 224)]:
            with self.subTest(size=(width, height)), tempfile.TemporaryDirectory() as temp:
                source, target = Path(temp) / 'input.mp4', Path(temp) / 'output.mp4'
                writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'mp4v'), 30., (width, height))
                self.assertTrue(writer.isOpened())
                for _ in range(7):
                    writer.write(np.zeros((height, width, 3), np.uint8))
                writer.release()
                def heatmaps(frames):
                    return np.zeros((len(frames), 288, 512), np.float32)
                with patch.object(app, 'LiveCourtGeometry', NoCourt), \
                     patch.object(app, 'predict_triplet_batch', lambda groups: [heatmaps(group) for group in groups]), \
                     patch.object(app.subprocess, 'run', return_value=Mock(returncode=1)):
                    bounces, info = app.process_video(str(source), str(target))
                self.assertEqual(bounces, [])
                self.assertEqual(info['method'], 'none')
                capture = cv2.VideoCapture(str(target))
                self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 7)
                self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), width)
                self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), height)
                capture.release()

    def test_analysis_only_replays_cached_candidates_in_source_order(self):
        with tempfile.TemporaryDirectory() as temp:
            source = str(Path(temp) / 'input.mp4')
            writer = cv2.VideoWriter(source, cv2.VideoWriter_fourcc(*'mp4v'), 30., (320, 180))
            self.assertTrue(writer.isOpened())
            for _ in range(7):
                writer.write(np.zeros((180, 320, 3), np.uint8))
            writer.release()
            cached = Mock(return_value=[])
            result = Mock()
            with patch.object(app, 'LiveCourtGeometry', NoCourt), \
                 patch.object(app, 'predict_triplet_batch') as predict, \
                 patch.object(app, 'create_video_writer') as render:
                bounces, court = app.process_video(source, None, analysis_result_cb=result, candidate_source=cached)
            self.assertEqual([entry.args[0] for entry in cached.call_args_list], list(range(7)))
            self.assertEqual(len(result.call_args.args[1]), 7)
            self.assertEqual(bounces, [])
            self.assertEqual(court['method'], 'none')
            predict.assert_not_called()
            render.assert_not_called()

    def test_core_verified_score_079_keeps_web_court_and_segment(self):
        detection = Mock(image_to_court=np.eye(3), court_to_image=np.eye(3),
                         quad=np.array([[0.,0.],[10.,0.],[10.,20.],[0.,20.]]), score=.81)
        tracker = Mock(cuts=0)
        tracker.update.return_value = detection
        live = geometry.LiveCourtGeometry(25)
        live.tracker = tracker
        frame = np.zeros((180, 320, 3), np.uint8)
        first = live.update(frame, 0)
        detection.score = .79
        second = live.update(frame, 1)
        self.assertTrue(second['calibrated'])
        self.assertEqual(first['segment_id'], second['segment_id'])

    def test_short_loss_holds_only_display_and_keeps_analysis_segments_separate(self):
        detection = Mock(image_to_court=np.eye(3), court_to_image=np.eye(3),
                         quad=np.array([[0.,0.],[10.,0.],[10.,20.],[0.,20.]]), score=.9)
        tracker = Mock(cuts=0)
        tracker.update.side_effect = [detection, None, detection]
        live = geometry.LiveCourtGeometry(59.94)
        live.tracker = tracker
        frame = np.zeros((180, 320, 3), np.uint8)
        first, lost, recovered = [live.update(frame, index) for index in [10, 11, 15]]
        self.assertTrue(lost['displayed'])
        self.assertTrue(lost['verifying'])
        self.assertFalse(lost['calibrated'])
        self.assertIsNone(lost['H'])
        self.assertIsNone(lost['H_inv'])
        self.assertIsNone(lost['court_poly'])
        self.assertEqual(len({state['segment_id'] for state in [first, lost, recovered]}), 3)
        self.assertEqual(first['display_segment_id'], recovered['display_segment_id'])

    def test_display_grace_expires_in_source_frames(self):
        detection = Mock(image_to_court=np.eye(3), court_to_image=np.eye(3),
                         quad=np.array([[0.,0.],[10.,0.],[10.,20.],[0.,20.]]), score=.9)
        tracker = Mock(cuts=0)
        tracker.update.side_effect = [detection, None, None, None]
        live = geometry.LiveCourtGeometry(59.94)
        live.tracker = tracker
        frame = np.zeros((180, 320, 3), np.uint8)
        first = live.update(frame, 10)
        live.update(frame, 11)
        self.assertTrue(live.update(frame, 37)['displayed'])
        expired = live.update(frame, 38)
        self.assertFalse(expired['displayed'])
        self.assertFalse(expired['verifying'])
        self.assertIsNone(expired['display_corners'])
        self.assertNotEqual(first['display_segment_id'], expired['display_segment_id'])

    def test_cut_immediately_discards_display_during_grace(self):
        detection = Mock(image_to_court=np.eye(3), court_to_image=np.eye(3),
                         quad=np.array([[0.,0.],[10.,0.],[10.,20.],[0.,20.]]), score=.9)
        tracker = Mock(cuts=0)
        tracker.update.side_effect = [detection, None, None]
        live = geometry.LiveCourtGeometry(59.94)
        live.tracker = tracker
        frame = np.zeros((180, 320, 3), np.uint8)
        first = live.update(frame, 10)
        self.assertTrue(live.update(frame, 11)['displayed'])
        tracker.cuts = 1
        cut = live.update(frame, 12)
        self.assertFalse(cut['displayed'])
        self.assertNotEqual(first['display_segment_id'], cut['display_segment_id'])

    def test_render_preserves_history_during_grace_without_projection_and_keeps_source_fps(self):
        with tempfile.TemporaryDirectory() as temp:
            source, target = Path(temp) / 'input.mp4', Path(temp) / 'output.mp4'
            source_fps = 60000 / 1001
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'mp4v'), source_fps, (320, 180))
            self.assertTrue(writer.isOpened())
            for _ in range(7):
                writer.write(np.zeros((180, 320, 3), np.uint8))
            writer.release()
            detection = Mock(image_to_court=np.eye(3), court_to_image=np.eye(3),
                             quad=np.array([[0.,0.],[10.,0.],[10.,20.],[0.,20.]]), score=.9)
            core_tracker = Mock(cuts=0)
            core_tracker.update.side_effect = [detection, None, None, None, detection, detection, detection]
            live = geometry.LiveCourtGeometry(source_fps)
            live.tracker = core_tracker
            mini_calls = []
            actual_mini = app.create_mini_court_overlay
            def record_mini(*args, **kwargs):
                mini_calls.append(dict(kwargs, bounces=list(kwargs['bounces'])))
                return actual_mini(*args, **kwargs)
            def bounces(trajectory, H, H_inv, fps, court_states):
                for index in [1, 2, 3]:
                    self.assertFalse(court_states[index]['calibrated'])
                    self.assertIsNone(court_states[index]['H'])
                return [dict(frame=0, x=10, y=20, court_x=1., court_y=2., is_in=True, id=1)]
            output_writer = Mock(needs_transcode=False)
            output_writer.isOpened.return_value = True
            pacer = Mock()
            analysis_updates = []
            with patch.object(app, 'LiveCourtGeometry', return_value=live), \
                 patch.object(app, 'predict_triplet_batch', lambda groups: [np.zeros((len(group), 288, 512), np.float32) for group in groups]), \
                 patch.object(app, 'detect_court_bounces', side_effect=bounces), \
                 patch.object(app, 'create_mini_court_overlay', side_effect=record_mini), \
                 patch.object(app, 'project_to_court') as projection, \
                 patch.object(app, 'SourceFramePacer', return_value=pacer) as create_pacer, \
                 patch.object(app, 'create_video_writer', return_value=output_writer) as create_writer:
                app.process_video(str(source), str(target), limit_to_source_fps=lambda: True,
                                  analysis_status_cb=analysis_updates.append)
            self.assertAlmostEqual(create_writer.call_args.args[1], source_fps, places=2)
            self.assertAlmostEqual(create_pacer.call_args.args[0], source_fps, places=2)
            self.assertTrue(create_pacer.call_args.args[1]())
            self.assertEqual(pacer.wait.call_count, 7)  # Analysis only; export never waits.
            self.assertEqual([state['phase'] for state in analysis_updates], ['analysis', 'export'])
            self.assertEqual(analysis_updates[-1]['analysis_frames'], 7)
            self.assertGreater(analysis_updates[-1]['analysis_fps'], 0)
            self.assertEqual(output_writer.write.call_count, 7)
            projection.assert_not_called()
            for index in [1, 2, 3, 4]:
                self.assertTrue(mini_calls[index]['court_calibrated'])
                self.assertEqual(len(mini_calls[index]['bounces']), 1)
                self.assertIsNone(mini_calls[index]['current_m'])

    def test_http_errors_are_explicit(self):
        import asyncio
        import httpx
        async def check():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app.app), base_url='http://test') as client:
                result = await client.post('/upload', files={'file': ('bad.txt', b'not a video', 'text/plain')})
                self.assertEqual(result.status_code, 400)
                self.assertIn('detail', result.json())
                self.assertEqual((await client.get('/status/does-not-exist')).status_code, 404)
                self.assertEqual((await client.get('/')).status_code, 200)
        asyncio.run(asyncio.wait_for(check(), timeout=15))

    def test_speed_api_toggles_active_analysis_and_rejects_export_changes(self):
        import asyncio
        import httpx
        key = 'unit-speed-control'
        app.job_status[key] = dict(status='processing', phase='analysis', limit_to_source_fps=False,
                                   source_fps=59.94, analysis_fps=72.5, speed_revision=0)
        async def check():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app.app), base_url='http://test') as client:
                for enabled in [True, False]:
                    result = await client.post('/speed/' + key, json={'limit_to_source_fps': enabled})
                    self.assertEqual(result.status_code, 200)
                    self.assertEqual(result.json()['limit_to_source_fps'], enabled)
                    self.assertEqual((await client.get('/status/' + key)).json()['limit_to_source_fps'], enabled)
                invalid = await client.post('/speed/' + key, json={'limit_to_source_fps': 'yes'})
                self.assertEqual(invalid.status_code, 422)
                self.assertEqual(app.job_status[key]['speed_revision'], 2)
                app.job_status[key]['phase'] = 'export'
                self.assertEqual((await client.post('/speed/' + key, json={'limit_to_source_fps': True})).status_code, 409)
                self.assertEqual((await client.post('/speed/missing-job', json={'limit_to_source_fps': True})).status_code, 404)
        try:
            asyncio.run(asyncio.wait_for(check(), timeout=15))
        finally:
            app.job_status.pop(key, None)

    def test_uploaded_job_defaults_to_unlimited_and_reports_actual_source_and_analysis_fps(self):
        import asyncio
        import httpx
        expected_mode = [False]
        def fake_process(in_path, out_path, progress_cb=None, court_debug_path=None,
                         limit_to_source_fps=None, analysis_status_cb=None):
            self.assertEqual(limit_to_source_fps(), expected_mode[0])
            analysis_status_cb(dict(phase='analysis', source_fps=59.94, analysis_fps=0.0, analysis_frames=0))
            analysis_status_cb(dict(phase='export', analysis_fps=72.5, analysis_frames=120))
            return [], dict(method='none', corners=[], has_image=False)
        async def check():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app.app), base_url='http://test') as client:
                for enabled in [False, True]:
                    expected_mode[0] = enabled
                    data = {'limit_to_source_fps': 'true'} if enabled else {}
                    response = await client.post('/upload', data=data,
                                                 files={'file': ('clip.mp4', b'synthetic', 'video/mp4')})
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.json()['limit_to_source_fps'], enabled)
                    key = response.json()['job_id']
                    try:
                        state = (await client.get('/status/' + key)).json()
                        self.assertEqual(state['limit_to_source_fps'], enabled)
                        self.assertEqual(state['source_fps'], 59.94)
                        self.assertEqual(state['analysis_fps'], 72.5)
                        self.assertEqual(state['analysis_frames'], 120)
                        self.assertEqual(state['phase'], 'done')
                        self.assertFalse((Path(temp) / (key + '_in.mp4')).exists())
                    finally:
                        app.job_status.pop(key, None)
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(app, 'UPLOAD_DIR', Path(temp)), patch.object(app, 'OUTPUT_DIR', Path(temp)), \
             patch.object(app, 'process_video', side_effect=fake_process):
            asyncio.run(asyncio.wait_for(check(), timeout=15))


if __name__ == '__main__':
    unittest.main()
