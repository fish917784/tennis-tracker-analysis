import sys
from pathlib import Path
from threading import Event
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "working/desktop"))
import file_playback


class FakeCapture:
    def __init__(self, count, *, opened=True):
        self.count = count
        self.opened = opened
        self.read_count = 0
        self.released = Event()

    def isOpened(self):
        return self.opened

    def read(self):
        if self.read_count == self.count:
            return False, None
        self.read_count += 1
        return True, np.full((2, 2, 3), self.read_count, dtype=np.uint8)

    def release(self):
        self.released.set()


class FilePlaybackTests(unittest.TestCase):
    def playback(self, count, *, opened=True):
        class Source(str):
            pass

        source = Source("test.mp4")
        capture = FakeCapture(count, opened=opened)
        patcher = patch.object(file_playback.cv2, "VideoCapture", return_value=capture)
        factory = patcher.start()
        self.addCleanup(patcher.stop)
        playback = file_playback.FilePlayback(source)
        self.addCleanup(playback.close)
        playback.start()
        self.wait_for(playback, lambda: bool(playback._frames) or playback._reader_done)
        self.assertIs(factory.call_args.args[0], source)
        return playback, capture

    def wait_for(self, playback, predicate):
        with playback._condition:
            self.assertTrue(playback._condition.wait_for(predicate, timeout=2))

    def assert_frame(self, shown, index, overlay):
        self.assertIsNotNone(shown)
        actual_index, frame, actual_overlay = shown
        self.assertEqual(actual_index, index)
        self.assertTrue((frame == index).all())
        self.assertEqual(actual_overlay, overlay)

    def test_faster_analysis_does_not_advance_the_movie_clock(self):
        playback, capture = self.playback(6)
        for index in range(1, 7):
            playback.publish(index, {"ball": index})
        self.wait_for(playback, lambda: len(playback._frames) == 4)
        self.assertEqual(capture.read_count, 4)

        self.assert_frame(playback.take(20.0, 4.0, 6), 1, {"ball": 1})
        self.assertIsNone(playback.take(20.2, 4.0, 6))
        self.assert_frame(playback.take(20.76, 4.0, 6), 4, {"ball": 4})
        self.wait_for(playback, lambda: playback._reader_done)
        self.assert_frame(playback.take(21.25, 4.0, 6, analysis_finished=True), 6, {"ball": 6})
        self.assertIsNone(playback.take(21.49, 4.0, 6, analysis_finished=True))
        self.assertFalse(playback.finished)
        self.assertIsNone(playback.take(21.50, 4.0, 6, analysis_finished=True))
        self.assertTrue(playback.finished)
        self.assertIsNone(playback.next_at)

    def test_slow_analysis_holds_and_resumes_in_order_at_source_cadence(self):
        playback, _ = self.playback(4)
        self.wait_for(playback, lambda: len(playback._frames) == 4)
        playback.publish(1, "first")
        self.assert_frame(playback.take(10.0, 10.0, 1), 1, "first")
        self.assertIsNone(playback.take(10.1, 10.0, 1))
        self.assertIsNone(playback.take(10.8, 10.0, 1))
        self.assertFalse(playback.finished)
        self.assertIsNone(playback.next_at)
        for index in (2, 3, 4):
            playback.publish(index, {"ball": index})
        self.wait_for(playback, lambda: playback._reader_done)
        self.assert_frame(playback.take(10.81, 10.0, 4, analysis_finished=True), 2, {"ball": 2})
        self.assertIsNone(playback.take(10.90, 10.0, 4, analysis_finished=True))
        self.assert_frame(playback.take(10.91, 10.0, 4, analysis_finished=True), 3, {"ball": 3})
        self.assert_frame(playback.take(11.01, 10.0, 4, analysis_finished=True), 4, {"ball": 4})
        self.assertIsNone(playback.take(11.11, 10.0, 4, analysis_finished=True))
        self.assertTrue(playback.finished)

    def test_analysis_ending_early_releases_an_unmatched_decoder_tail(self):
        playback, _ = self.playback(3)
        self.wait_for(playback, lambda: playback._reader_done)
        playback.publish(1, "first")
        self.assert_frame(playback.take(10., 2., 1, analysis_finished=True), 1, "first")
        self.assertIsNone(playback.take(10.50, 2., 1, analysis_finished=True))
        self.assertTrue(playback.finished)
        self.assertTrue(playback.error)

    def test_short_ui_delay_can_draw_one_missed_frame_without_rebasing(self):
        playback, _ = self.playback(8)
        self.wait_for(playback, lambda: len(playback._frames) == 4)
        self.assert_frame(playback.take(10., 60., 8), 1, None)
        now = 10. + 2 / 60. + .002
        self.assert_frame(playback.take(now, 60., 8), 2, None)
        self.assertEqual(playback.next_at, now)
        self.assert_frame(playback.take(now + .002, 60., 8), 3, None)
        self.assertAlmostEqual(playback.next_at, 10. + 3 / 60.)

    def test_fractional_rate_starts_with_the_first_ready_frame(self):
        playback = file_playback.FilePlayback("wait.mp4")
        self.addCleanup(playback.close)
        self.assertIsNone(playback.take(10.0, 60000 / 1001, 2))
        playback._frames.extend((i, np.full((2, 2, 3), i, np.uint8)) for i in (1, 2))
        self.assert_frame(playback.take(15.0, 60000 / 1001, 2), 1, None)
        self.assertIsNone(playback.take(15.01, 60000 / 1001, 2))
        self.assert_frame(playback.take(15.0 + 1001 / 60000, 60000 / 1001, 2), 2, None)

    def test_failed_decoder_finishes_and_releases_capture(self):
        playback, capture = self.playback(0, opened=False)
        self.assertIsNone(playback.take(10.0, 30.0, 0, analysis_finished=True))
        self.assertTrue(playback.finished)
        self.assertTrue(capture.released.is_set())
        self.assertTrue(playback.error)

    def test_unreadable_overlay_keeps_source_playback_running(self):
        playback, _ = self.playback(2)
        self.wait_for(playback, lambda: playback._reader_done)
        playback.publish(1, "overlay")
        playback.publish(2, "overlay")
        with playback._spool_lock:
            playback._spool.truncate(3)
        self.assert_frame(playback.take(10., 2., 2), 1, None)
        self.assert_frame(playback.take(10.5, 2., 2), 2, None)
        self.assertTrue(playback.error)

    def test_stop_unblocks_decoder_and_close_releases_spool(self):
        playback, capture = self.playback(20)
        self.wait_for(playback, lambda: len(playback._frames) == 4)
        playback.publish(10, "ahead")
        playback.stop()
        playback.close()
        playback.close()
        self.assertTrue(capture.released.is_set())
        self.assertIsNone(playback._spool)
        playback.publish(11, "after close")
        self.assertIsNone(playback.take(12.0, 30.0, 11, analysis_finished=True))


if __name__ == "__main__":
    unittest.main()
