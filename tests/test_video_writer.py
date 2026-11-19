import importlib.util
from pathlib import Path
import subprocess
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('writer_test_module', ROOT / 'working/web/video_writer.py')
writer_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(writer_module)


class WriterTests(unittest.TestCase):
    def process(self, wait_result=0):
        process = Mock()
        process.poll.return_value = None
        process.wait.return_value = wait_result
        return process

    def test_direct_writer_preserves_bgr_frames_size_and_fps(self):
        process = self.process()
        with patch.object(writer_module.subprocess, 'Popen', return_value=process) as spawn:
            writer = writer_module.H264Writer('unused.mp4', 25., (320, 180), 'libx264')
        frame = np.zeros((180, 320, 3), np.uint8)
        writer.write(frame)
        self.assertEqual(process.stdin.write.call_args.args[0].nbytes, frame.nbytes)
        command = spawn.call_args.args[0]
        self.assertIn('320x180', command)
        self.assertEqual(command[command.index('-framerate') + 1], '25.0')
        self.assertFalse(writer.needs_transcode)
        writer.release()
        writer.release()
        process.wait.assert_called_once()

    def test_wrong_frame_size_is_rejected_before_encoding(self):
        process = self.process()
        with patch.object(writer_module.subprocess, 'Popen', return_value=process):
            writer = writer_module.H264Writer('unused.mp4', 25., (320, 180), 'libx264')
        with self.assertRaises(ValueError):
            writer.write(np.zeros((90, 160, 3), np.uint8))
        process.stdin.write.assert_not_called()
        writer.release()

    def test_failed_encoder_is_reported(self):
        process = self.process(wait_result=1)
        with patch.object(writer_module.subprocess, 'Popen', return_value=process):
            writer = writer_module.H264Writer('unused.mp4', 25., (320, 180), 'libx264')
        writer._error_log.write(b'encoder test failure')
        with self.assertRaisesRegex(RuntimeError, 'encoder test failure'):
            writer.release()

    def test_timeout_stops_only_owned_encoder(self):
        process = self.process()
        process.wait.side_effect = [subprocess.TimeoutExpired('ffmpeg', 120), 0]
        with patch.object(writer_module.subprocess, 'Popen', return_value=process):
            writer = writer_module.H264Writer('unused.mp4', 25., (320, 180), 'libx264')
        with self.assertRaisesRegex(RuntimeError, 'timed out'):
            writer.release()
        process.kill.assert_called_once()


if __name__ == '__main__':
    unittest.main()
