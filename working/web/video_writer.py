"""Stream rendered BGR frames directly to a browser-compatible H.264 file."""
from functools import lru_cache
import os
import shutil
import subprocess
import tempfile

import cv2
import numpy as np


@lru_cache(maxsize=1)
def _nvenc_available():
    if not shutil.which('ffmpeg'):
        return False
    try:
        result = subprocess.run(
            ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-f', 'lavfi',
             '-i', 'color=c=black:s=320x240:r=25', '-frames:v', '1',
             '-c:v', 'h264_nvenc', '-f', 'null', '-'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
        )
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


class H264Writer:
    needs_transcode = False

    def __init__(self, path, fps, size, encoder):
        self.size = tuple(size)
        self.encoder = encoder
        self._closed = False
        self._error_log = tempfile.TemporaryFile()
        options = (['-c:v', 'h264_nvenc', '-preset', 'p4', '-rc', 'vbr', '-cq', '22', '-b:v', '0']
                   if encoder == 'h264_nvenc' else
                   ['-c:v', 'libx264', '-preset', 'veryfast', '-crf', '22'])
        command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                   '-f', 'rawvideo', '-pixel_format', 'bgr24',
                   '-video_size', f'{size[0]}x{size[1]}', '-framerate', str(fps),
                   '-i', 'pipe:0', '-an', *options, '-pix_fmt', 'yuv420p',
                   '-movflags', '+faststart', str(path)]
        try:
            self._process = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                stderr=self._error_log,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
            )
        except BaseException:
            self._error_log.close()
            raise

    def isOpened(self):
        return not self._closed and self._process.poll() is None

    def write(self, frame):
        if self._closed:
            raise ValueError('Video writer is already closed.')
        if frame.shape != (self.size[1], self.size[0], 3) or frame.dtype != np.uint8:
            raise ValueError('Video frames must retain their original uint8 BGR size.')
        try:
            self._process.stdin.write(memoryview(np.ascontiguousarray(frame)).cast('B'))
        except (OSError, BrokenPipeError) as error:
            raise RuntimeError('H.264 encoder stopped while writing video.') from error

    def release(self):
        if self._closed:
            return
        self._closed = True
        try:
            try:
                self._process.stdin.close()
            except (OSError, BrokenPipeError):
                pass
            try:
                result = self._process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
                raise RuntimeError('H.264 encoder timed out.')
            if result != 0:
                self._error_log.seek(0)
                detail = self._error_log.read().decode('utf-8', errors='replace')[-2000:]
                raise RuntimeError(f'H.264 encoding failed: {detail}')
        finally:
            self._error_log.close()


class LegacyWriter:
    needs_transcode = True

    def __init__(self, path, fps, size):
        self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    def isOpened(self):
        return self.writer.isOpened()

    def write(self, frame):
        self.writer.write(frame)

    def release(self):
        self.writer.release()


def create_video_writer(path, fps, size):
    encoder = os.environ.get('TENNIS_VIDEO_ENCODER', 'auto').strip().lower()
    if encoder not in {'auto', 'h264_nvenc', 'libx264', 'legacy'}:
        raise ValueError('TENNIS_VIDEO_ENCODER must be auto, h264_nvenc, libx264, or legacy.')
    if encoder == 'legacy' or not shutil.which('ffmpeg'):
        return LegacyWriter(path, fps, size)
    if encoder == 'auto':
        encoder = 'h264_nvenc' if min(size) >= 160 and _nvenc_available() else 'libx264'
    return H264Writer(path, fps, size, encoder)
