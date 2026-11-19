"""Play a file on its source clock while a separate worker analyzes it."""

from collections import deque
import math
import pickle
from struct import Struct, error as StructError
from tempfile import TemporaryFile
from threading import Condition, Lock, Thread

import cv2


_RECORD_HEADER = Struct("<QI")
_DECODE_AHEAD = 4


class FilePlayback:
    def __init__(self, source):
        self.source = source
        self.next_at = None
        self.finished = False
        self.error = None
        self._stopped = False
        self._origin = None
        self._waiting_for_analysis = False
        self._frames = deque()
        self._condition = Condition()
        self._reader_done = False
        self._last_decoded = 0
        self._reader = None
        self._spool_lock = Lock()
        try:
            self._spool = TemporaryFile(mode="w+b")
        except OSError as exc:
            self._spool = None
            self.error = f"無法建立預覽標記暫存：{exc}"
        self._write_offset = 0
        self._read_offset = 0
        self._played_through = 0

    def start(self):
        with self._condition:
            if self._reader is None and not self._stopped:
                self._reader = Thread(target=self._decode, daemon=True)
                self._reader.start()

    def _decode(self):
        capture = None
        try:
            capture = cv2.VideoCapture(self.source)
            if not capture.isOpened():
                raise OSError("無法開啟影片預覽")
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._stopped or len(self._frames) < _DECODE_AHEAD)
                    if self._stopped:
                        break
                ok, frame = capture.read()
                if not ok:
                    if self._last_decoded == 0 and not self._stopped:
                        self.error = "無法讀取影片預覽影格"
                    break
                with self._condition:
                    if self._stopped:
                        break
                    self._last_decoded += 1
                    self._frames.append((self._last_decoded, frame))
                    self._condition.notify_all()
        except (OSError, cv2.error) as exc:
            self.error = str(exc)
        finally:
            try:
                if capture is not None:
                    capture.release()
            finally:
                with self._condition:
                    self._reader_done = True
                    self._condition.notify_all()

    def publish(self, index, overlay):
        """Write snapshots in source order, skipping frames already played."""
        with self._spool_lock:
            if self.finished or self._stopped or self._spool is None or index <= self._played_through:
                return
            try:
                payload = pickle.dumps(overlay, protocol=pickle.HIGHEST_PROTOCOL)
                self._spool.seek(self._write_offset)
                self._spool.write(_RECORD_HEADER.pack(index, len(payload)))
                self._spool.write(payload)
                self._write_offset = self._spool.tell()
            except (OSError, pickle.PicklingError) as exc:
                self.error = f"無法儲存預覽標記：{exc}"
                self._close_spool()

    def _overlay_for(self, index):
        with self._spool_lock:
            self._played_through = index
            if self._spool is None:
                return None
            try:
                while self._read_offset < self._write_offset:
                    self._spool.seek(self._read_offset)
                    source_index, length = _RECORD_HEADER.unpack(self._spool.read(_RECORD_HEADER.size))
                    if source_index > index:
                        break
                    self._read_offset += _RECORD_HEADER.size + length
                    if source_index == index:
                        return pickle.loads(self._spool.read(length))
            except (OSError, EOFError, StructError, pickle.UnpicklingError) as exc:
                self.error = f"無法讀取預覽標記：{exc}"
                self._close_spool()
            return None

    def take(self, now, source_fps, analyzed_through, *, analysis_finished=False):
        """Play up to source speed and hold when the next matching result is pending."""
        selected = None
        with self._condition:
            if self.finished:
                return None
            if self._origin is None:
                if not self._frames:
                    if self._reader_done:
                        self.finished = True
                        self.next_at = None
                    return None
                self._origin = now
            slot = math.floor((now - self._origin) * source_fps + 1e-9)
            target = slot + 1
            if self._reader_done and not self._frames and target > self._last_decoded:
                self.finished = True
                self.next_at = None
                return None
            if target > analyzed_through or self._waiting_for_analysis:
                if self._frames and self._frames[0][0] <= min(target, analyzed_through):
                    # Resume at the first queued result; waiting time never
                    # becomes playback debt that would skip ahead.
                    target = self._frames[0][0]
                    slot = target - 1
                    self._origin = now - slot / source_fps
                    self._waiting_for_analysis = False
                else:
                    if analysis_finished and self._frames and self._frames[0][0] > analyzed_through:
                        self.error = self.error or "部分預覽影格未完成分析"
                        self.finished = True
                    self._waiting_for_analysis = True
                    self.next_at = None
                    return None
            self.next_at = self._origin + target / source_fps
            current_slot_at = self._origin + slot / source_fps
            if (self._frames and self._frames[0][0] == target - 1
                    and self._played_through == target - 2
                    and now - current_slot_at <= min(.004, .25 / source_fps)
                    and not (self._reader_done and target >= self._last_decoded)):
                # A short UI delay can leave one frame ready just past its
                # slot. Draw it and recheck the current slot without rebasing.
                selected = self._frames.popleft()
                self.next_at = now
            else:
                while self._frames and self._frames[0][0] <= target:
                    selected = self._frames.popleft()
            if selected is not None:
                self._condition.notify_all()
        if selected is None:
            return None
        index, frame = selected
        return index, frame, self._overlay_for(index)

    def stop(self):
        with self._condition:
            self._stopped = True
            self.finished = True
            self.next_at = None
            self._frames.clear()
            self._condition.notify_all()

    def _close_spool(self):
        if self._spool is not None:
            spool, self._spool = self._spool, None
            try:
                spool.close()
            except OSError as exc:
                if self.error is None:
                    self.error = f"無法關閉預覽標記暫存：{exc}"

    def close(self):
        self.stop()
        if self._reader is not None:
            # The decoder owns the capture and releases it on its own thread.
            self._reader.join(timeout=0.25)
        with self._spool_lock:
            self._close_spool()
