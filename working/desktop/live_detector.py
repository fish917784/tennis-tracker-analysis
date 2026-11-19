"""Desktop inference for the EffB0 and TrackNetV2 checkpoints."""

from collections import deque
from dataclasses import dataclass
from math import isfinite
import os
from pathlib import Path
from queue import Empty, SimpleQueue
from threading import Condition, Event, Thread
from time import perf_counter

import cv2
import numpy as np
import torch

from inference_runtime import predict_heatmaps, predict_triplets


ROOT = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get("TENNIS_WEIGHTS_DIR", ROOT / "model")).expanduser()

V2_MODEL = "TrackNetV2 MSPE + CSCA"
WEIGHTS = {
    "EfficientNetV2-B0": "EfficientNetV2B0_MSPE_DecoderCSCA_best_epoch89.pt",
    V2_MODEL: "TrackNetV2_MSPE_CSCA_best_epoch67.pt",
}
DEFAULT_MODEL = next(iter(WEIGHTS))
WIDTH, HEIGHT = 512, 288


@dataclass(frozen=True)
class ModelChoice:
    name: str
    path: Path

    @classmethod
    def preset(cls, name):
        return cls(name, MODEL_DIR / WEIGHTS[name])


class Detector:
    def __init__(self, choice, device):
        if choice.name == "EfficientNetV2-B0":
            from tracknet_v2_b0_mspe_csca import TrackNetV2B0MSPECSCA

            model = TrackNetV2B0MSPECSCA()
        elif choice.name == V2_MODEL:
            from trajectorynet_pp import TrajectoryNetPP

            model = TrajectoryNetPP()
        else:
            raise ValueError(f"不支援的模型：{choice.name}")

        checkpoint = torch.load(choice.path, map_location="cpu", weights_only=True)
        state = checkpoint.get("model", checkpoint.get("net", checkpoint))
        model.load_state_dict(state, strict=True)
        self.device = device
        self.choice = choice
        self.model = model.eval().to(device, memory_format=torch.channels_last)
        self.warmup()

    @torch.inference_mode()
    def warmup(self):
        blank = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        output = predict_heatmaps(self.model, [blank] * 3, self.device)
        if output.shape != (3, HEIGHT, WIDTH) or not np.isfinite(output).all():
            raise ValueError("模型必須回傳三張有效的 288 × 512 熱力圖。")

    @torch.inference_mode()
    def predict(self, frames, realtime=False):
        # Playback pacing remains controlled by the caller. Inference precision
        # is shared by desktop and Web and can be forced to FP32 via the env.
        return predict_heatmaps(self.model, frames, self.device)

    def predict_batch(self, frame_groups):
        return predict_triplets(self.model, frame_groups, self.device)


def mark_ball(frame, heatmap):
    mask = (heatmap > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    rectangles = [cv2.boundingRect(contour) for contour in contours]
    x, y, w, h = max(rectangles, key=lambda rect: rect[2] * rect[3])
    height, width = frame.shape[:2]
    px = min(width - 1, int((x + w / 2) * width / WIDTH))
    py = min(height - 1, int((y + h / 2) * height / HEIGHT))
    radius = max(6, round(min(width, height) / 90))
    cv2.circle(frame, (px, py), radius + 1, (14, 18, 24), 4, cv2.LINE_AA)
    cv2.circle(frame, (px, py), radius, (80, 243, 197), 2, cv2.LINE_AA)
    cv2.circle(frame, (px, py), 2, (80, 243, 197), -1, cv2.LINE_AA)
    return px, py


class DetectionRun(Thread):
    def __init__(self, source, choice, realtime=False):
        super().__init__(daemon=True)
        self.source = source
        self.requests = SimpleQueue()
        self.requests.put(choice)
        self.events = SimpleQueue()
        self.stop = Event()
        self.realtime = Event()
        self.set_realtime(realtime)
        self.previews = deque(maxlen=9)
        self.skipped_frames = 0
        self.source_fps = None
        self.camera = None

    def set_realtime(self, enabled):
        if enabled:
            self.realtime.set()
        else:
            self.realtime.clear()

    @property
    def input_fps(self):
        return self.camera.fps if self.camera is not None else self.source_fps

    def detect_camera(self, detector, device):
        self.camera = CameraCapture(self.source)
        self.camera.start()
        deadline = None
        while not self.stop.is_set():
            detector = self.update_model(detector, device)
            batch = self.camera.take_triplet(self.stop)
            if not batch:
                break
            frames = [frame for _, _, frame in batch]
            started = perf_counter()
            heatmaps = detector.predict(frames, realtime=True)
            ready = perf_counter()
            inference_ms = (ready - started) * 1000
            first_time = batch[0][1]
            # Use the measured capture spacing even when the driver reports an incorrect FPS.
            period = max(1 / 240, min(1, (batch[-1][1] - first_time) / 2))
            deadline = max(deadline or ready, ready - period)
            for (index, captured_at, frame), heatmap in zip(batch, heatmaps):
                point = mark_ball(frame, heatmap)
                preview = (index, detector.choice.name, frame, point, inference_ms)
                self.previews.append((deadline + captured_at - first_time, preview))
            deadline += batch[-1][1] - first_time + period

    def update_model(self, detector, device):
        choice = None
        try:
            while True:
                choice = self.requests.get_nowait()
        except Empty:
            pass
        if choice is None or (detector and choice == detector.choice):
            return detector

        self.events.put(("status", f"正在載入 {choice.name}…"))
        try:
            candidate = Detector(choice, device)
        except Exception as exc:
            if detector is None:
                raise
            self.events.put(("switch_error", f"切換失敗，繼續使用 {detector.choice.name}。\n\n{exc}"))
            return detector
        self.events.put(("active", choice))
        return candidate

    def run(self):
        capture = None
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device.type == "cuda":
                torch.set_float32_matmul_precision("high")
                hardware = torch.cuda.get_device_name(device)
            else:
                torch.set_num_threads(min(torch.get_num_threads(), 4))
                hardware = "CPU"
            self.events.put(("device", hardware))
            detector = self.update_model(None, device)
            if self.stop.is_set():
                return

            if isinstance(self.source, int):
                self.detect_camera(detector, device)
                return

            capture = cv2.VideoCapture(self.source)
            if not capture.isOpened():
                raise OSError(f"無法開啟影像來源（{self.source}）。")

            fps = capture.get(cv2.CAP_PROP_FPS)
            self.source_fps = fps if 1 <= fps <= 240 else None
            frame_period = 1 / (self.source_fps or 30)
            reported_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            total_frames = int(reported_frames) if isfinite(reported_frames) and reported_frames > 0 else 0
            deadline = None
            index = 0
            while not self.stop.is_set():
                if frame_period and deadline is not None:
                    # Infer the next triplet while the previous one is still on screen.
                    lead = 3 * frame_period
                    if self.stop.wait(max(0, deadline - lead - perf_counter())):
                        return
                detector = self.update_model(detector, device)
                realtime = frame_period > 0 and self.realtime.is_set()
                if realtime and index > 0:
                    # Retain up to two late frames and read consecutive triplets for TrackNet.
                    late_frames = int((perf_counter() - deadline) / frame_period)
                    skip = max(0, late_frames - 2)
                    if total_frames:
                        skip = min(skip, max(0, total_frames - index - 3))
                    for _ in range(skip):
                        if self.stop.is_set():
                            return
                        if not capture.grab():
                            break
                        index += 1
                        self.skipped_frames += 1
                        deadline += frame_period

                frames = []
                for _ in range(3):
                    if self.stop.is_set():
                        return
                    ok, frame = capture.read()
                    if not ok:
                        break
                    frames.append(frame)
                if not frames:
                    if index == 0:
                        raise OSError("無法讀取影像，請確認攝影機或影片可正常播放。")
                    if deadline is not None and self.stop.wait(max(0, deadline - perf_counter())):
                        return
                    self.events.put(("status", "影片播放完畢"))
                    break

                started = perf_counter()
                heatmaps = detector.predict(frames, realtime=realtime)
                ready = perf_counter()
                inference_ms = (ready - started) * 1000
                if deadline is None or not realtime:
                    deadline = max(deadline or ready, ready)
                for frame, heatmap in zip(frames, heatmaps):
                    index += 1
                    point = mark_ball(frame, heatmap)
                    preview = (index, detector.choice.name, frame, point, inference_ms)
                    self.previews.append((deadline, preview))
                    deadline += frame_period
        except Exception as exc:
            self.events.put(("error", str(exc)))
        finally:
            if capture is not None:
                capture.release()
            if self.camera is not None:
                self.camera.stop.set()
                self.camera.join()


class CameraCapture(Thread):
    def __init__(self, index):
        super().__init__(daemon=True)
        self.index = index
        self.stop = Event()
        self.frames = deque(maxlen=6)
        self.available = Condition()
        self.error = None
        self.fps = None

    def take_triplet(self, stop):
        with self.available:
            while len(self.frames) < 3 and self.error is None and not stop.is_set():
                self.available.wait(timeout=0.1)
            if stop.is_set():
                return None
            if self.error is not None:
                raise self.error
            return [self.frames.popleft() for _ in range(3)]

    def run(self):
        capture = None
        try:
            capture = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
            if not capture.isOpened():
                capture.release()
                capture = cv2.VideoCapture(self.index, cv2.CAP_MSMF)
            if not capture.isOpened():
                raise OSError(f"無法開啟攝影機（{self.index}）。")

            frame_index = 0
            sample_start = perf_counter()
            sample_count = 0
            while not self.stop.is_set():
                ok, frame = capture.read()
                if not ok:
                    raise OSError("無法讀取攝影機畫面。")
                captured_at = perf_counter()
                frame_index += 1
                sample_count += 1
                elapsed = captured_at - sample_start
                if elapsed >= 1:
                    self.fps = sample_count / elapsed
                    sample_start = captured_at
                    sample_count = 0
                with self.available:
                    self.frames.append((frame_index, captured_at, frame))
                    self.available.notify()
        except Exception as exc:
            with self.available:
                self.error = exc
                self.available.notify()
        finally:
            if capture is not None:
                capture.release()
