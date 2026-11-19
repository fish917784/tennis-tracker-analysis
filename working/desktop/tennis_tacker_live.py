"""
網球即時追蹤與鷹眼落點系統 (Tennis Tracker Live)
========================================================================
整合：
1. 【即時自動球場校準】：標準網球場幾何標線模型匹配
2. 【即時 TrackNet 球體追蹤】：EfficientNetV2-B0 MSPE-CSCA 深度注意力網路
3. 【即時 CatBoost 彈跳落點】：滑動視窗動態差值回歸
4. 【即時 2D 鷹眼俯視小球場】：右上角即時顯示球體公尺投影座標與歷史落點
5. 【地面投影估速】：空中球的高度會影響估值，不代表真實三維球速
6. 【相機與影片雙模】：支援 Webcam/HDMI 擷取卡即時輸入，或各格式網球影片檔案
"""

from __future__ import annotations

import os
import sys
import math
from importlib.util import module_from_spec, spec_from_file_location
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from collections import deque
from pathlib import Path
from queue import Empty, SimpleQueue
from threading import Thread, Event, Condition, Lock
from time import perf_counter
from typing import Optional, List, Dict, Tuple, Any

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageTk

# 加入相關路徑
ROOT = Path(__file__).resolve().parent
CODEX_COURT_DIR = Path(os.environ.get("TENNIS_COURT_DIR", str(ROOT.parent / "court")))
BOUNCE_MODEL_PATH = os.environ.get("TENNIS_BOUNCE_MODEL", str(ROOT.parent / "models" / "ctb_regr_bounce.cbm"))
VERIFIED_COURT_DIR = ROOT.parent.parent / "test_artifacts" / "tennis-videos"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(CODEX_COURT_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_COURT_DIR))

# 載入偵測器模組
from live_detector import DEFAULT_MODEL, MODEL_DIR, WEIGHTS, Detector, ModelChoice, CameraCapture, WIDTH, HEIGHT

try:
    import catboost as ctb
    _CATBOOST_READY = True
except ImportError:
    _CATBOOST_READY = False

try:
    from court_detector import detect_court, CourtDetection
    from court_tracking import CourtTracker, court_geometry_is_visible, court_poses_consistent, refine_court_geometry, court_view_changed
    _COURT_READY = True
except ImportError:
    try:
        from court_detector import detect_court, CourtDetection
        CourtTracker = None
        court_geometry_is_visible = None
        court_view_changed = None
        _COURT_READY = True
    except ImportError:
        _COURT_READY = False
        CourtTracker = None
        court_geometry_is_visible = None
        court_view_changed = None

# ITF 標準球場物理尺寸 (公尺)
COURT_WIDTH_M = 10.9728   # 36 ft
COURT_LENGTH_M = 23.7744  # 78 ft

COURT_CORNERS_4_M = np.array([
    [0.0, 0.0],
    [COURT_WIDTH_M, 0.0],
    [COURT_WIDTH_M, COURT_LENGTH_M],
    [0.0, COURT_LENGTH_M]
], dtype=np.float32)


def load_desktop_source_court(source):
    if not (VERIFIED_COURT_DIR / "verified-evidence.json").is_file():
        return None
    # Desktop and Web use the same source fingerprint and calibration checks.
    loader = ROOT.parent / "web" / "verified_court.py"
    spec = spec_from_file_location("_tennis_verified_source_court", loader)
    if spec is None or spec.loader is None:
        return None
    try:
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
    except (OSError, ImportError):
        return None
    return module.load_verified_court(Path(source), VERIFIED_COURT_DIR)


def preview_schedule(source_fps, count, ready_at, next_deadline=None):
    """Schedule each source frame at its video cadence, without catch-up bursts."""
    if not math.isfinite(source_fps) or source_fps <= 0:
        raise ValueError("Preview source FPS must be positive and finite.")
    if not isinstance(count, (int, np.integer)) or count < 1:
        raise ValueError("Preview frame count must be a positive integer.")
    if not math.isfinite(ready_at) or (next_deadline is not None and not math.isfinite(next_deadline)):
        raise ValueError("Preview times must be finite.")
    start = ready_at if next_deadline is None else max(ready_at, next_deadline)
    period = 1.0 / source_fps
    return [start + index * period for index in range(count)], start + count * period


# ── 2D 俯視小球場渲染器 (動態色票與自動重製) ──────────────────────
def render_mini_court(mc_w: int = 120, mc_h: int = 240,
                      current_m: Optional[Tuple[float, float]] = None,
                      bounces: Optional[List[Dict[str, Any]]] = None,
                      court_calibrated: bool = True,
                      inner_color: Optional[Tuple[int, int, int]] = None,
                      outer_color: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
    canvas = np.zeros((mc_h, mc_w, 3), dtype=np.uint8)

    # 1. 未偵測到球場：去掉落點、重製顯示、呈現未校準狀態
    if not court_calibrated:
        canvas[:] = (20, 26, 36)
        cv2.rectangle(canvas, (4, 4), (mc_w - 4, mc_h - 4), (45, 60, 80), 1)
        cv2.putText(canvas, "NO COURT", (14, mc_h // 2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (70, 70, 240), 2, cv2.LINE_AA)
        cv2.putText(canvas, "RESET / PAUSED", (8, mc_h // 2 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (140, 160, 180), 1, cv2.LINE_AA)
        return canvas

    # 2. 外場底色：實時跟隨場地外圍顏色同步 (若無採樣則預設質感深藍綠)
    if outer_color is not None:
        bg_col = tuple(max(10, min(255, int(c * 0.75))) for c in outer_color)
    else:
        bg_col = (28, 42, 56)
    canvas[:] = bg_col

    pad = 8
    cw, ch = mc_w - 2 * pad, mc_h - 2 * pad
    sx, sy = cw / COURT_WIDTH_M, ch / COURT_LENGTH_M

    def to_m(x_m, y_m):
        return int(pad + x_m * sx), int(pad + y_m * sy)

    # 3. 內場填色：實時跟隨真實場地內場色彩同步 (紅土/藍地/草地)！
    in_col = inner_color if inner_color is not None else (38, 62, 110)
    tl = to_m(0, 0)
    br = to_m(COURT_WIDTH_M, COURT_LENGTH_M)
    cv2.rectangle(canvas, tl, br, in_col, -1)
    cv2.rectangle(canvas, tl, br, (230, 230, 230), 1)

    # 白線標記
    line_col = (225, 225, 225)
    # 單打線 (兩側距邊緣 1.37m)
    cv2.line(canvas, to_m(1.37, 0), to_m(1.37, COURT_LENGTH_M), line_col, 1)
    cv2.line(canvas, to_m(COURT_WIDTH_M - 1.37, 0), to_m(COURT_WIDTH_M - 1.37, COURT_LENGTH_M), line_col, 1)

    # 球網 (中央 11.885m)
    net_y = to_m(0, COURT_LENGTH_M / 2)[1]
    cv2.line(canvas, (to_m(0, COURT_LENGTH_M / 2)[0] - 2, net_y),
             (to_m(COURT_WIDTH_M, COURT_LENGTH_M / 2)[0] + 2, net_y), (90, 220, 255), 2)

    # 發球線 (距底線 5.485m)
    cv2.line(canvas, to_m(1.37, 5.485), to_m(COURT_WIDTH_M - 1.37, 5.485), line_col, 1)
    cv2.line(canvas, to_m(1.37, COURT_LENGTH_M - 5.485), to_m(COURT_WIDTH_M - 1.37, COURT_LENGTH_M - 5.485), line_col, 1)
    cv2.line(canvas, to_m(COURT_WIDTH_M / 2, 5.485), to_m(COURT_WIDTH_M / 2, COURT_LENGTH_M - 5.485), line_col, 1)

    # 4. 標記歷史落點 (只有在球場已標定時繪製)
    if bounces and court_calibrated:
        for b in bounces:
            bx_m, by_m = b["court_x"], b["court_y"]
            if -2.0 <= bx_m <= 13.0 and -2.0 <= by_m <= 26.0:
                mx, my = to_m(max(0.0, min(COURT_WIDTH_M, bx_m)), max(0.0, min(COURT_LENGTH_M, by_m)))
                color = (50, 220, 80) if b["is_in"] else (60, 70, 240)
                cv2.circle(canvas, (mx, my), 4, color, -1, cv2.LINE_AA)
                cv2.circle(canvas, (mx, my), 5, (255, 255, 255), 1, cv2.LINE_AA)

    # 5. 標記當前球體即時位置
    if current_m and court_calibrated:
        cx_m, cy_m = current_m
        if -2.0 <= cx_m <= 13.0 and -2.0 <= cy_m <= 26.0:
            cx, cy = to_m(max(0.0, min(COURT_WIDTH_M, cx_m)), max(0.0, min(COURT_LENGTH_M, cy_m)))
            cv2.circle(canvas, (cx, cy), 5, (80, 243, 197), -1, cv2.LINE_AA)
            cv2.circle(canvas, (cx, cy), 7, (255, 255, 255), 1, cv2.LINE_AA)

    return canvas


# ── 球場偵測與光流時序追蹤背景線程 ──────────────────────────────────
class CourtWorker(Thread):
    def __init__(self, callback, fps: float = 30.0):
        super().__init__(daemon=True)
        self.callback = callback
        self.fps = fps
        self.lock = Lock()
        self.cv = Condition(self.lock)
        self.pending = None
        self.stop_event = Event()
        self.busy = False
        self.tracker = None
        if _COURT_READY and CourtTracker is not None:
            try:
                self.tracker = CourtTracker(fps=self.fps)
            except Exception:
                self.tracker = None

    def submit(self, frame: np.ndarray, frame_idx: int):
        with self.cv:
            self.pending = (frame.copy(), frame_idx)
            self.cv.notify()

    def run(self):
        while not self.stop_event.is_set():
            with self.cv:
                while self.pending is None and not self.stop_event.is_set():
                    self.cv.wait(timeout=0.04)
                if self.stop_event.is_set():
                    break
                frame, frame_idx = self.pending
                self.pending = None
                self.busy = True

            det = None
            if _COURT_READY:
                try:
                    # 優先使用來源影格排程、當前場線驗證及時序共識。
                    if self.tracker is not None:
                        det = self.tracker.update(frame, frame_idx=frame_idx)
                    # 舊環境未提供追蹤器時才使用單幀備援。
                    elif self.tracker is None:
                        det = detect_court(frame, max_width=320)
                        if getattr(det, "partial", False):
                            det = None
                except Exception:
                    det = None

            self.busy = False
            self.callback(det, frame_idx)


def extract_ball_candidates(heatmap: np.ndarray, width: int, height: int, min_thresh: float = 0.32) -> list[dict]:
    """從熱力圖提取候選球體（過濾超大面積雜訊與細長線條，使用加權質心精準定位）"""
    max_val = float(heatmap.max())
    if max_val < min_thresh:
        return []
    cut = max(min_thresh, max_val * 0.50)
    mask = (heatmap >= cut).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 30 or h > 30:  # 網球在 512x288 下尺寸緊湊，排除選手肢體或大塊反光
            continue
        aspect = max(w / max(1, h), h / max(1, w))
        if aspect > 3.2:      # 排除球網、場地標線等細長條雜訊
            continue
        roi = heatmap[y:y+h, x:x+w]
        peak = float(roi.max())
        M = cv2.moments(roi)
        if M["m00"] > 1e-4:
            cx = x + M["m10"] / M["m00"]
            cy = y + M["m01"] / M["m00"]
        else:
            cx = x + w / 2.0
            cy = y + h / 2.0
        px = min(width - 1, int(round(cx * width / WIDTH)))
        py = min(height - 1, int(round(cy * height / HEIGHT)))
        candidates.append({"pt": (px, py), "peak": peak, "w": w, "h": h})
    candidates.sort(key=lambda item: item["peak"], reverse=True)
    return candidates


class BallContinuityTracker:
    """運動學連續性追蹤器：杜絕跳躍雜訊、選手鞋拍誤判與軌跡亂跳"""
    def __init__(self, width: int = 1280, height: int = 720, fps: float = 30.0):
        self.width = width
        self.height = height
        self.fps = fps
        self.max_speed_px = max(65.0, width * 0.07)
        self.last_pos = None
        self.velocity = (0.0, 0.0)
        self.lost_frames = 0

    def update(self, candidates: list[dict], width: int = 0, height: int = 0) -> Optional[tuple[int, int]]:
        if width > 0 and width != self.width:
            self.width = width
            self.height = height
            self.max_speed_px = max(65.0, width * 0.07)

        if not candidates:
            self.lost_frames += 1
            if self.lost_frames > 8:
                self.last_pos = None
                self.velocity = (0.0, 0.0)
            return None

        # 若當前無軌跡，需信心足夠 (>=0.42) 才作為新起點
        if self.last_pos is None or self.lost_frames > 8:
            best = candidates[0]
            if best["peak"] >= 0.42:
                self.last_pos = best["pt"]
                self.velocity = (0.0, 0.0)
                self.lost_frames = 0
                return best["pt"]
            else:
                self.lost_frames += 1
                return None

        dt = 1 + self.lost_frames
        pred_x = self.last_pos[0] + self.velocity[0] * dt
        pred_y = self.last_pos[1] + self.velocity[1] * dt
        search_radius = self.max_speed_px * (1.0 + 0.5 * self.lost_frames)

        best_cand = None
        best_score = -1e9

        for cand in candidates:
            pt = cand["pt"]
            dist_to_pred = math.hypot(pt[0] - pred_x, pt[1] - pred_y)
            dist_to_last = math.hypot(pt[0] - self.last_pos[0], pt[1] - self.last_pos[1])

            # 跨螢幕遠距離雜訊直接剔除（防止瞬間瞬移到選手腳下或對側場地）
            if dist_to_last > search_radius and dist_to_pred > search_radius:
                continue

            penalty = dist_to_pred / max(1.0, search_radius)
            score = cand["peak"] * 1.6 - penalty
            if score > best_score:
                best_score = score
                best_cand = cand

        if best_cand is not None:
            new_pt = best_cand["pt"]
            raw_vx = (new_pt[0] - self.last_pos[0]) / dt
            raw_vy = (new_pt[1] - self.last_pos[1]) / dt
            self.velocity = (0.6 * raw_vx + 0.4 * self.velocity[0],
                             0.6 * raw_vy + 0.4 * self.velocity[1])
            self.last_pos = new_pt
            self.lost_frames = 0
            return new_pt
        else:
            self.lost_frames += 1
            if self.lost_frames > 8:
                self.last_pos = None
                self.velocity = (0.0, 0.0)
            return None


# ── 即時追蹤與分析線程 ──────────────────────────────────────────
class LiveTrackerRun(Thread):
    def __init__(self, source, choice, realtime=False, fast_analysis=True, six_frame_batch=True):
        super().__init__(daemon=True)
        self.source = source
        self.choice = choice
        self.requests = SimpleQueue()
        self.requests.put(choice)
        self.events = SimpleQueue()
        self.stop = Event()
        self.realtime = Event()
        self.fast_analysis = Event()
        self.six_frame_batch = Event()
        if fast_analysis:
            self.fast_analysis.set()
        if six_frame_batch:
            self.six_frame_batch.set()
        if realtime:
            self.realtime.set()
        self.previews = deque(maxlen=60)
        self.skipped_frames = 0
        self.source_fps = None
        self.camera = None

        # 顯示開關 (可動態由 GUI 控制)
        self.show_trail = True
        self.show_court = True
        self.show_minicourt = True
        self.show_speed = True
        self.show_fps = True

        self.current_fps = 0.0
        self.last_inference_ms = 0.0
        self._fps_timestamps = deque(maxlen=30)
        self._analysis_frame_count = 0

        # 即時分析狀態
        self.H: Optional[np.ndarray] = None
        self.H_inv: Optional[np.ndarray] = None
        self.court_corners: Optional[np.ndarray] = None
        self.court_calibrated = False
        self._display_geometry = None
        self._court_invalid_since = None
        self._last_local_court_check = -1
        self.court_displayed = False
        self.court_inner_color: Optional[Tuple[int, int, int]] = None
        self.court_outer_color: Optional[Tuple[int, int, int]] = None
        self.no_court_frames = 0
        self.tracking_paused = False

        # The worker publishes results; only this analysis thread mutates state.
        self._court_results = SimpleQueue()
        self._last_court_result = -1
        self._verified_court_result = None
        self._court_verified_frame = -1
        self._last_court_submit = 0
        self._last_processed_frame = None
        self._court_thumbnail = None
        self._source_court = None
        self._source_court_segment = None
        self.last_error = None
        # 球場偵測線程 (每 5 幀偵測一次)
        self.court_worker = CourtWorker(self.on_court_detected)
        self.court_worker.start()

        self.ball_trail = deque(maxlen=16)
        self.ball_history = deque(maxlen=60)  # (frame_idx, x, y, vis, court_x, court_y, speed)
        self.ball_tracker = BallContinuityTracker()
        self.bounces = []
        self.active_ripples = []  # (frame_start, x, y, is_in, id, text)
        self.last_bounce_frame = -999

        self.catboost_model = None
        self._load_catboost()

        self.current_speed_kmh = 0.0
        self.max_speed_kmh = 0.0

    def _load_catboost(self):
        if _CATBOOST_READY and os.path.exists(BOUNCE_MODEL_PATH):
            try:
                self.catboost_model = ctb.CatBoostRegressor()
                self.catboost_model.load_model(BOUNCE_MODEL_PATH)
                self.events.put(("log", f"CatBoost 模型載入成功：{BOUNCE_MODEL_PATH}"))
            except Exception as e:
                self.events.put(("log", f"CatBoost 載入失敗：{e}"))
                self.catboost_model = None

    @property
    def input_fps(self):
        return self.camera.fps if self.camera is not None else self.source_fps

    def _record_analysis_fps(self, frame_count, timestamp=None):
        self._analysis_frame_count += frame_count
        now = perf_counter() if timestamp is None else timestamp
        self._fps_timestamps.append((now, self._analysis_frame_count))
        if len(self._fps_timestamps) >= 2:
            elapsed = now - self._fps_timestamps[0][0]
            if elapsed > 0:
                processed = self._analysis_frame_count - self._fps_timestamps[0][1]
                self.current_fps = round(processed / elapsed, 1)

    def _wait_for_analysis_slot(self, deadline):
        """Wait only while the file rate limit is active and ahead of schedule."""
        while deadline is not None and not self.fast_analysis.is_set():
            remaining = deadline - perf_counter()
            if remaining <= 0 or self.stop.wait(min(remaining, 0.05)):
                break
        return not self.stop.is_set()

    def project_point(self, px: float, py: float) -> Tuple[float, float, bool]:
        if not self.court_calibrated or self.H is None:
            return 0.0, 0.0, False
        pt_img = np.array([[[px, py]]], dtype=np.float32)
        pt_court = cv2.perspectiveTransform(pt_img, self.H)[0][0]
        xm, ym = float(pt_court[0]), float(pt_court[1])
        is_in = (-0.20 <= xm <= 11.17) and (-0.20 <= ym <= 23.97)
        return xm, ym, is_in

    def on_court_detected(self, det: Optional[Any], frame_idx: int):
        """Thread-safe publication; never mutate analysis state in the worker."""
        self._court_results.put((det, frame_idx))

    def _reset_ball_motion(self):
        self.ball_history.clear()
        self.ball_trail.clear()
        self.ball_tracker.last_pos = None
        self.ball_tracker.velocity = (0.0, 0.0)
        self.ball_tracker.lost_frames = 0
        self.current_speed_kmh = 0.0

    def _clear_court_display(self):
        had_history = bool(self.bounces or self.active_ripples or self._display_geometry is not None)
        self.bounces.clear()
        self.active_ripples.clear()
        self._display_geometry = None
        self._court_invalid_since = None
        self.court_inner_color = None
        self.court_outer_color = None
        if had_history:
            self.events.put(("reset_bounces", None))

    def _court_for_display(self, frame_idx, fps):
        if self.court_calibrated:
            self.court_displayed = True
            return self._display_geometry
        holding = (self._display_geometry is not None and self._court_invalid_since is not None
                   and frame_idx - self._court_invalid_since < max(1, round(fps * .45)))
        if not holding and self._display_geometry is not None:
            self._clear_court_display()
            self.events.put(("court", {"calibrated": False, "verifying": False}))
        self.court_displayed = bool(holding)
        return self._display_geometry if holding else None

    def _apply_court_result(self, det: Optional[Any], frame_idx: int, hard=False, observed_frame=None):
        """套用已驗證場地；失效立即停判，短暫保留外觀供重新確認。"""
        self._last_court_result = frame_idx
        observed_frame = frame_idx if observed_frame is None else observed_frame
        # Candidates arrive from the verified tracker/current-frame check.
        # Requiring a second, higher threshold after a brief occlusion can
        # otherwise reject the same valid tracked pose until its score rises.
        threshold = 0.77
        if det is not None and getattr(det, "score", 0.0) >= threshold:
            self._set_court_geometry(det.image_to_court, det.court_to_image, det.quad, observed_frame,
                                     partial=getattr(det, "partial", False))
            self.events.put(("court", {
                "score": float(det.score),
                "val_score": float(det.validation_score),
                "corners": self.court_corners.tolist(),
                "calibrated": True
            }))
        else:
            # 沒偵測到場地：清除矩陣與角點，取消標定
            self.court_calibrated = False
            self.H = None
            self.H_inv = None
            self.court_corners = None
            self._reset_ball_motion()
            self.active_ripples.clear()
            if hard or self._display_geometry is None:
                self._clear_court_display()
            elif self._court_invalid_since is None:
                self._court_invalid_since = observed_frame

            self.events.put(("court", {
                "score": 0.0,
                "val_score": 0.0,
                "corners": None,
                "calibrated": False,
                "verifying": self._display_geometry is not None,
            }))

    def _set_court_geometry(self, image_to_court, court_to_image, quad, observed_frame, *, partial=False):
        self._court_verified_frame = observed_frame
        if not self.court_calibrated:
            self._reset_ball_motion()
        self.H = image_to_court
        self.H_inv = court_to_image
        self.court_corners = quad.astype(np.float32)
        self._display_geometry = {"quad": self.court_corners.copy(), "H_inv": self.H_inv.copy(), "partial": partial}
        self._court_invalid_since = None
        self.court_calibrated = True
        self.no_court_frames = 0
        self.tracking_paused = False

    def _verify_court_on_frame(self, frame: np.ndarray, matrix=None, allow_display=False, *, partial=None) -> bool:
        self._verified_court_result = None
        if court_geometry_is_visible is None:
            return False
        try:
            known = self.H_inv
            if known is None and allow_display and self._display_geometry is not None:
                known = self._display_geometry["H_inv"]
            candidate = known if matrix is None else matrix
            established = known is not None
            if established and matrix is not None:
                established = court_poses_consistent(known, matrix, frame.shape)
            if partial is None:
                partial = (matrix is None and self._display_geometry is not None
                           and self._display_geometry.get("partial", False))
            options = {"established": established}
            if partial:
                options["partial"] = True
            self._verified_court_result = refine_court_geometry(frame, candidate, **options)
            return self._verified_court_result is not None
        except (cv2.error, ValueError, TypeError):
            return False

    def sample_court_colors(self, frame: np.ndarray):
        """從影像中採樣真實球場的內場與外場色彩（使小地圖顏色與場地實時同步）"""
        if not self.court_calibrated or self.H_inv is None:
            return
        h, w = frame.shape[:2]
        try:
            # 1. 內場採樣點 (發球區中央安全空曠區)
            pts_inner = np.array([[[5.485, 8.0]], [[5.485, 15.77]]], dtype=np.float32)
            pts_inner_img = cv2.perspectiveTransform(pts_inner, self.H_inv)
            in_cols = []
            for p in pts_inner_img:
                ix, iy = int(round(p[0][0])), int(round(p[0][1]))
                if 12 <= ix < w - 12 and 12 <= iy < h - 12:
                    patch = frame[iy-8:iy+9, ix-8:ix+9]
                    in_cols.append(np.median(patch, axis=(0, 1)))
            if in_cols:
                cur_in = np.mean(in_cols, axis=0).astype(int)
                if self.court_inner_color is None:
                    self.court_inner_color = tuple(int(c) for c in cur_in)
                else:
                    self.court_inner_color = tuple(
                        int(0.20 * cur_in[i] + 0.80 * self.court_inner_color[i]) for i in range(3)
                    )

            # 2. 外場採樣點 (邊線兩側外圍 1.5 米處)
            pts_outer = np.array([[[-1.5, 11.88]], [[12.47, 11.88]]], dtype=np.float32)
            pts_outer_img = cv2.perspectiveTransform(pts_outer, self.H_inv)
            out_cols = []
            for p in pts_outer_img:
                ox, oy = int(round(p[0][0])), int(round(p[0][1]))
                if 12 <= ox < w - 12 and 12 <= oy < h - 12:
                    patch = frame[oy-8:oy+9, ox-8:ox+9]
                    out_cols.append(np.median(patch, axis=(0, 1)))
            if out_cols:
                cur_out = np.mean(out_cols, axis=0).astype(int)
                if self.court_outer_color is None:
                    self.court_outer_color = tuple(int(c) for c in cur_out)
                else:
                    self.court_outer_color = tuple(
                        int(0.20 * cur_out[i] + 0.80 * self.court_outer_color[i]) for i in range(3)
                    )
        except Exception:
            pass

    def check_live_bounce(self, frame_idx: int, fps: float):
        """以 5 影格滑動視窗進行 CatBoost 即時觸地反彈回歸"""
        # 【重要限制】：沒偵測到場地就沒有彈跳
        if not self.court_calibrated or self.H is None:
            return

        if self.catboost_model is None or len(self.ball_history) < 5:
            return

        # 檢查倒數第 3 幀 (中心幀)
        # 需要歷史中有：t-2, t-1, t, t+1, t+2
        items = list(self.ball_history)
        if len(items) < 5:
            return

        # 取最後 5 個連續影格
        w = items[-5:]
        if any(b[0] - a[0] != 1 for a, b in zip(w, w[1:])):
            return
        f_center = w[2][0]
        if f_center - self.last_bounce_frame < int(fps * 0.40):
            return

        # 必須 5 幀皆可見
        if not all(pt[3] for pt in w):
            return

        x_curr, y_curr = w[2][1], w[2][2]
        eps = 1e-15
        feats = {
            "x_diff_1": abs(w[1][1] - x_curr),
            "x_diff_2": abs(w[0][1] - x_curr),
            "x_diff_inv_1": abs(w[3][1] - x_curr),
            "x_diff_inv_2": abs(w[4][1] - x_curr),
            "x_div_1": abs(abs(w[1][1] - x_curr) / (abs(w[3][1] - x_curr) + eps)),
            "x_div_2": abs(abs(w[0][1] - x_curr) / (abs(w[4][1] - x_curr) + eps)),
            "y_diff_1": w[1][2] - y_curr,
            "y_diff_2": w[0][2] - y_curr,
            "y_diff_inv_1": w[3][2] - y_curr,
            "y_diff_inv_2": w[4][2] - y_curr,
            "y_div_1": (w[1][2] - y_curr) / (w[3][2] - y_curr + eps),
            "y_div_2": (w[0][2] - y_curr) / (w[4][2] - y_curr + eps),
        }
        df = pd.DataFrame([feats])
        score = float(self.catboost_model.predict(df)[0])

        if score >= 0.45:
            xm, ym, is_in = self.project_point(x_curr, y_curr)
            # 範圍過濾
            if -2.0 <= xm <= 13.0 and -2.0 <= ym <= 26.0:
                self.last_bounce_frame = f_center
                bid = len(self.bounces) + 1
                time_s = round(f_center / fps, 2)
                mins = int(time_s // 60)
                secs = time_s % 60
                ts = f"{mins:02d}:{secs:05.2f}"
                dec = "IN (界內)" if is_in else "OUT (界外)"

                event_dict = {
                    "id": bid,
                    "frame": int(f_center),
                    "x": int(round(x_curr)),
                    "y": int(round(y_curr)),
                    "court_x": round(xm, 2),
                    "court_y": round(ym, 2),
                    "is_in": is_in,
                    "decision": dec,
                    "time_sec": time_s,
                    "timestamp": ts,
                    "confidence": round(score, 3),
                    "speed_kmh": round(w[2][6], 1),
                    "speed_type": "ground_projection_estimate",
                }
                self.bounces.append(event_dict)
                self.events.put(("bounce", event_dict))
                # 加入動態波紋
                self.active_ripples.append((frame_idx, int(round(x_curr)), int(round(y_curr)), is_in, bid))

    def calibrate_court(self, frame: np.ndarray):
        """相容方法：單幀球場幾何標定"""
        if not _COURT_READY:
            return
        try:
            det = detect_court(frame, max_width=320)
            self.on_court_detected(None if getattr(det, "partial", False) else det, 0)
        except Exception:
            self.on_court_detected(None, 0)

    def _update_live_court(self, frame, frame_idx, fps):
        # Verify the displayed pose at the tracking cadence. Waiting half a
        # second for a busy acquisition worker lets small camera tilts add up
        # beyond the local alignment range, even though the court stays visible.
        max_result_age = max(1, round(fps / 9))
        thumbnail = cv2.cvtColor(cv2.resize(frame, (96, 54)), cv2.COLOR_BGR2GRAY)
        cut = False
        if self._court_thumbnail is not None:
            if court_view_changed is not None:
                cut = court_view_changed(self._court_thumbnail, thumbnail)
            else:
                difference = cv2.absdiff(thumbnail, self._court_thumbnail)
                cut = cv2.mean(difference)[0] > 21 and np.count_nonzero(difference > 29) > 0.31 * difference.size
        self._court_thumbnail = thumbnail
        if cut:
            self._apply_court_result(None, frame_idx, hard=True)
        recovering_display = (self._display_geometry is not None and self._court_invalid_since is not None
                              and frame_idx - self._court_invalid_since < max(1, round(fps * .45)))
        try:
            while True:
                result, result_frame = self._court_results.get_nowait()
                if self._last_court_result < result_frame <= frame_idx:
                    options = {"allow_display": recovering_display}
                    if getattr(result, "partial", False):
                        options["partial"] = True
                    if result is not None and self._verify_court_on_frame(frame, result.court_to_image, **options):
                        self._apply_court_result(self._verified_court_result or result, result_frame, observed_frame=frame_idx)
                        if self.court_calibrated:
                            self._court_verified_frame = frame_idx
                    elif (self.court_calibrated or recovering_display) and self._verify_court_on_frame(frame, allow_display=recovering_display):
                        # A single failed worker reply must not erase a court
                        # that the current image still supports.
                        self._last_court_result = result_frame
                        if self._verified_court_result is not None:
                            self._apply_court_result(self._verified_court_result, result_frame, observed_frame=frame_idx)
                        self._court_verified_frame = frame_idx
                    else:
                        self._apply_court_result(None, result_frame, observed_frame=frame_idx)
        except Empty:
            pass
        if ((self.court_calibrated or recovering_display)
                and frame_idx - max(self._court_verified_frame, self._last_local_court_check) >= max_result_age):
            self._last_local_court_check = frame_idx
            if self._verify_court_on_frame(frame, allow_display=recovering_display):
                if self._verified_court_result is not None:
                    self._apply_court_result(self._verified_court_result, self._last_court_result, observed_frame=frame_idx)
                self._court_verified_frame = frame_idx
            else:
                self._apply_court_result(None, self._last_court_result, observed_frame=frame_idx)

        # 1. 球場標定與動態場地色票採樣
        check_interval = max(1, round(fps / 8))
        if frame_idx == 1:
            if _COURT_READY:
                try:
                    det = detect_court(frame, max_width=320)
                    if (det is not None and not getattr(det, "partial", False) and det.score >= .94
                            and self._verify_court_on_frame(frame, det.court_to_image)):
                        self._apply_court_result(det, frame_idx)
                    elif self.court_worker is not None:
                        self.court_worker.submit(frame, frame_idx)
                        self._last_court_submit = frame_idx
                except Exception:
                    self._apply_court_result(None, frame_idx)
        elif cut or frame_idx - self._last_court_submit >= check_interval:
            if self.court_worker is not None:
                self.court_worker.submit(frame, frame_idx)
                self._last_court_submit = frame_idx

    def _update_verified_court(self, frame, frame_idx):
        state = self._source_court.update(frame, frame_idx - 1)
        changed = self._source_court_segment is not None and state["segment_id"] != self._source_court_segment
        self._source_court_segment = state["segment_id"]
        if not state["calibrated"]:
            if self.court_calibrated or (not state["displayed"] and self._display_geometry is not None):
                self._apply_court_result(None, frame_idx, hard=not state["displayed"])
            return
        if changed and self.court_calibrated:
            self._apply_court_result(None, frame_idx, hard=True)
        was_calibrated = self.court_calibrated
        self._set_court_geometry(state["H"], state["H_inv"], state["corners"], frame_idx)
        if not was_calibrated:
            self.events.put(("court", {
                "corners": self.court_corners.tolist(),
                "calibrated": True,
                "verified_source": True,
            }))

    def process_and_draw(self, frame: np.ndarray, heatmap: np.ndarray, frame_idx: int, fps: float) -> Tuple[np.ndarray, Optional[Tuple[int, int]]]:
        canvas = frame.copy()
        h, w = canvas.shape[:2]
        frame_step = 1 if self._last_processed_frame is None else max(1, frame_idx - self._last_processed_frame)
        if self._last_processed_frame is not None and frame_idx != self._last_processed_frame + 1:
            self._reset_ball_motion()
        self._last_processed_frame = frame_idx
        if self._source_court is None:
            self._update_live_court(frame, frame_idx, fps)
        else:
            self._update_verified_court(frame, frame_idx)

        # 檢測場地鎖定狀態與 0.5 秒無球場關閉機制
        max_no_court_frames = max(1, int(fps * 0.5))  # 0.5 秒對應影格數
        if self.court_calibrated:
            self.no_court_frames = 0
            self.tracking_paused = False
            # 每 5 幀採樣一次真實球場內/外場顏色
            if frame_idx % 5 == 0:
                self.sample_court_colors(frame)
        else:
            self.no_court_frames += frame_step
            if self.no_court_frames >= max_no_court_frames:
                self.tracking_paused = True
                self.ball_tracker.last_pos = None
                self.ball_tracker.velocity = (0.0, 0.0)
                self.ball_trail.clear()

        # 2. 定位球體 (【使用者要求】：沒偵測到場地 0.5 秒就先關閉球追蹤，直到球場偵測到才重新開啟)
        pt = None
        if not self.tracking_paused:
            cands = extract_ball_candidates(heatmap, w, h, min_thresh=0.32)
            pt = self.ball_tracker.update(cands, width=w, height=h)

        # 3. 速度計算與軌跡記錄
        visible = pt is not None
        cx_m, cy_m = None, None
        if visible:
            px, py = pt
            if self.court_calibrated and self.H is not None:
                cx_m, cy_m, _ = self.project_point(px, py)

            # Ground-plane projection estimate; airborne height is unobserved.
            previous_speed = self.current_speed_kmh
            self.current_speed_kmh = 0.0
            if self.court_calibrated and len(self.ball_history) >= 3:
                prev = self.ball_history[-3]
                if prev[3] and prev[4] is not None and cx_m is not None:
                    dist_m = math.hypot(cx_m - prev[4], cy_m - prev[5])
                    dt_s = (frame_idx - prev[0]) / fps
                    if dt_s > 0:
                        speed = (dist_m / dt_s) * 3.6
                        if 15.0 <= speed <= 260.0:
                            # 指數平滑
                            self.current_speed_kmh = round(0.4 * speed + 0.6 * previous_speed, 1)
                            self.max_speed_kmh = max(self.max_speed_kmh, self.current_speed_kmh)

            self.ball_history.append((frame_idx, px, py, True, cx_m, cy_m, self.current_speed_kmh))
            if len(self.ball_trail) > 0:
                last_px, last_py = self.ball_trail[-1]
                if math.hypot(px - last_px, py - last_py) > max(75.0, w * 0.08):
                    self.ball_trail.clear()
            self.ball_trail.append((px, py))
        else:
            self.current_speed_kmh = 0.0
            self.ball_history.append((frame_idx, 0, 0, False, None, None, 0.0))
            if len(self.ball_trail) > 0:
                self.ball_trail.popleft()

        # 4. CatBoost 即時落點判斷 (若沒偵測到場地將在內部直接 return)
        self.check_live_bounce(frame_idx, fps)

        # ── 繪圖圖層 ──
        display_geometry = self._court_for_display(frame_idx, fps)
        verifying = display_geometry is not None and not self.court_calibrated

        # A. 球場標線 (ITF Court Overlay，僅在偵測到球場時繪製)
        if self.show_court and display_geometry is not None:
            poly = display_geometry["quad"].astype(np.int32)
            cv2.polylines(canvas, [poly], True, (60, 190, 220) if verifying else (60, 240, 160), 2, cv2.LINE_AA)
            for c_pt in display_geometry["quad"]:
                cv2.circle(canvas, (int(round(c_pt[0])), int(round(c_pt[1]))), 5, (30, 160, 255), -1, cv2.LINE_AA)
            if verifying:
                cv2.putText(canvas, "COURT: VERIFYING (LINE CALLS PAUSED)", (18, 76),
                            cv2.FONT_HERSHEY_SIMPLEX, .45, (60, 190, 220), 1, cv2.LINE_AA)

        # B. 球跡光暈 (Glowing Trail)
        if self.show_trail and len(self.ball_trail) >= 2:
            trail_list = list(self.ball_trail)
            for j in range(1, len(trail_list)):
                if math.hypot(trail_list[j][0] - trail_list[j-1][0], trail_list[j][1] - trail_list[j-1][1]) > max(75.0, w * 0.08):
                    continue
                alpha = j / len(trail_list)
                col = (int(197 * (1 - alpha)), int(243 * alpha), int(80 * alpha))
                thick = max(1, int(3 * alpha))
                cv2.line(canvas, trail_list[j-1], trail_list[j], col, thick, cv2.LINE_AA)

        # C. 當前球心光圈
        if visible and pt is not None:
            cv2.circle(canvas, pt, 8, (14, 18, 24), 3, cv2.LINE_AA)
            cv2.circle(canvas, pt, 7, (80, 243, 197), 2, cv2.LINE_AA)
            cv2.circle(canvas, pt, 2, (80, 243, 197), -1, cv2.LINE_AA)

        # D. 動態落點衝擊波 (Bounce Ripples)
        self.active_ripples = [r for r in self.active_ripples if frame_idx - r[0] <= 35]
        for f_start, bx, by, is_in, bid in self.active_ripples:
            age = frame_idx - f_start
            radius = int(8 + age * 1.2)
            alpha = max(0.0, 1.0 - (age / 35.0))
            col = (int(50 * alpha), int(230 * alpha), int(80 * alpha)) if is_in else (int(50 * alpha), int(60 * alpha), int(240 * alpha))
            cv2.circle(canvas, (bx, by), radius, col, 2, cv2.LINE_AA)
            cv2.circle(canvas, (bx, by), max(3, radius - 6), col, 1, cv2.LINE_AA)
            tag = f"#{bid} {'IN' if is_in else 'OUT'}"
            cv2.putText(canvas, tag, (bx - 22, max(24, by - radius - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 2, cv2.LINE_AA)

        # E. 右上角 2D 鷹眼俯視小球場 (小地圖：落點重製 + 色彩與真實場地實時同步)
        if self.show_minicourt:
            scale = min(1.0, max(1, w - 28) / 110, max(1, h - 48) / 220)
            mc_w, mc_h = max(8, round(110 * scale)), max(8, round(220 * scale))
            ox = max(0, w - mc_w - 14)
            oy = min(14, max(0, h - mc_h))
            cur_pos = (cx_m, cy_m) if cx_m is not None and self.court_calibrated else None
            mini_court_img = render_mini_court(
                110, 220,
                current_m=cur_pos,
                bounces=self.bounces if self.court_calibrated else [],
                court_calibrated=self.court_calibrated or verifying,
                inner_color=self.court_inner_color,
                outer_color=self.court_outer_color
            )
            if verifying:
                cv2.putText(mini_court_img, "VERIFYING", (12, 110), cv2.FONT_HERSHEY_SIMPLEX, .38, (60, 190, 220), 1, cv2.LINE_AA)
            if mini_court_img.shape[:2] != (mc_h, mc_w):
                mini_court_img = cv2.resize(mini_court_img, (mc_w, mc_h), interpolation=cv2.INTER_AREA)
            cv2.rectangle(canvas, (ox - 4, oy - 4), (ox + mc_w + 4, oy + mc_h + 18), (16, 20, 26), -1)
            cv2.rectangle(canvas, (ox - 4, oy - 4), (ox + mc_w + 4, oy + mc_h + 18), (60, 80, 100), 1)
            cv2.putText(canvas, "2D MINI-COURT", (ox + 6, oy + mc_h + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (160, 200, 220), 1, cv2.LINE_AA)
            canvas[oy:oy+mc_h, ox:ox+mc_w] = mini_court_img

        # 狀態標示：若無球場超過 0.5 秒已暫停球追蹤
        if self.tracking_paused:
            cv2.putText(canvas, "BALL TRACKING: PAUSED (NO COURT > 0.5s)", (18, h - 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (60, 70, 245), 2, cv2.LINE_AA)

        # F. 頂部儀表與時速 / FPS HUD
        if self.show_speed or self.show_fps:
            hud_w = 510
            cv2.rectangle(canvas, (10, 10), (hud_w, 54), (16, 22, 30), -1)
            cv2.rectangle(canvas, (10, 10), (hud_w, 54), (50, 70, 90), 1)

            # 速度欄
            if self.show_speed:
                if self.tracking_paused:
                    spd_text = "BALL: PAUSED [NO COURT]"
                    spd_col = (60, 70, 240)
                elif self.court_calibrated and self.current_speed_kmh > 0:
                    spd_text = f"PROJECTED: {self.current_speed_kmh:.0f} km/h"
                    spd_col = (80, 243, 197)
                elif not self.court_calibrated:
                    spd_text = "PROJECTED: [NO COURT]"
                    spd_col = (60, 70, 240)
                else:
                    spd_text = "PROJECTED: -- km/h"
                    spd_col = (80, 243, 197)
                cv2.putText(canvas, spd_text, (18, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.46, spd_col, 1, cv2.LINE_AA)
                max_text = f"EST MAX: {self.max_speed_kmh:.0f} km/h"
                bounce_text = f"BOUNCES: {len(self.bounces)}"
                cv2.putText(canvas, max_text, (230, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (200, 220, 240), 1, cv2.LINE_AA)
                cv2.putText(canvas, bounce_text, (230, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (180, 200, 220), 1, cv2.LINE_AA)

            # 分隔線
            cv2.line(canvas, (336, 14), (336, 50), (45, 60, 75), 1)

            # 即時 FPS 欄
            if self.show_fps:
                fps_val = fps
                if fps_val > 0:
                    fps_text = f"SOURCE FPS: {fps_val:.2f}"
                    fps_col = (80, 243, 197)
                else:
                    fps_text = "FPS: --"
                    fps_col = (180, 200, 220)
                cv2.putText(canvas, fps_text, (348, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.39, fps_col, 1, cv2.LINE_AA)
                analysis_text = f"ANALYSIS: {self.current_fps:.1f} FPS" if self.current_fps > 0 else "ANALYSIS: -- FPS"
                cv2.putText(canvas, analysis_text, (348, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (160, 180, 200), 1, cv2.LINE_AA)

        return canvas, pt

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

    def detect_camera(self, detector, device):
        self.camera = CameraCapture(self.source)
        self.camera.start()
        deadline = None
        fps = 30.0
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
            self.last_inference_ms = inference_ms
            self._record_analysis_fps(len(frames), ready)
            first_time = batch[0][1]
            period = max(1 / 240, min(1, (batch[-1][1] - first_time) / 2))
            fps = 1.0 / period if period > 0 else 30.0
            deadline = max(deadline or ready, ready - period)

            for (index, captured_at, frame), heatmap in zip(batch, heatmaps):
                drawn_frame, point = self.process_and_draw(frame, heatmap, index, fps)
                preview = (index, detector.choice.name, drawn_frame, point, inference_ms)
                self.previews.append((deadline + captured_at - first_time, preview))
            deadline += batch[-1][1] - first_time + period

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
                raise OSError(f"無法開啟影片檔案：{self.source}")

            fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
            self.source_fps = fps if math.isfinite(fps) and fps > 0 else 30.0
            source_court = load_desktop_source_court(self.source)
            if source_court is not None and source_court.matches(
                self.source_fps,
                int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            ):
                self._source_court = source_court
            elif self.court_worker and CourtTracker is not None:
                self.court_worker.tracker = CourtTracker(fps=self.source_fps)
            deadline = None
            previous_mode = self.fast_analysis.is_set()
            batch_memory_ok = True
            index = 0

            while not self.stop.is_set():
                detector = self.update_model(detector, device)
                realtime = self.realtime.is_set()
                use_six = (batch_memory_ok and self.six_frame_batch.is_set() and self.fast_analysis.is_set()
                           and device.type == "cuda" and detector.choice.name == DEFAULT_MODEL)

                frames = []
                for _ in range(6 if use_six else 3):
                    if self.stop.is_set():
                        return
                    ok, frame = capture.read()
                    if not ok:
                        break
                    frames.append(frame)

                if not frames:
                    self.events.put(("status", "影片播放完畢 ✓"))
                    break

                started = perf_counter()
                if len(frames) <= 3:
                    heatmaps = detector.predict(frames, realtime=realtime)
                else:
                    groups = (frames[:3], frames[3:])
                    try:
                        heatmaps = np.concatenate(detector.predict_batch(groups), axis=0)
                    except torch.cuda.OutOfMemoryError:
                        batch_memory_ok = False
                        self.events.put(("status", "顯示卡記憶體不足，本次分析已切回 3 幀批次"))
                    if not batch_memory_ok:
                        # Retry the already decoded frames after the failed
                        # batch has unwound and released its GPU tensors.
                        heatmaps = np.concatenate([detector.predict(group, realtime=realtime) for group in groups], axis=0)
                ready = perf_counter()
                inference_ms = (ready - started) * 1000
                self.last_inference_ms = inference_ms

                if not self._wait_for_analysis_slot(deadline):
                    return
                fast_analysis = self.fast_analysis.is_set()
                released_at = perf_counter()
                if fast_analysis != previous_mode:
                    self._fps_timestamps.clear()
                    self.current_fps = 0.0
                    previous_mode = fast_analysis
                self._record_analysis_fps(len(frames), released_at)
                if fast_analysis:
                    deadlines, deadline = [released_at] * len(frames), None
                else:
                    deadlines, deadline = preview_schedule(self.source_fps, len(frames), released_at, deadline)

                milliseconds_per_triplet = inference_ms * 3 / len(frames)
                for frame, heatmap, display_at in zip(frames, heatmaps, deadlines):
                    if self.stop.is_set():
                        return
                    index += 1
                    drawn_frame, point = self.process_and_draw(frame, heatmap, index, self.source_fps)
                    preview = (index, detector.choice.name, drawn_frame, point, milliseconds_per_triplet)
                    self.previews.append((display_at, preview))

        except Exception as exc:
            self.last_error = str(exc)
            self.events.put(("error", str(exc)))
        finally:
            if self.court_worker is not None:
                self.court_worker.stop_event.set()
                with self.court_worker.cv:
                    self.court_worker.cv.notify()
                self.court_worker.join(timeout=2.0)
            if capture is not None:
                capture.release()
            if self.camera is not None:
                self.camera.stop.set()
                self.camera.join()


# ── 主 GUI 介面 ───────────────────────────────────────────────────
class TennisTrackerLiveApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.run: Optional[LiveTrackerRun] = None
        self.last_preview = None
        self.pending_preview = None
        self.image = None
        self.closing = False

        self.choice = ModelChoice.preset(DEFAULT_MODEL)
        self.model_name = tk.StringVar(value=DEFAULT_MODEL)
        self.camera_id = tk.StringVar(value="0")
        self.realtime = tk.BooleanVar(value=True)
        self.analysis_limit_var = tk.BooleanVar(value=False)
        self.six_frame_batch_var = tk.BooleanVar(value=True)

        # 顯示開關
        self.show_trail_var = tk.BooleanVar(value=True)
        self.show_court_var = tk.BooleanVar(value=True)
        self.show_minicourt_var = tk.BooleanVar(value=True)
        self.show_speed_var = tk.BooleanVar(value=True)
        self.show_fps_var = tk.BooleanVar(value=True)

        # 狀態變數
        self.preview_times = deque()
        self.next_fps_update = None
        self.preview_fps = tk.StringVar(value="來源 FPS: -- | 預覽: -- | 分析: --")
        self.status = tk.StringVar(value="請開啟影片或攝影機開始即時追蹤")
        self.details = tk.StringVar(value="等待影像輸入")
        self.device = tk.StringVar(value="")
        self.court_status = tk.StringVar(value="球場標定：待偵測 (未校準)")

        self.radar_speed = tk.StringVar(value="-- km/h")
        self.radar_max = tk.StringVar(value="投影最高: -- km/h")
        self.stats_bounces = tk.StringVar(value="落點數: 0 (IN: 0 / OUT: 0)")

        root.title("🏸 網球即時追蹤與鷹眼落點系統 (tennis_tacker_live)")
        root.geometry("1320x820")
        root.minsize(960, 600)
        root.configure(bg="#10151c")
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.option_add("*Font", ("Microsoft JhengHei UI", 9))

        self._build_style()
        self._build_layout()
        root.after(10, self.poll)

    def _build_style(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#10151c")
        style.configure("TLabel", background="#10151c", foreground="#d0dce8")
        style.configure("Header.TLabel", font=("Microsoft JhengHei UI", 12, "bold"), foreground="#50f3c5")
        style.configure("TButton", font=("Microsoft JhengHei UI", 9, "bold"))
        style.configure("TCheckbutton", background="#16202c", foreground="#c0d0e0")
        style.configure("Treeview", background="#16202c", foreground="#ffffff", fieldbackground="#16202c", rowheight=24)
        style.configure("Treeview.Heading", background="#223042", foreground="#50c8f3", font=("Microsoft JhengHei UI", 9, "bold"))

    def _build_layout(self):
        main_container = ttk.Frame(self.root, padding=10)
        main_container.pack(fill="both", expand=True)

        # 頂部操作工具列 (雙行清晰排版)
        toolbar = tk.Frame(main_container, bg="#16202c", bd=1, relief="ridge", padx=10, pady=6)
        toolbar.pack(fill="x", pady=(0, 8))

        row1 = tk.Frame(toolbar, bg="#16202c")
        row1.pack(fill="x", pady=(0, 5))

        self.video_button = ttk.Button(row1, text="📂 開啟影片…", command=self.open_video)
        self.video_button.pack(side="left", padx=(0, 6))

        tk.Label(row1, text="攝影機:", bg="#16202c", fg="#d0dce8").pack(side="left", padx=(8, 2))
        ttk.Spinbox(row1, from_=0, to=10, width=3, textvariable=self.camera_id).pack(side="left")
        self.camera_button = ttk.Button(row1, text="📹 啟動攝影機", command=self.open_camera)
        self.camera_button.pack(side="left", padx=6)

        self.stop_button = ttk.Button(row1, text="⏹ 停止", command=self.stop, state="disabled")
        self.stop_button.pack(side="left", padx=(0, 10))

        tk.Label(row1, text="模型:", bg="#16202c", fg="#d0dce8").pack(side="left", padx=(4, 2))
        self.model_selector = ttk.Combobox(row1, state="readonly", values=list(WEIGHTS),
                                            textvariable=self.model_name, width=20)
        self.model_selector.pack(side="left", padx=(0, 10))
        self.model_selector.bind("<<ComboboxSelected>>", self.select_preset)

        # 右側醒目即時 FPS 徽章與硬體
        self.fps_badge = tk.Label(row1, textvariable=self.preview_fps, bg="#1e2c3a", fg="#50f3c5",
                                  font=("Microsoft JhengHei UI", 10, "bold"), padx=10, pady=2, bd=1, relief="solid")
        self.fps_badge.pack(side="right", padx=(8, 0))

        self.device_badge = tk.Label(row1, textvariable=self.device, bg="#16202c", fg="#8fa0b2",
                                     font=("Microsoft JhengHei UI", 9))
        self.device_badge.pack(side="right", padx=6)

        row2 = tk.Frame(toolbar, bg="#16202c")
        row2.pack(fill="x")

        ttk.Checkbutton(row2, text="即時預覽", variable=self.realtime, command=self.update_toggles).pack(side="left", padx=4)
        self.analysis_limit_button = ttk.Checkbutton(
            row2, text="分析限速（來源 FPS）", variable=self.analysis_limit_var, command=self.update_toggles
        )
        self.analysis_limit_button.pack(side="left", padx=4)
        ttk.Checkbutton(row2, text="EffB0 6 幀批次", variable=self.six_frame_batch_var,
                        command=self.update_toggles).pack(side="left", padx=4)
        ttk.Checkbutton(row2, text="🎾 球跡光暈", variable=self.show_trail_var, command=self.update_toggles).pack(side="left", padx=4)
        ttk.Checkbutton(row2, text="📐 球場標線", variable=self.show_court_var, command=self.update_toggles).pack(side="left", padx=4)
        ttk.Checkbutton(row2, text="🗺️ 2D 小球場", variable=self.show_minicourt_var, command=self.update_toggles).pack(side="left", padx=4)
        ttk.Checkbutton(row2, text="⚡ 地面投影估速", variable=self.show_speed_var, command=self.update_toggles).pack(side="left", padx=4)
        ttk.Checkbutton(row2, text="⏱️ 顯示 FPS", variable=self.show_fps_var, command=self.update_toggles).pack(side="left", padx=4)

        # 中間主體：左影像 Canvas，右資訊側邊欄
        body = ttk.Frame(main_container)
        body.pack(fill="both", expand=True)

        # 左側 Canvas
        canvas_frame = ttk.Frame(body)
        canvas_frame.pack(side="left", fill="both", expand=True)
        self.canvas = tk.Canvas(canvas_frame, background="#0a0e14", highlightthickness=1, highlightbackground="#223042")
        self.canvas.pack(fill="both", expand=True)
        self.frame_item = self.canvas.create_image(0, 0, anchor="center", state="hidden")
        self.empty_item = self.canvas.create_text(0, 0, text="網球即時偵測預覽\n請開啟影片或攝影機", fill="#627894",
                                                  font=("Microsoft JhengHei UI", 15, "bold"), justify="center")
        self.canvas.bind("<Configure>", lambda event: self.render())

        # 右側控制與統計側邊欄
        sidebar = ttk.Frame(body, width=380, padding=(10, 0))
        sidebar.pack(side="right", fill="y")
        sidebar.pack_propagate(False)

        # 1. 儀表板卡片
        speed_card = tk.Frame(sidebar, bg="#16202c", bd=1, relief="ridge", padx=12, pady=10)
        speed_card.pack(fill="x", pady=(0, 8))
        tk.Label(speed_card, text="⚡ 地面投影估速 (非真實球速)", bg="#16202c", fg="#8fa0b2", font=("Microsoft JhengHei UI", 9, "bold")).pack(anchor="w")
        self.speed_lbl = tk.Label(speed_card, textvariable=self.radar_speed, bg="#16202c", fg="#50f3c5", font=("Impact", 28))
        self.speed_lbl.pack(anchor="w")
        tk.Label(speed_card, text="球的高度會影響估值，非真實三維球速", bg="#16202c", fg="#8fa0b2",
                 font=("Microsoft JhengHei UI", 8)).pack(anchor="w")
        tk.Label(speed_card, textvariable=self.radar_max, bg="#16202c", fg="#f5b041", font=("Microsoft JhengHei UI", 9)).pack(anchor="w")

        # 即時 FPS 顯示列
        fps_box = tk.Frame(speed_card, bg="#10151c", bd=1, relief="sunken", padx=8, pady=4)
        fps_box.pack(fill="x", pady=(6, 0))
        tk.Label(fps_box, text="⚡ 影格率: ", bg="#10151c", fg="#8fa0b2", font=("Microsoft JhengHei UI", 9, "bold")).pack(side="left")
        self.fps_sidebar_lbl = tk.Label(fps_box, textvariable=self.preview_fps, bg="#10151c", fg="#50f3c5", font=("Microsoft JhengHei UI", 9, "bold"))
        self.fps_sidebar_lbl.pack(side="left")

        # 2. 球場校準狀態卡片
        court_card = tk.Frame(sidebar, bg="#16202c", bd=1, relief="ridge", padx=12, pady=8)
        court_card.pack(fill="x", pady=(0, 8))
        self.court_status_lbl = tk.Label(court_card, textvariable=self.court_status, bg="#16202c", fg="#50c8f3", font=("Microsoft JhengHei UI", 9, "bold"))
        self.court_status_lbl.pack(anchor="w")

        # 3. 即時落點列表 (Live Bounces Treeview)
        bounces_card = tk.Frame(sidebar, bg="#16202c", bd=1, relief="ridge", padx=10, pady=8)
        bounces_card.pack(fill="both", expand=True, pady=(0, 8))
        tk.Label(bounces_card, text="🎾 即時落點記錄 (CatBoost ML Bounces)", bg="#16202c", fg="#8fa0b2", font=("Microsoft JhengHei UI", 9, "bold")).pack(anchor="w", pady=(0, 4))

        columns = ("id", "decision", "time", "coord", "speed", "conf")
        self.tree = ttk.Treeview(bounces_card, columns=columns, show="headings", height=12)
        self.tree.heading("id", text="#")
        self.tree.heading("decision", text="判定")
        self.tree.heading("time", text="時間")
        self.tree.heading("coord", text="座標 (X,Y)")
        self.tree.heading("speed", text="投影估速")
        self.tree.heading("conf", text="信心度")

        self.tree.column("id", width=28, anchor="center")
        self.tree.column("decision", width=62, anchor="center")
        self.tree.column("time", width=55, anchor="center")
        self.tree.column("coord", width=85, anchor="center")
        self.tree.column("speed", width=55, anchor="center")
        self.tree.column("conf", width=52, anchor="center")

        tree_scroll = ttk.Scrollbar(bounces_card, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        tree_scroll.pack(side="right", fill="y")

        self.tree.tag_configure("in", foreground="#50f3c5")
        self.tree.tag_configure("out", foreground="#ff6b6b")

        # 4. 統計看板
        stats_card = tk.Frame(sidebar, bg="#16202c", bd=1, relief="ridge", padx=12, pady=8)
        stats_card.pack(fill="x")
        tk.Label(stats_card, textvariable=self.stats_bounces, bg="#16202c", fg="#e0e8f0", font=("Microsoft JhengHei UI", 9, "bold")).pack(anchor="w")

        # 底部狀態列
        footer = ttk.Frame(main_container)
        footer.pack(fill="x", pady=(8, 0))
        ttk.Label(footer, textvariable=self.status, foreground="#8fa0b2").pack(side="left")
        ttk.Label(footer, textvariable=self.preview_fps, foreground="#50f3c5", font=("Microsoft JhengHei UI", 9, "bold")).pack(side="left", padx=25)
        ttk.Label(footer, textvariable=self.details, foreground="#50f3c5").pack(side="right")

    def update_toggles(self):
        if self.run:
            self.run.realtime.set() if self.realtime.get() else self.run.realtime.clear()
            self.run.six_frame_batch.set() if self.six_frame_batch_var.get() else self.run.six_frame_batch.clear()
            fast_analysis = not self.analysis_limit_var.get()
            if not isinstance(self.run.source, int) and self.run.fast_analysis.is_set() != fast_analysis:
                self.run.previews.clear()
                self.pending_preview = None
                self.preview_times.clear()
                self.next_fps_update = None
                self.run.fast_analysis.set() if fast_analysis else self.run.fast_analysis.clear()
            self.run.show_trail = self.show_trail_var.get()
            self.run.show_court = self.show_court_var.get()
            self.run.show_minicourt = self.show_minicourt_var.get()
            self.run.show_speed = self.show_speed_var.get()
            self.run.show_fps = self.show_fps_var.get()

    def select_preset(self, event=None):
        self.set_choice(ModelChoice.preset(self.model_name.get()))

    def set_choice(self, choice):
        self.choice = choice
        if self.run:
            self.run.requests.put(choice)

    def open_video(self):
        path = filedialog.askopenfilename(
            parent=self.root, title="選擇網球影片",
            filetypes=[("網球影片", "*.mp4 *.mov *.avi *.mkv *.m4v *.webm"), ("所有檔案", "*.*")]
        )
        if path:
            self.start(path)

    def open_camera(self):
        try:
            cam_idx = int(self.camera_id.get())
        except ValueError:
            messagebox.showerror("攝影機編號錯誤", "請輸入有效的攝影機編號 (0, 1, 2...)", parent=self.root)
            return
        self.start(cam_idx)

    def start(self, source):
        if self.run and self.run.is_alive():
            self.stop()
        self.tree.delete(*self.tree.get_children())
        self.last_preview = None
        self.pending_preview = None
        self.preview_times.clear()
        self.next_fps_update = None
        self.radar_speed.set("-- km/h")
        self.radar_max.set("投影最高: -- km/h")
        self.preview_fps.set("來源 FPS: -- | 預覽: -- | 分析: --")
        self.stats_bounces.set("落點數: 0 (IN: 0 / OUT: 0)")
        self.court_status.set("球場標定：每5幀定期檢測中…")
        if hasattr(self, "court_status_lbl"):
            self.court_status_lbl.configure(fg="#50c8f3")

        self.status.set("系統啟動中…")
        self.details.set("等待影像輸入")
        self.set_running(True)

        self.analysis_limit_button.configure(state="disabled" if isinstance(source, int) else "normal")
        self.run = LiveTrackerRun(source, self.choice, realtime=self.realtime.get(),
                                  fast_analysis=not self.analysis_limit_var.get(),
                                  six_frame_batch=self.six_frame_batch_var.get())
        self.update_toggles()
        self.run.start()

    def set_running(self, running: bool):
        self.video_button.configure(state="disabled" if running else "normal")
        self.camera_button.configure(state="disabled" if running else "normal")
        self.stop_button.configure(state="normal" if running else "disabled")
        if not running:
            self.analysis_limit_button.configure(state="normal")

    def stop(self):
        if self.run and self.run.is_alive():
            self.run.stop.set()
            self.stop_button.configure(state="disabled")
            self.status.set("正在停止…")

    def close(self):
        if self.run and self.run.is_alive():
            self.closing = True
            self.stop()
        else:
            self.root.destroy()

    def poll(self):
        run = self.run
        if run:
            preview = None
            now = perf_counter()
            if not run.stop.is_set():
                try:
                    # 全速影片只顯示最新預覽，分析仍逐幀進行。
                    is_camera = isinstance(run.source, int)
                    if not is_camera and run.fast_analysis.is_set():
                        while len(run.previews) > 1:
                            run.previews.popleft()
                        self.pending_preview = None
                    if is_camera and self.realtime.get() and len(run.previews) > 18:
                        while len(run.previews) > 6:
                            run.previews.popleft()
                        self.pending_preview = None

                    if self.pending_preview is None and run.previews:
                        self.pending_preview = run.previews.popleft()

                    if self.pending_preview and self.pending_preview[0] <= now:
                        _, preview = self.pending_preview
                        self.pending_preview = None
                except IndexError:
                    pass

            if preview is not None and preview is not self.last_preview:
                self.last_preview = preview
                self.render()
                displayed_at = perf_counter()
                self.preview_times.append(displayed_at)

            # 更新 FPS (滑動窗口瞬時平滑計算，杜絕冷啟動低幀誤導)
            now = perf_counter()
            if self.next_fps_update is None or now >= self.next_fps_update:
                while self.preview_times and self.preview_times[0] < now - 1.2:
                    self.preview_times.popleft()
                src_fps = run.input_fps
                src_text = f"{src_fps:.2f}" if src_fps is not None else "--"
                if len(self.preview_times) >= 4:
                    dt = self.preview_times[-1] - self.preview_times[0]
                    if dt > 0.05:
                        cur_fps = round((len(self.preview_times) - 1) / dt, 1)
                        self.preview_fps.set(f"來源 FPS: {src_text} | 預覽: {cur_fps:.1f} | 分析: {run.current_fps:.1f}")
                elif run.is_alive():
                    self.preview_fps.set(f"來源 FPS: {src_text} | 預覽: 測量中 | 分析: {run.current_fps:.1f}")
                self.next_fps_update = now + 0.20

            # 處理事件佇列 (落點、球場、狀態)
            if hasattr(run, "current_speed_kmh"):
                self.radar_speed.set(f"{run.current_speed_kmh:.0f} km/h" if run.current_speed_kmh > 0 else "-- km/h")
            try:
                while True:
                    kind, val = run.events.get_nowait()
                    if kind == "device":
                        self.device.set(f"硬體: {val}")
                    elif kind == "status":
                        self.status.set(val)
                    elif kind == "active":
                        self.status.set(f"運行中: {val.name}")
                    elif kind == "switch_error":
                        messagebox.showwarning("模型切換失敗", val, parent=self.root)
                    elif kind == "error":
                        run.last_error = str(val)
                        self.status.set(f"分析失敗：{val}")
                        messagebox.showerror("分析失敗", str(val), parent=self.root)
                    elif kind == "court":
                        if val.get("calibrated", False):
                            if val.get("verified_source"):
                                self.court_status.set("球場標定：已套用這支原片的校正 ✓")
                            else:
                                self.court_status.set(f"球場標定：已鎖定 ✓ (得分: {val['score']:.2f})")
                            if hasattr(self, "court_status_lbl"):
                                self.court_status_lbl.configure(fg="#50f3c5")
                        elif val.get("verifying", False):
                            self.court_status.set("球場標定：確認中（暫停落點判定）")
                            if hasattr(self, "court_status_lbl"):
                                self.court_status_lbl.configure(fg="#d9b45b")
                        else:
                            self.court_status.set("球場標定：未偵測到場地 (落點已重製/暫停追蹤)")
                            if hasattr(self, "court_status_lbl"):
                                self.court_status_lbl.configure(fg="#ff6b6b")
                    elif kind == "reset_bounces":
                        self.tree.delete(*self.tree.get_children())
                        self.stats_bounces.set("落點數: 0 (IN: 0 / OUT: 0)")
                    elif kind == "bounce":
                        self._add_bounce_row(val)
            except Empty:
                pass

            # 停止條件
            if not run.is_alive() and (run.stop.is_set() or (self.pending_preview is None and not run.previews)):
                self.run = None
                self.preview_fps.set("預覽 FPS  —")
                if self.closing:
                    self.root.destroy()
                    return
                self.set_running(False)
                self.status.set(f"分析失敗：{run.last_error}" if run.last_error else "已停止")

        self.root.after(8, self.poll)

    def _add_bounce_row(self, b: Dict[str, Any]):
        tag = "in" if b["is_in"] else "out"
        dec_str = "✓ IN" if b["is_in"] else "✕ OUT"
        coord_str = f"({b['court_x']}m, {b['court_y']}m)"
        spd_str = f"{b['speed_kmh']} km/h" if b.get("speed_kmh") else "--"
        conf_str = f"{(b['confidence'] * 100):.1f}%"

        item_id = self.tree.insert("", "end", values=(b["id"], dec_str, b["timestamp"], coord_str, spd_str, conf_str), tags=(tag,))
        self.tree.see(item_id)

        # 更新儀表板
        if b.get("speed_kmh"):
            self.radar_speed.set(f"{b['speed_kmh']:.0f} km/h")
        if self.run:
            self.radar_max.set(f"投影最高: {self.run.max_speed_kmh:.0f} km/h")
            in_count = sum(1 for item in self.run.bounces if item["is_in"])
            out_count = len(self.run.bounces) - in_count
            in_rate = (in_count / len(self.run.bounces) * 100) if self.run.bounces else 0
            self.stats_bounces.set(f"落點數: {len(self.run.bounces)} (IN: {in_count} / OUT: {out_count} · 進球率 {in_rate:.0f}%)")

    def render(self):
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        center = width / 2, height / 2
        if self.last_preview is None:
            self.canvas.itemconfigure(self.frame_item, state="hidden", image="")
            self.image = None
            self.canvas.coords(self.empty_item, *center)
            self.canvas.itemconfigure(self.empty_item, state="normal")
            return

        index, model, frame, point, inference_ms = self.last_preview
        frame_height, frame_width = frame.shape[:2]
        scale = min(width / frame_width, height / frame_height, 1)
        if scale < 1:
            size = (max(1, round(frame_width * scale)), max(1, round(frame_height * scale)))
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self.image = ImageTk.PhotoImage(image)
        self.canvas.itemconfigure(self.empty_item, state="hidden")
        self.canvas.coords(self.frame_item, *center)
        self.canvas.itemconfigure(self.frame_item, state="normal", image=self.image)
        loc_str = f"球位 ({point[0]}, {point[1]})" if point else "未見球體"
        self.details.set(f"第 {index} 幀   |   {loc_str}   |   推論: {inference_ms:.1f} ms/3 幀")


if __name__ == "__main__":
    win = tk.Tk()
    app = TennisTrackerLiveApp(win)
    win.mainloop()
