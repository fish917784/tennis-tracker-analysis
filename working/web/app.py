from __future__ import annotations
# ETN-B0 Ball Tracker with Auto Court Detection & Hawk-Eye Landing Point Analysis
# 先全自動定位球場幾何 (Court Geometry & Homography) 再進行完整賽事落點分析
import sys, uuid, shutil, os, json, time, subprocess
from collections import deque
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "desktop"))

import cv2, numpy as np, torch
from live_detector import DEFAULT_MODEL, MODEL_DIR, WEIGHTS
from tracknet_v2_b0_mspe_csca import TrackNetV2B0MSPECSCA
from inference_runtime import predict_heatmaps, predict_triplets

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pydantic import BaseModel, StrictBool

if __package__:
    from .court_detector import LiveCourtGeometry
    from .bounce_detector import detect_court_bounces_catboost as detect_court_bounces
    from .video_writer import create_video_writer
    from .verified_court import load_verified_court
else:
    from court_detector import LiveCourtGeometry
    from bounce_detector import detect_court_bounces_catboost as detect_court_bounces
    from video_writer import create_video_writer
    from verified_court import load_verified_court

# ── 路徑設定 ───────────────────────────────────────────────────
MODEL_PATH = MODEL_DIR / WEIGHTS[DEFAULT_MODEL]
UPLOAD_DIR = ROOT / "uploads"; UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = ROOT / "outputs"; OUTPUT_DIR.mkdir(exist_ok=True)
STATIC_DIR = ROOT / "static"; STATIC_DIR.mkdir(exist_ok=True)
VERIFIED_COURT_DIR = ROOT.parent.parent / "test_artifacts" / "tennis-videos"

app = FastAPI(title="🏸 ETN-B0 Court-Aware Ball Tracker")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(ROOT / "templates"))

WIDTH, HEIGHT = 512, 288
MAX_BALL_SPEED_PX = 75.0   # 30fps 下相鄰兩幀最大合法像素位移（超過視為假跳躍）

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
job_status: dict[str, dict] = {}

# ── 全域模型（延遲載入）────────────────────────────────────────
_model = None


class SourceFramePacer:
    """Optionally pace completed analysis frames without a catch-up burst."""

    def __init__(self, fps, limited=None, clock=None, sleeper=None):
        self.period = 1.0 / fps
        self.limited = limited or (lambda: False)
        self.clock = clock or time.perf_counter
        self.sleep = sleeper or time.sleep
        self.next_frame = None

    def wait(self):
        if not self.limited():
            self.next_frame = None
            return
        now = self.clock()
        if self.next_frame is None:
            self.next_frame = now + self.period
            return
        deadline = self.next_frame
        while self.limited():
            remaining = deadline - self.clock()
            if remaining <= 0:
                break
            # A live switch back to unlimited releases a wait promptly.
            self.sleep(min(remaining, .025))
        if not self.limited():
            self.next_frame = None
            return
        self.next_frame = max(deadline, self.clock()) + self.period


class AnalysisFrameRate:
    """Measure recent completed frames, starting fresh after a speed change."""

    def __init__(self, clock=None):
        self.clock = clock or time.perf_counter
        self.timestamps = deque()
        self.mode = None
        self.fps = 0.0

    def record(self, limited):
        now = self.clock()
        if limited != self.mode:
            self.timestamps.clear()
            self.mode = limited
        self.timestamps.append(now)
        while len(self.timestamps) > 2 and self.timestamps[0] < now - 1.0:
            self.timestamps.popleft()
        elapsed = now - self.timestamps[0]
        self.fps = (len(self.timestamps) - 1) / elapsed if elapsed > 0 else 0.0
        return self.fps


def get_model():
    global _model
    if _model is not None:
        return _model
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"找不到 EffB0 權重：{MODEL_PATH}。請放入模型資料夾或設定 TENNIS_WEIGHTS_DIR。")
    print(f"[INFO] 載入模型：{MODEL_PATH}  device={device}")
    m = TrackNetV2B0MSPECSCA()
    ck = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    state = ck.get("model", ck.get("net", ck))
    m.load_state_dict(state, strict=True)
    _model = m.eval().to(device, memory_format=torch.channels_last)
    print("[INFO] 模型載入完成 ✓")
    return _model


def extract_ball_candidates(heatmap: np.ndarray, width: int, height: int, min_thresh: float = 0.32) -> list[dict]:
    """從熱力圖提取所有候選球體（過濾超大面積雜訊與細長線條，使用加權質心精準定位）"""
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

    def update(self, candidates: list[dict]) -> Optional[tuple[int, int]]:
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
            dist_to_pred = np.hypot(pt[0] - pred_x, pt[1] - pred_y)
            dist_to_last = np.hypot(pt[0] - self.last_pos[0], pt[1] - self.last_pos[1])

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


def locate_ball(heatmap: np.ndarray, width: int, height: int, thresh: float = 0.32) -> Optional[tuple[int, int]]:
    """相容函數：單幀高斯峰值定位"""
    cands = extract_ball_candidates(heatmap, width, height, min_thresh=thresh)
    return cands[0]["pt"] if cands else None


def project_to_court(px: float, py: float, H: np.ndarray) -> tuple[float, float, bool]:
    """將影像像素 (px, py) 轉換為球場座標 (Xm, Ym) 並判斷是否界內 (IN)"""
    pt_img = np.array([[[px, py]]], dtype=np.float32)
    pt_court = cv2.perspectiveTransform(pt_img, H)[0][0]
    xm = float(pt_court[0])
    ym = float(pt_court[1])
    # 網球標準邊界判斷 (含雙打線 0~10.97m, 底線 0~23.77m，包含壓線寬度容差 0.20m)
    is_in = (-0.20 <= xm <= 11.17) and (-0.20 <= ym <= 23.97)
    return xm, ym, is_in


# ── 2D 俯視小球場渲染器 (Mini-Court：動態色票與自動重製) ────────────
def sample_court_colors_from_frame(frame: np.ndarray, H_inv: np.ndarray) -> tuple[Optional[tuple[int, int, int]], Optional[tuple[int, int, int]]]:
    """從影像中採樣真實球場的內場與外場色彩（使小地圖顏色與場地實時同步）"""
    if H_inv is None:
        return None, None
    h, w = frame.shape[:2]
    inner_color, outer_color = None, None
    try:
        pts_inner = np.array([[[5.485, 8.0]], [[5.485, 15.77]]], dtype=np.float32)
        pts_inner_img = cv2.perspectiveTransform(pts_inner, H_inv)
        in_cols = []
        for p in pts_inner_img:
            ix, iy = int(round(p[0][0])), int(round(p[0][1]))
            if 12 <= ix < w - 12 and 12 <= iy < h - 12:
                patch = frame[iy-8:iy+9, ix-8:ix+9]
                in_cols.append(np.median(patch, axis=(0, 1)))
        if in_cols:
            inner_color = tuple(int(c) for c in np.mean(in_cols, axis=0))

        pts_outer = np.array([[[-1.5, 11.88]], [[12.47, 11.88]]], dtype=np.float32)
        pts_outer_img = cv2.perspectiveTransform(pts_outer, H_inv)
        out_cols = []
        for p in pts_outer_img:
            ox, oy = int(round(p[0][0])), int(round(p[0][1]))
            if 12 <= ox < w - 12 and 12 <= oy < h - 12:
                patch = frame[oy-8:oy+9, ox-8:ox+9]
                out_cols.append(np.median(patch, axis=(0, 1)))
        if out_cols:
            outer_color = tuple(int(c) for c in np.mean(out_cols, axis=0))
    except Exception:
        pass
    return inner_color, outer_color


def create_mini_court_overlay(mc_w: int = 100, mc_h: int = 220,
                              current_m: Optional[tuple[float, float]] = None,
                              bounces: Optional[list[dict]] = None,
                              court_calibrated: bool = True,
                              inner_color: Optional[tuple[int, int, int]] = None,
                              outer_color: Optional[tuple[int, int, int]] = None) -> np.ndarray:
    """繪製標準 2D 俯視網球場圖，實時同步真實場地色彩，未標定時清空重製"""
    canvas = np.zeros((mc_h, mc_w, 3), dtype=np.uint8)

    if not court_calibrated:
        canvas[:] = (20, 26, 36)
        cv2.rectangle(canvas, (4, 4), (mc_w - 4, mc_h - 4), (45, 60, 80), 1)
        cv2.putText(canvas, "NO COURT", (10, mc_h // 2 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (70, 70, 240), 2, cv2.LINE_AA)
        cv2.putText(canvas, "RESET", (24, mc_h // 2 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (140, 160, 180), 1, cv2.LINE_AA)
        return canvas

    if outer_color is not None:
        bg_col = tuple(max(10, min(255, int(c * 0.75))) for c in outer_color)
    else:
        bg_col = (28, 42, 56)
    canvas[:] = bg_col

    pad = 6
    cw, ch = mc_w - 2 * pad, mc_h - 2 * pad
    sx, sy = cw / 10.97, ch / 23.77

    def to_m(x_m, y_m):
        return int(pad + x_m * sx), int(pad + y_m * sy)

    in_col = inner_color if inner_color is not None else (38, 68, 140)
    tl = to_m(0, 0)
    br = to_m(10.97, 23.77)
    cv2.rectangle(canvas, tl, br, in_col, -1)
    cv2.rectangle(canvas, tl, br, (230, 230, 230), 1)

    line_col = (225, 225, 225)
    # 單打線 (兩側距邊緣 1.37m)
    cv2.line(canvas, to_m(1.37, 0), to_m(1.37, 23.77), line_col, 1)
    cv2.line(canvas, to_m(9.60, 0), to_m(9.60, 23.77), line_col, 1)

    # 球網 (中央 11.885m)
    net_y = to_m(0, 11.885)[1]
    cv2.line(canvas, (to_m(0, 11.885)[0]-2, net_y), (to_m(10.97, 11.885)[0]+2, net_y), (100, 220, 255), 2)

    # 發球線 (距底線 5.485m)
    cv2.line(canvas, to_m(1.37, 5.485), to_m(9.60, 5.485), line_col, 1)
    cv2.line(canvas, to_m(1.37, 18.285), to_m(9.60, 18.285), line_col, 1)
    cv2.line(canvas, to_m(5.485, 5.485), to_m(5.485, 18.285), line_col, 1)

    # 標記歷史落點
    if bounces and court_calibrated:
        for b in bounces:
            bx_m, by_m = b["court_x"], b["court_y"]
            if -3.0 <= bx_m <= 14.0 and -3.0 <= by_m <= 27.0:
                mx, my = to_m(max(0.0, min(10.97, bx_m)), max(0.0, min(23.77, by_m)))
                color = (50, 220, 80) if b["is_in"] else (50, 70, 240)
                cv2.circle(canvas, (mx, my), 4, color, -1, cv2.LINE_AA)
                cv2.circle(canvas, (mx, my), 5, (255, 255, 255), 1, cv2.LINE_AA)

    # 標記當前球體即時位置
    if current_m and court_calibrated:
        cx_m, cy_m = current_m
        if -2.0 <= cx_m <= 13.0 and -2.0 <= cy_m <= 26.0:
            cx, cy = to_m(max(0.0, min(10.97, cx_m)), max(0.0, min(23.77, cy_m)))
            cv2.circle(canvas, (cx, cy), 5, (80, 243, 197), -1, cv2.LINE_AA)
            cv2.circle(canvas, (cx, cy), 7, (255, 255, 255), 1, cv2.LINE_AA)

    return canvas


# ── 推論單一三幀組 ─────────────────────────────────────────────
@torch.inference_mode()
def predict_triplet(frames_bgr: list) -> np.ndarray:
    model = get_model()
    return predict_heatmaps(model, frames_bgr, device, use_amp=False, interpolation=cv2.INTER_AREA)


@torch.inference_mode()
def predict_triplet_batch(groups: list) -> list[np.ndarray]:
    return predict_triplets(get_model(), groups, device, use_amp=False, interpolation=cv2.INTER_AREA)


# ── CatBoost 回歸落點偵測演算法 (ctb_regr_bounce.cbm) ──────────────


class CourtBallTrajectory:
    """Advance offline ball tracking with the verified court state of one frame."""

    def __init__(self, width, height, fps):
        self.size_and_rate = (width, height, fps)
        self.ball = BallContinuityTracker(*self.size_and_rate)
        self.segment = None
        self.no_court_frames = 0
        self.pause_after = max(1, int(fps * 0.5))
        self.court_margin = max(90.0, max(width, height) * 0.15)

    def update(self, candidates, court_state):
        if court_state["segment_id"] != self.segment:
            self.ball = BallContinuityTracker(*self.size_and_rate)
            self.segment = court_state["segment_id"]
        calibrated = court_state["calibrated"]
        self.no_court_frames = 0 if calibrated else self.no_court_frames + 1
        point = self.ball.update(candidates) if self.no_court_frames < self.pause_after else None
        if point is not None:
            x, y = point
            if (not calibrated or cv2.pointPolygonTest(court_state["court_poly"], (float(x), float(y)), True)
                    >= -self.court_margin):
                return {"x": x, "y": y, "visible": True}
        return {"x": 0, "y": 0, "visible": False}


# ── 影片處理主流程 ─────────────────────────────────────────────
def process_video(in_path: str, out_path: Optional[str], progress_cb=None, court_debug_path: str = None,
                  limit_to_source_fps=None, analysis_status_cb=None, *, analysis_result_cb=None,
                  candidate_source=None, court_geometry=None) -> tuple[list[dict], dict]:
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise ValueError(f"無法讀取影片：{in_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    if width < 32 or height < 32:
        cap.release()
        raise ValueError("影片解析度至少需要 32 × 32 像素")

    # 1. 自動偵測球場幾何 (Auto Court Line Detection & Homography)
    if progress_cb:
        progress_cb(0.04)

    geometry = (court_geometry if court_geometry is not None and court_geometry.matches(fps, width, height, total)
                else LiveCourtGeometry(fps))
    court_states = []
    H, H_inv, court_poly, corners_img, method = None, None, None, None, "none"

    court_info = {
        "method": method,
        "corners": [],
        "has_image": False,
    }

    # ─────────────────────────────────────────────────────────────
    # 第一階段：全片模型推論 + 自動球場空間過濾 + 速度限制 (5% ~ 50%)
    # ─────────────────────────────────────────────────────────────
    raw_trajectory: list[dict] = []
    triplet_buf = []
    triplets_per_batch = int(os.environ.get('TENNIS_TRIPLET_BATCH', '4' if device.type == 'cuda' else '1'))
    if not 1 <= triplets_per_batch <= 4:
        cap.release()
        raise ValueError('TENNIS_TRIPLET_BATCH must be between 1 and 4.')
    batch_frames = 3 * triplets_per_batch
    tracker = CourtBallTrajectory(width, height, fps)
    limited = limit_to_source_fps or (lambda: False)
    pacer = SourceFramePacer(fps, limited)
    analysis_rate = AnalysisFrameRate()
    analysis_started = time.perf_counter()
    if analysis_status_cb:
        analysis_status_cb(dict(phase="analysis", source_fps=fps, analysis_frames=0, analysis_fps=0.0))

    while True:
        ret, frame = cap.read()
        triplet_buf.append(frame if ret else None)

        if len(triplet_buf) == batch_frames or (not ret and triplet_buf):
            valid = [f for f in triplet_buf if f is not None]
            if not valid:
                break
            groups = [valid[index:index + 3] for index in range(0, len(valid), 3)]
            heatmaps = (np.concatenate(predict_triplet_batch(groups), axis=0)
                        if candidate_source is None else [None] * len(valid))
            for f_bgr, hmap in zip(valid, heatmaps):
                pacer.wait()
                state = geometry.update(f_bgr, frame_idx=len(raw_trajectory))
                court_states.append(state)
                if state["calibrated"] and court_info["method"] == "none":
                    court_info.update(method=state["method"], corners=state["corners"].round(1).tolist())
                    if court_debug_path:
                        debug = f_bgr.copy()
                        cv2.polylines(debug, [state["court_poly"]], True, (50, 243, 197), 2, cv2.LINE_AA)
                        court_info["has_image"] = bool(cv2.imwrite(court_debug_path, debug))
                cands = (extract_ball_candidates(hmap, width, height, min_thresh=0.32)
                         if candidate_source is None else candidate_source(len(raw_trajectory)))
                raw_trajectory.append(tracker.update(cands, state))
                analysis_rate.record(bool(limited()))

                if progress_cb and len(raw_trajectory) % 15 == 0:
                    p1 = 0.05 + (len(raw_trajectory) / total) * 0.45
                    progress_cb(min(p1, 0.49))
                if analysis_status_cb and len(raw_trajectory) % 15 == 0:
                    analysis_status_cb(dict(analysis_frames=len(raw_trajectory),
                                            analysis_fps=round(analysis_rate.fps, 2)))
            triplet_buf = []

        if not ret:
            break

    cap.release()
    if analysis_status_cb:
        elapsed = max(time.perf_counter() - analysis_started, 1e-9)
        analysis_status_cb(dict(phase="export", analysis_frames=len(raw_trajectory),
                                analysis_fps=round(analysis_rate.fps, 2),
                                analysis_average_fps=round(len(raw_trajectory) / elapsed, 2)))

    # ─────────────────────────────────────────────────────────────
    # 計算真實球場落點與 IN/OUT 判定 (完整賽事跨網擊球分段)
    # ─────────────────────────────────────────────────────────────
    bounces = detect_court_bounces(raw_trajectory, None, None, fps=fps, court_states=court_states)
    if analysis_result_cb is not None:
        analysis_result_cb(fps, raw_trajectory, court_states, bounces)
    if out_path is None:
        if progress_cb:
            progress_cb(1.0)
        return bounces, court_info
    bounce_dict = {b["frame"]: b for b in bounces}

    # ─────────────────────────────────────────────────────────────
    # 第二階段：渲染輸出影片 (含 2D 小球場、平滑軌跡與落點衝擊光環) (50% ~ 100%)
    # ─────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(in_path)
    writer = create_video_writer(out_path, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise ValueError("無法建立輸出影片，請檢查輸出路徑與編碼器")

    trail: list[tuple[int, int]] = []
    active_bounces = []     # (start_frame, x, y, is_in, id)
    past_bounces = []
    inner_color, outer_color = None, None

    scale = min(1.0, max(1, width - 24) / 100, max(1, height - 48) / 220)
    mc_w, mc_h = max(8, round(100 * scale)), max(8, round(220 * scale))
    ox = max(0, width - mc_w - 12)
    oy = min(12, max(0, height - mc_h))
    previous_segment = None
    previous_display_segment = None

    try:
        for f_idx in range(len(raw_trajectory)):
            ret, frame = cap.read()
            if not ret:
                break

            canvas = frame.copy()
            state = court_states[f_idx]
            court_calibrated = state["calibrated"]
            H, H_inv, court_poly = state["H"], state["H_inv"], state["court_poly"]
            corners_img, method = state["corners"], state["method"]
            court_displayed = state.get("displayed", court_calibrated)
            verifying = state.get("verifying", False)
            display_poly = state.get("display_court_poly", court_poly)
            display_corners = state.get("display_corners", corners_img)
            if state["segment_id"] != previous_segment:
                trail.clear()
                active_bounces.clear()
                previous_segment = state["segment_id"]
            display_segment = state.get("display_segment_id", state["segment_id"])
            if display_segment != previous_display_segment:
                past_bounces.clear()
                inner_color, outer_color = None, None
                previous_display_segment = display_segment
            cur_pt = raw_trajectory[f_idx]
            x, y, visible = cur_pt["x"], cur_pt["y"], cur_pt["visible"]

            # 動態採樣場地真實顏色 (使小地圖顏色與場地實時同步)
            if court_calibrated and (inner_color is None or f_idx % 30 == 0):
                cur_in, cur_out = sample_court_colors_from_frame(frame, H_inv)
                if cur_in is not None:
                    inner_color = cur_in
                if cur_out is not None:
                    outer_color = cur_out

            # 前 45 幀顯示自動偵測的球場輪廓綠色標記（讓使用者清楚看到球場已被偵測）
            if f_idx < 45 and court_displayed:
                alpha = max(0.0, 1.0 - (f_idx / 45.0))
                overlay = canvas.copy()
                line_color = (40, 190, 240) if verifying else (50, 243, 197)
                label = "COURT: VERIFYING (LINE CALLS PAUSED)" if verifying else f"COURT DETECTED ({method.upper()})"
                cv2.polylines(overlay, [display_poly], True, line_color, 2, cv2.LINE_AA)
                cv2.putText(overlay, label,
                            (int(display_corners[0][0]), int(display_corners[0][1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, line_color, 2, cv2.LINE_AA)
                cv2.addWeighted(overlay, alpha * 0.8, canvas, 1.0 - alpha * 0.8, 0, canvas)

            # 觸發新落點
            if f_idx in bounce_dict:
                b_info = bounce_dict[f_idx]
                active_bounces.append((f_idx, b_info["x"], b_info["y"], b_info["is_in"], b_info["id"]))
                past_bounces.append(b_info)

            # 1. 繪製平滑連續軌跡
            if visible:
                if len(trail) > 0:
                    dist = np.hypot(x - trail[-1][0], y - trail[-1][1])
                    if dist > MAX_BALL_SPEED_PX * 1.5:
                        trail.clear()
                trail.append((x, y))
            else:
                if len(trail) > 0:
                    trail.pop(0)

            if len(trail) > 12:
                trail.pop(0)

            for j in range(1, len(trail)):
                alpha = j / len(trail)
                color = (int(197 * (1 - alpha)), int(243 * alpha), int(80 * alpha))
                thickness = max(1, int(3 * alpha))
                cv2.line(canvas, trail[j-1], trail[j], color, thickness, cv2.LINE_AA)

            # 2. 繪製球體當前光圈
            if visible:
                cv2.circle(canvas, (x, y), 8, (14, 18, 24), 3, cv2.LINE_AA)
                cv2.circle(canvas, (x, y), 7, (80, 243, 197), 2, cv2.LINE_AA)
                cv2.circle(canvas, (x, y), 2, (80, 243, 197), -1, cv2.LINE_AA)

            # 3. 繪製精緻落點衝擊光圈 (持續 32 幀)
            active_bounces = [b for b in active_bounces if f_idx - b[0] <= 32]
            for f_start, bx, by, is_in, bid in active_bounces:
                age = f_idx - f_start
                radius = int(8 + age * 1.1)
                alpha = max(0.0, 1.0 - (age / 32.0))
                color = (int(50 * alpha), int(230 * alpha), int(80 * alpha)) if is_in else (int(50 * alpha), int(60 * alpha), int(240 * alpha))
                cv2.circle(canvas, (bx, by), radius, color, 2, cv2.LINE_AA)
                cv2.circle(canvas, (bx, by), max(3, radius - 6), color, 1, cv2.LINE_AA)
                tag = f"#{bid} {'IN' if is_in else 'OUT'}"
                cv2.putText(canvas, tag, (bx - 20, max(22, by - radius - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

            # 4. 右上角 2D 俯視小球場 (Mini-Court)
            current_m = None
            if visible and court_calibrated:
                xm, ym, _ = project_to_court(x, y, H)
                current_m = (xm, ym)
            mini_court_img = create_mini_court_overlay(
                100, 220,
                current_m=current_m,
                bounces=past_bounces if court_displayed else [],
                court_calibrated=court_displayed,
                inner_color=inner_color,
                outer_color=outer_color
            )
            if verifying:
                cv2.putText(mini_court_img, "VERIFYING", (9, 16),
                            cv2.FONT_HERSHEY_SIMPLEX, .33, (40, 190, 240), 1, cv2.LINE_AA)
            if mini_court_img.shape[:2] != (mc_h, mc_w):
                mini_court_img = cv2.resize(mini_court_img, (mc_w, mc_h), interpolation=cv2.INTER_AREA)

            cv2.rectangle(canvas, (ox - 4, oy - 4), (ox + mc_w + 4, oy + mc_h + 18), (16, 20, 26), -1)
            cv2.rectangle(canvas, (ox - 4, oy - 4), (ox + mc_w + 4, oy + mc_h + 18), (60, 80, 100), 1)
            cv2.putText(canvas, "2D MINI-COURT", (ox + 2, oy + mc_h + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32 * mc_w / 100, (160, 200, 220), 1, cv2.LINE_AA)
            canvas[oy:oy+mc_h, ox:ox+mc_w] = mini_court_img

            # 5. 頂部狀態 HUD
            hud_bg = canvas[8:44, 8:430]
            if hud_bg.shape[0] > 0 and hud_bg.shape[1] > 0:
                dark_overlay = np.zeros_like(hud_bg)
                cv2.addWeighted(dark_overlay, 0.65, hud_bg, 0.35, 0, hud_bg)
                canvas[8:44, 8:430] = hud_bg
            cv2.rectangle(canvas, (8, 8), (430, 44), (60, 80, 90), 1, cv2.LINE_AA)
            court_status = "VERIFYING (LINE CALLS PAUSED)" if verifying else method.upper()
            hud_text = f"ETN-B0 Court-Aware | Court: {court_status} | Bounces: {len(past_bounces)} | {f_idx+1}/{total}"
            if width < 600:
                hud_text = f"Court: {court_status} | Bounces: {len(past_bounces)} | {f_idx+1}/{total}"
            text_width = cv2.getTextSize(hud_text, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)[0][0]
            hud_scale = 0.44 * min(1.0, max(1, width - 32) / max(1, text_width))
            cv2.putText(canvas, hud_text, (16, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, hud_scale, (230, 255, 240), 1, cv2.LINE_AA)

            writer.write(canvas)

            if progress_cb and f_idx % 15 == 0:
                p2 = 0.5 + (f_idx / total) * 0.5
                progress_cb(min(p2, 0.99))

    finally:
        cap.release()
        writer.release()

    # ── FFmpeg H.264 轉碼 (支援瀏覽器硬體加速與流暢 60 FPS 播放，避免 mp4v 卡頓) ──
    if getattr(writer, "needs_transcode", True):
        try:
            tmp_h264 = str(out_path) + ".h264.mp4"
            cmd = [
                "ffmpeg", "-y", "-i", str(out_path),
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                tmp_h264
            ]
            res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
            if res.returncode == 0 and os.path.exists(tmp_h264) and os.path.getsize(tmp_h264) > 1000:
                os.replace(tmp_h264, str(out_path))
                print(f"[FFmpeg] Successfully transcoded to H.264 with +faststart: {out_path}")
        except Exception as e:
            print(f"[FFmpeg Warning] Transcoding failed: {e}")

    if progress_cb:
        progress_cb(1.0)

    return bounces, court_info


# ═══════════════════════════════════════════════════════════════
#  API 路由
# ═══════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={"request": request})


@app.post("/upload")
async def upload(bg: BackgroundTasks, file: UploadFile = File(...), limit_to_source_fps: bool = Form(False)):
    ext = Path(file.filename).suffix.lower()
    if ext not in {".mp4", ".avi", ".mov", ".mkv"}:
        raise HTTPException(400, "請上傳 mp4 / avi / mov / mkv 格式影片")

    jid      = str(uuid.uuid4())[:8]
    in_path  = str(UPLOAD_DIR / f"{jid}_in{ext}")
    out_path = str(OUTPUT_DIR / f"{jid}_out.mp4")
    court_img_path = str(OUTPUT_DIR / f"{jid}_court.jpg")

    with open(in_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    t_start = time.time()
    job_status[jid] = {
        "status": "processing",
        "progress": 0,
        "bounces": [],
        "court": None,
        "start_time": t_start,
        "elapsed_sec": 0.0,
        "phase": "analysis",
        "limit_to_source_fps": limit_to_source_fps,
        "source_fps": None,
        "analysis_fps": 0.0,
        "analysis_average_fps": 0.0,
        "analysis_frames": 0,
        "speed_revision": 0,
    }

    def run():
        try:
            def on_progress(p):
                elapsed = round(time.time() - t_start, 1)
                job_status[jid].update(progress=round(p * 100, 1), elapsed_sec=elapsed)

            verified_geometry = load_verified_court(Path(in_path), VERIFIED_COURT_DIR)
            court_options = {"court_geometry": verified_geometry} if verified_geometry is not None else {}
            bounces, court_info = process_video(
                in_path, out_path,
                on_progress,
                court_debug_path=court_img_path,
                limit_to_source_fps=lambda: job_status[jid]["limit_to_source_fps"],
                analysis_status_cb=lambda state: job_status[jid].update(state),
                **court_options,
            )
            t_end = time.time()
            duration_sec = round(t_end - t_start, 2)
            res = {
                "status": "done",
                "progress": 100.0,
                "bounces": bounces,
                "court": court_info,
                "duration_sec": duration_sec,
                "elapsed_sec": duration_sec,
                "phase": "done",
                "limit_to_source_fps": job_status[jid]["limit_to_source_fps"],
                "source_fps": job_status[jid]["source_fps"],
                "analysis_fps": job_status[jid]["analysis_fps"],
                "analysis_average_fps": job_status[jid]["analysis_average_fps"],
                "analysis_frames": job_status[jid]["analysis_frames"],
            }
            job_status[jid].update(res)
            # 儲存持久化 metadata
            meta_path = OUTPUT_DIR / f"{jid}_meta.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(res, f)
        except Exception as e:
            job_status[jid].update(status="error", message=str(e))
        finally:
            Path(in_path).unlink(missing_ok=True)

    bg.add_task(run)
    return {"job_id": jid, "limit_to_source_fps": limit_to_source_fps}


class AnalysisSpeedRequest(BaseModel):
    limit_to_source_fps: StrictBool


@app.post("/speed/{jid}")
async def set_analysis_speed(jid: str, request: AnalysisSpeedRequest):
    state = job_status.get(jid)
    if state is None:
        raise HTTPException(404, "找不到此任務")
    if state.get("status") != "processing" or state.get("phase") != "analysis":
        raise HTTPException(409, "模型分析已結束；落點計算與影片輸出保持不限速")
    state["limit_to_source_fps"] = request.limit_to_source_fps
    state["analysis_fps"] = 0.0
    state["speed_revision"] = state.get("speed_revision", 0) + 1
    return {key: state.get(key) for key in
            ("status", "phase", "limit_to_source_fps", "source_fps", "analysis_fps", "speed_revision")}


@app.get("/status/{jid}")
async def status(jid: str):
    if jid not in job_status:
        meta_path = OUTPUT_DIR / f"{jid}_meta.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                job_status[jid] = json.load(f)
        else:
            raise HTTPException(404, "找不到此任務")
    elif job_status[jid].get("status") == "processing":
        st = job_status[jid].get("start_time")
        if st:
            job_status[jid]["elapsed_sec"] = round(time.time() - st, 1)
    return job_status[jid]


@app.get("/court_img/{jid}")
async def court_img(jid: str):
    p = OUTPUT_DIR / f"{jid}_court.jpg"
    if not p.exists():
        raise HTTPException(404, "尚未生成球場校準圖")
    return FileResponse(str(p), media_type="image/jpeg")


@app.get("/download/{jid}")
async def download(jid: str):
    p = OUTPUT_DIR / f"{jid}_out.mp4"
    if not p.exists():
        raise HTTPException(404, "輸出影片尚未完成")
    return FileResponse(str(p), media_type="video/mp4",
                        filename=f"tennis_court_tracked_{jid}.mp4")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8018)
