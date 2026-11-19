from __future__ import annotations
"""
Tennis Court Detector
========================================================================
透過共用的球場偵測模組進行網球場全自動幾何定位。
採用 Dirk Farin ITF 標準球場 10 條標線模型配對演算法 (Template Model-Support Scoring)：
1. 影像頂帽變換 (Top-Hat Ridge Mask) 提取微弱或反光白線。
2. Hough 直線聚類提取橫向與縱向邊界候選線段。
3. 16 組種子射影矩陣 (Seed Transforms) 配對 ITF 標準尺寸與雙打外框。
4. 全場地距離變換模型支持度 (Model Support & Validation Score) 評分與 RANSAC 驗證。
5. 多影格共識取樣 (Multi-frame Consensus) 消除選手遮擋或重播鏡頭干擾。
"""

import os
import sys
import math
import importlib.util
import importlib
import types
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import cv2
import numpy as np

# 載入共用的球場偵測模組
CODEX_COURT_DIR = Path(os.environ.get("TENNIS_COURT_DIR", str(Path(__file__).resolve().parent.parent / "court")))

try:
    package_name = "_tennis_court_core"
    package = types.ModuleType(package_name)
    package.__path__ = [str(CODEX_COURT_DIR)]
    sys.modules[package_name] = package
    codex_court = importlib.import_module(f"{package_name}.court_detector")
    CourtTracker = importlib.import_module(f"{package_name}.court_tracking").CourtTracker
    _CODEX_AVAILABLE = True
except Exception as e:
    print(f"[WARN] 無法載入球場偵測模組：{e}")
    _CODEX_AVAILABLE = False
    codex_court = None
    CourtTracker = None

# ITF 標準球場物理尺寸 (公尺)
COURT_WIDTH_M = 10.9728   # 36 ft
COURT_LENGTH_M = 23.7744  # 78 ft

COURT_CORNERS_4_M = np.array([
    [0.0, 0.0],
    [COURT_WIDTH_M, 0.0],
    [COURT_WIDTH_M, COURT_LENGTH_M],
    [0.0, COURT_LENGTH_M]
], dtype=np.float32)


class LiveCourtGeometry:
    """A verified court state for each source frame, with camera boundaries."""

    def __init__(self, fps: float):
        self.tracker = CourtTracker(fps) if _CODEX_AVAILABLE else None
        self.segment_id = 0
        self._calibrated = False
        self._cuts = 0
        self._display_segment_id = 0
        self._display_corners = None
        self._invalid_since = None
        self._grace_frames = max(1, round(fps * .45))
        self._last_frame = -1

    def update(self, frame: np.ndarray, frame_idx: int) -> dict:
        detection = self.tracker.update(frame, frame_idx=frame_idx) if self.tracker is not None else None
        cuts = self.tracker.cuts if self.tracker is not None else 0
        valid = (detection is not None and detection.score >= 0.77
                 and np.isfinite(detection.image_to_court).all()
                 and np.isfinite(detection.court_to_image).all())
        cut = cuts != self._cuts or frame_idx <= self._last_frame
        if cut or valid != self._calibrated:
            self.segment_id += 1
        if cut:
            self._display_corners = None
            self._invalid_since = None
            self._display_segment_id += 1
        if valid:
            self._display_corners = detection.quad.copy()
            self._invalid_since = None
        elif self._display_corners is not None:
            if self._invalid_since is None:
                self._invalid_since = frame_idx
            if frame_idx - self._invalid_since >= self._grace_frames:
                self._display_corners = None
                self._invalid_since = None
                self._display_segment_id += 1
        displayed = self._display_corners is not None
        self._last_frame = frame_idx
        self._cuts, self._calibrated = cuts, valid
        return {
            "H": detection.image_to_court.copy() if valid else None,
            "H_inv": detection.court_to_image.copy() if valid else None,
            "court_poly": detection.quad.astype(np.int32) if valid else None,
            "corners": detection.quad.copy() if valid else None,
            "method": "codex_tracked" if valid else "none",
            "calibrated": bool(valid),
            "segment_id": self.segment_id,
            # Appearance may survive a short occlusion. These fields never
            # supply a homography to projection or bounce prediction.
            "displayed": displayed,
            "verifying": displayed and not valid,
            "display_segment_id": self._display_segment_id,
            "display_corners": self._display_corners.copy() if displayed else None,
            "display_court_poly": self._display_corners.astype(np.int32) if displayed else None,
        }


def get_court_geometry(cap_or_path, width: int, height: int,
                       debug_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """
    全自動取得球場幾何 (H, H_inv, 多邊形, 4角點, 偵測方式字串)
    優先使用共用的球場模型匹配引擎。
    """
    need_release = False
    if isinstance(cap_or_path, str):
        cap = cv2.VideoCapture(cap_or_path)
        need_release = True
    else:
        cap = cap_or_path

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    # 針對長影片進行智能分段取樣 (涵蓋發球起點與各區間)
    if total <= 300:
        sample_indices = np.linspace(2, total - 1, min(12, total), dtype=int)
    else:
        early = np.linspace(15, min(total - 1, 600), 10, dtype=int)
        mid = np.linspace(min(total - 1, 650), min(total - 1, 2000), 6, dtype=int)
        late = np.linspace(min(total - 1, 2100), min(total - 1, 4000), 4, dtype=int)
        sample_indices = np.unique(np.concatenate([early, mid, late]))

    matched_detections = []
    best_frame = None

    if _CODEX_AVAILABLE and codex_court is not None:
        for f_idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f_idx))
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            try:
                det = codex_court.detect_court(frame, max_width=960)
                if det is not None:
                    matched_detections.append((int(f_idx), frame, det))
                    # 若已累積足夠高信賴度影格且分數極高，提早結束取樣以加快速度
                    if len(matched_detections) >= 5 and det.score >= 0.98 and det.validation_score >= 0.98:
                        break
            except Exception as e:
                pass

    if need_release:
        cap.release()

    # 判定共識群集 (Consensus Selection)
    if matched_detections:
        method = "codex_model_match"
        diag = math.hypot(width, height)
        corner_tol = 0.015 * diag  # 角點容差

        # 依角點相近程度分群
        clusters = []
        for idx, f_img, det in matched_detections:
            added = False
            for c in clusters:
                ref_quad = c["items"][0][2].quad
                max_diff = np.max(np.linalg.norm(det.quad - ref_quad, axis=1))
                if max_diff <= corner_tol:
                    c["items"].append((idx, f_img, det))
                    c["scores"].append(det.score + det.validation_score)
                    added = True
                    break
            if not added:
                clusters.append({
                    "items": [(idx, f_img, det)],
                    "scores": [det.score + det.validation_score]
                })

        # 選取支持度最多且分數最高的群集
        clusters.sort(key=lambda c: (len(c["items"]), np.mean(c["scores"])), reverse=True)
        winner_cluster = clusters[0]["items"]

        # 以該群集最高分者作為代表影格與主偵測結果
        best_item = max(winner_cluster, key=lambda x: (x[2].validation_score, x[2].score))
        best_f_idx, best_frame, best_det = best_item

        # 取群集中位數角點以獲得極致亞像素平滑精度
        quads = np.array([x[2].quad for x in winner_cluster], dtype=np.float32)
        corners_img = np.median(quads, axis=0).astype(np.float32)

        # 重新計算正逆單應性矩陣以符合中位數角點
        H, _ = cv2.findHomography(corners_img, COURT_CORNERS_4_M)
        H_inv, _ = cv2.findHomography(COURT_CORNERS_4_M, corners_img)
        court_poly = corners_img.astype(np.int32)

        # 繪製視覺驗證圖
        if debug_path and best_frame is not None:
            dbg = best_frame.copy()
            try:
                dbg = codex_court.draw_overlay(dbg, best_det)
            except Exception:
                cv2.polylines(dbg, [court_poly], True, (76, 230, 75), 2, cv2.LINE_AA)

            colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]
            labels = ["TL", "TR", "BR", "BL"]
            for pt, col, lbl in zip(corners_img, colors, labels):
                cv2.circle(dbg, (int(round(pt[0])), int(round(pt[1]))), 6, col, -1, cv2.LINE_AA)
                cv2.putText(dbg, f"{lbl}({pt[0]:.0f},{pt[1]:.0f})",
                            (int(round(pt[0])) + 8, int(round(pt[1])) - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)

            info_str = f"CODEX MODEL MATCH | Score: {best_det.score:.2f} | Val: {best_det.validation_score:.2f} | Consensus: {len(winner_cluster)}/{len(matched_detections)}"
            cv2.rectangle(dbg, (12, 12), (680, 48), (20, 20, 20), -1)
            cv2.putText(dbg, info_str, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 243, 197), 1, cv2.LINE_AA)
            cv2.imwrite(debug_path, dbg)

        return H, H_inv, court_poly, corners_img, method

    # 若未能檢出，則啟用幾何自適應備援 (Fallback)
    method = "none"
    tl = [width * 0.308, height * 0.247]
    tr = [width * 0.688, height * 0.240]
    br = [width * 0.844, height * 0.736]
    bl = [width * 0.156, height * 0.744]
    corners_img = np.array([tl, tr, br, bl], dtype=np.float32)

    H, H_inv = None, None
    court_poly = corners_img.astype(np.int32)

    return H, H_inv, court_poly, corners_img, method


def detect_court_multi_frame(cap_or_path, n_samples: int = 15, debug_path: Optional[str] = None) -> Optional[np.ndarray]:
    """相容舊版介面：回傳 4 個角點 [TL, TR, BR, BL]"""
    if isinstance(cap_or_path, str):
        cap = cv2.VideoCapture(cap_or_path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        cap.release()
        vpath = cap_or_path
    else:
        w = int(cap_or_path.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        h = int(cap_or_path.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        vpath = cap_or_path

    H, H_inv, court_poly, corners_img, method = get_court_geometry(vpath, w, h, debug_path)
    return corners_img
