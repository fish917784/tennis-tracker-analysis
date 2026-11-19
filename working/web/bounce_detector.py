from __future__ import annotations
"""
CatBoost ML Tennis Ball Bounce Detector
=============================================================================
使用 CatBoost Regressor 深度回歸模型進行網球落地彈跳點偵測。
特徵工程：
1. 提取球體軌跡在時序上的前後時間差 (x_lag, y_lag, x_lag_inv, y_lag_inv)。
2. 計算位移絕對差值與反向差值 (diff & diff_inv)。
3. 計算位移比值 (x_div, y_div) 以捕捉觸地瞬態法向減速與垂直動量反轉。
4. 結合 ITF 標準球場幾何透視投影 (Homography Projection)，過濾非比賽場地之發球準備拍球或出鏡雜訊。
"""

import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Set

import numpy as np
import pandas as pd
import cv2

try:
    import catboost as ctb
    from scipy.interpolate import CubicSpline
    from scipy.spatial import distance
    _CATBOOST_AVAILABLE = True
except ImportError:
    _CATBOOST_AVAILABLE = False

_LOCAL_MODEL = Path(__file__).resolve().parent.parent / "models" / "ctb_regr_bounce.cbm"
BOUNCE_MODEL_PATH = os.environ.get("TENNIS_BOUNCE_MODEL", str(_LOCAL_MODEL))
_detector_instance: Optional[CatBoostBounceDetector] = None


class CatBoostBounceDetector:
    def __init__(self, model_path: str = BOUNCE_MODEL_PATH, threshold: float = 0.45):
        self.model_path = model_path
        self.threshold = threshold
        self.model: Optional[ctb.CatBoostRegressor] = None
        self._load_model()

    def _load_model(self):
        if not _CATBOOST_AVAILABLE:
            print("[WARN] catboost 或 scipy 模組未安裝，無法啟用 CatBoost 落點偵測")
            return
        if not os.path.exists(self.model_path):
            print(f"[WARN] 找不到 CatBoost 模型檔案：{self.model_path}")
            return
        try:
            self.model = ctb.CatBoostRegressor()
            self.model.load_model(self.model_path)
            print(f"[INFO] 成功載入 CatBoost 落點回歸模型：{self.model_path}")
        except Exception as e:
            print(f"[WARN] 載入 CatBoost 模型失敗：{e}")
            self.model = None

    def is_ready(self) -> bool:
        return self.model is not None

    def prepare_features(self, x_ball: list, y_ball: list) -> Tuple[pd.DataFrame, list]:
        labels = pd.DataFrame({
            "frame": range(len(x_ball)),
            "x-coordinate": np.asarray(x_ball, dtype=np.float64),
            "y-coordinate": np.asarray(y_ball, dtype=np.float64),
        })

        num = 3
        eps = 1e-15
        for i in range(1, num):
            labels[f"x_lag_{i}"] = labels["x-coordinate"].shift(i)
            labels[f"x_lag_inv_{i}"] = labels["x-coordinate"].shift(-i)
            labels[f"y_lag_{i}"] = labels["y-coordinate"].shift(i)
            labels[f"y_lag_inv_{i}"] = labels["y-coordinate"].shift(-i)
            labels[f"x_diff_{i}"] = (labels[f"x_lag_{i}"] - labels["x-coordinate"]).abs()
            labels[f"y_diff_{i}"] = labels[f"y_lag_{i}"] - labels["y-coordinate"]
            labels[f"x_diff_inv_{i}"] = (labels[f"x_lag_inv_{i}"] - labels["x-coordinate"]).abs()
            labels[f"y_diff_inv_{i}"] = labels[f"y_lag_inv_{i}"] - labels["y-coordinate"]
            labels[f"x_div_{i}"] = (labels[f"x_diff_{i}"] / (labels[f"x_diff_inv_{i}"] + eps)).abs()
            labels[f"y_div_{i}"] = labels[f"y_diff_{i}"] / (labels[f"y_diff_inv_{i}"] + eps)

        # Both coordinates must exist throughout the five-frame window.
        required = ["x-coordinate", "y-coordinate"]
        for i in range(1, num):
            required.extend([f"{axis}_{lag}_{i}" for axis in ("x", "y") for lag in ("lag", "lag_inv")])
        labels = labels[np.isfinite(labels[required].astype(float)).all(axis=1)]

        colnames_x = (
            [f"x_diff_{i}" for i in range(1, num)] +
            [f"x_diff_inv_{i}" for i in range(1, num)] +
            [f"x_div_{i}" for i in range(1, num)]
        )
        colnames_y = (
            [f"y_diff_{i}" for i in range(1, num)] +
            [f"y_diff_inv_{i}" for i in range(1, num)] +
            [f"y_div_{i}" for i in range(1, num)]
        )
        colnames = colnames_x + colnames_y

        finite = np.isfinite(labels[colnames]).all(axis=1)
        labels = labels[finite]
        features = labels[colnames]
        return features, list(labels["frame"])

    def smooth_predictions(self, x_ball: list, y_ball: list) -> Tuple[list, list]:
        x_copy = list(x_ball)
        y_copy = list(y_ball)
        is_none = [int(x is None or y is None or not np.isfinite(x) or not np.isfinite(y))
                   for x, y in zip(x_copy, y_copy)]
        for i, missing in enumerate(is_none):
            if missing:
                x_copy[i], y_copy[i] = None, None
        interp = 5
        counter = 0
        for num in range(interp, len(x_copy) - 1):
            if is_none[num] and sum(is_none[num - interp:num]) == 0 and counter < 3:
                x_ext, y_ext = self.extrapolate(x_copy[num - interp:num], y_copy[num - interp:num])
                x_copy[num] = x_ext
                y_copy[num] = y_ext
                is_none[num] = 0
                if not is_none[num + 1]:
                    dist = distance.euclidean((x_ext, y_ext), (x_copy[num + 1], y_copy[num + 1]))
                    if dist > 80:
                        x_copy[num + 1], y_copy[num + 1], is_none[num + 1] = None, None, 1
                counter += 1
            else:
                counter = 0
        return x_copy, y_copy

    def extrapolate(self, x_coords: list, y_coords: list) -> Tuple[float, float]:
        xs = list(range(len(x_coords)))
        func_x = CubicSpline(xs, x_coords, bc_type="natural")
        x_ext = func_x(len(x_coords))
        func_y = CubicSpline(xs, y_coords, bc_type="natural")
        y_ext = func_y(len(x_coords))
        return float(x_ext), float(y_ext)

    def postprocess(self, ind_bounce: np.ndarray, preds: np.ndarray, frame_numbers=None) -> list:
        if len(ind_bounce) == 0:
            return []
        ind_bounce_filtered = [ind_bounce[0]]
        for i in range(1, len(ind_bounce)):
            previous = ind_bounce[i - 1]
            current = ind_bounce[i]
            previous_frame = previous if frame_numbers is None else frame_numbers[previous]
            current_frame = current if frame_numbers is None else frame_numbers[current]
            if current_frame - previous_frame != 1:
                cur_ind = ind_bounce[i]
                ind_bounce_filtered.append(cur_ind)
            elif preds[current] > preds[ind_bounce_filtered[-1]]:
                ind_bounce_filtered[-1] = ind_bounce[i]
        return ind_bounce_filtered

    def predict_bounces(self, x_ball: list, y_ball: list, smooth: bool = True) -> Dict[int, float]:
        """回傳 {frame_idx: confidence_score}"""
        if not self.is_ready():
            return {}
        if smooth:
            x_ball, y_ball = self.smooth_predictions(x_ball, y_ball)
        features, num_frames = self.prepare_features(x_ball, y_ball)
        if len(features) == 0:
            return {}

        preds = self.model.predict(features)
        ind_bounce = np.where(preds > self.threshold)[0]
        if len(ind_bounce) > 0:
            ind_bounce = self.postprocess(ind_bounce, preds, frame_numbers=num_frames)

        result = {}
        for idx in ind_bounce:
            f = num_frames[idx]
            result[f] = float(preds[idx])
        return result


def get_bounce_detector() -> Optional[CatBoostBounceDetector]:
    global _detector_instance
    if _detector_instance is None and os.path.exists(BOUNCE_MODEL_PATH):
        _detector_instance = CatBoostBounceDetector(BOUNCE_MODEL_PATH)
    return _detector_instance


def detect_court_bounces_catboost(
    trajectory: list[dict], H: Optional[np.ndarray], H_inv: Optional[np.ndarray],
    fps: float = 30.0, court_states: Optional[list[dict]] = None,
) -> list[dict]:
    """Predict within verified camera segments and project with each frame's H.

    Legacy callers may supply one verified H/H_inv pair. Web callers supply
    court_states so no interpolation or feature window crosses a camera cut
    or a period without a verified court.
    """
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Bounce detection requires a positive frame rate.")
    total = len(trajectory)
    if court_states is None:
        calibrated = _valid_homography(H) and _valid_homography(H_inv)
        court_states = [dict(H=H, H_inv=H_inv, calibrated=calibrated, segment_id=0)] * total
    if len(court_states) != total:
        raise ValueError("Each trajectory frame must have a matching court state.")
    if total < 5:
        return []

    detector = get_bounce_detector()
    bounces = []
    # Debouncing is a heuristic, not a physical lower bound on all bounces.
    min_gap = max(1, int(fps * 0.40))
    start = 0
    while start < total:
        state = court_states[start]
        if not _valid_court_state(state):
            start += 1
            continue
        segment_id = state.get("segment_id", 0)
        end = start + 1
        while (end < total and _valid_court_state(court_states[end])
               and court_states[end].get("segment_id", 0) == segment_id):
            end += 1
        if end - start < 5:
            start = end
            continue

        points = trajectory[start:end]
        states = court_states[start:end]
        xs = [pt["x"] if pt["visible"] else None for pt in points]
        ys = [pt["y"] if pt["visible"] else None for pt in points]
        if detector is not None and detector.is_ready():
            candidates = detector.predict_bounces(xs, ys, smooth=True)
        else:
            fallback = _fallback_kinematic_bounces(
                points, state["H"], state["H_inv"], fps, court_states=states,
            )
            candidates = {b["frame"]: b.get("confidence", 0.0) for b in fallback}

        valid = []
        for local_frame, score in candidates.items():
            if not 0 <= local_frame < len(points):
                continue
            px, py = xs[local_frame], ys[local_frame]
            if px is None or py is None or not np.isfinite([px, py, score]).all():
                continue
            matrix = np.asarray(states[local_frame]["H"], dtype=np.float64)
            projected = matrix @ np.array([px, py, 1.0])
            if not np.isfinite(projected).all() or abs(projected[2]) <= 1e-9:
                continue
            xm, ym = (projected[:2] / projected[2]).tolist()
            if -2.0 <= xm <= 13.0 and -2.0 <= ym <= 26.0:
                valid.append((local_frame, float(score), px, py, xm, ym))

        # Keep the strongest candidate in the refractory neighborhood.
        selected = []
        for candidate in sorted(valid, key=lambda item: (-item[1], item[0])):
            if all(abs(candidate[0] - kept[0]) >= min_gap for kept in selected):
                selected.append(candidate)
        for local_frame, score, px, py, xm, ym in sorted(selected):
            f = start + local_frame
            is_in = (-0.20 <= xm <= 11.17) and (-0.20 <= ym <= 23.97)
            time_s = round(f / fps, 2)
            mins = int(time_s // 60)
            secs = time_s % 60
            bounces.append({
                "id": len(bounces) + 1,
                "frame": int(f),
                "x": int(round(px)),
                "y": int(round(py)),
                "court_x": round(xm, 2),
                "court_y": round(ym, 2),
                "is_in": is_in,
                "decision": "IN (界內)" if is_in else "OUT (界外)",
                "time_sec": time_s,
                "timestamp": f"{mins:02d}:{secs:05.2f}",
                "confidence": round(score, 3),
                "segment_id": segment_id,
            })
        start = end
    return bounces


def _valid_homography(matrix) -> bool:
    if matrix is None:
        return False
    value = np.asarray(matrix, dtype=np.float64)
    return (value.shape == (3, 3) and np.isfinite(value).all()
            and abs(np.linalg.det(value)) > 1e-12)


def _valid_court_state(state: dict) -> bool:
    return (bool(state.get("calibrated", False)) and _valid_homography(state.get("H"))
            and _valid_homography(state.get("H_inv")))


def _fallback_kinematic_bounces(
    trajectory: list[dict], H: np.ndarray, H_inv: np.ndarray, fps: float = 30.0,
    court_states: Optional[list[dict]] = None,
) -> list[dict]:
    """運動學轉折點備援演算法"""
    total = len(trajectory)
    xs = np.array([pt["x"] for pt in trajectory], dtype=np.float32)
    ys = np.array([pt["y"] for pt in trajectory], dtype=np.float32)
    vis = np.array([pt["visible"] for pt in trajectory], dtype=bool)

    pt_net_m = np.array([[[5.485, 11.885]]], dtype=np.float32)
    matrices = [H_inv] * total if court_states is None else [state["H_inv"] for state in court_states]
    net_ys = [float(cv2.perspectiveTransform(pt_net_m, matrix)[0, 0, 1]) for matrix in matrices]

    segments = []
    seg_start = 0
    cur_side = "near" if (vis[0] and ys[0] >= net_ys[0]) else "far"

    for i in range(1, total):
        if not vis[i]: continue
        side = "near" if ys[i] >= net_ys[i] else "far"
        if side != cur_side:
            future = ["near" if ys[k] >= net_ys[k] else "far" for k in range(i, min(total, i+4)) if vis[k]]
            if future.count(side) >= 2:
                segments.append((seg_start, i, cur_side))
                seg_start = i
                cur_side = side
    segments.append((seg_start, total, cur_side))

    bounces = []
    for s_idx, (s_start, s_end, side) in enumerate(segments):
        if s_end - s_start < 5: continue
        if s_idx == 0 and s_start == 0 and side == "far": continue

        best_f = None
        if side == "near":
            cands = []
            for k in range(s_start + 1, s_end - 1):
                if not vis[k]: continue
                p_k, n_k = max(s_start, k - 2), min(s_end - 1, k + 2)
                dy_in = (ys[k] - ys[p_k]) / max(1, k - p_k)
                dy_out = (ys[n_k] - ys[k]) / max(1, n_k - k)
                if (dy_in > 0.4 and dy_out <= 0.0) or (dy_in > 0.8 and dy_out < 0.2):
                    cands.append(k)
            best_f = cands[0] if cands else max([k for k in range(s_start, s_end) if vis[k]], key=lambda k: ys[k], default=None)
        else:
            cands = []
            for k in range(s_start + 2, s_end - 2):
                if not vis[k]: continue
                p_k, n_k = max(s_start, k - 2), min(s_end - 1, k + 2)
                dy_in = (ys[k] - ys[p_k]) / max(1, k - p_k)
                dy_out = (ys[n_k] - ys[k]) / max(1, n_k - k)
                if (dy_in > 0.2 and dy_out <= -0.4) or (dy_in >= 0.0 and dy_out <= -0.8):
                    cands.append(k)
            best_f = cands[0] if cands else max([k for k in range(s_start + (s_end-s_start)//3, s_end) if vis[k]], key=lambda k: ys[k], default=None)

        if best_f is not None:
            bx, by = float(xs[best_f]), float(ys[best_f])
            pt_img = np.array([[[bx, by]]], dtype=np.float32)
            matrix = H if court_states is None else court_states[best_f]["H"]
            pt_court = cv2.perspectiveTransform(pt_img, matrix)[0][0]
            xm, ym = float(pt_court[0]), float(pt_court[1])
            if -2.0 <= xm <= 13.0 and -2.0 <= ym <= 26.0:
                is_in = (-0.20 <= xm <= 11.17) and (-0.20 <= ym <= 23.97)
                time_s = round(best_f / fps, 2)
                mins = int(time_s // 60)
                secs = time_s % 60
                bounces.append({
                    "id": len(bounces) + 1,
                    "frame": int(best_f),
                    "x": int(round(bx)),
                    "y": int(round(by)),
                    "court_x": round(xm, 2),
                    "court_y": round(ym, 2),
                    "is_in": is_in,
                    "decision": "IN (界內)" if is_in else "OUT (界外)",
                    "time_sec": time_s,
                    "timestamp": f"{mins:02d}:{secs:05.2f}"
                })
    return bounces
