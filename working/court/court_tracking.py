"""Follow visible court markings through consecutive broadcast frames."""

from collections import deque
from math import isfinite
from typing import Optional

import cv2
import numpy as np

if __package__:
    from .court_detector import CourtDetection, _line_mask, detect_court, _detect_at_width
    from .court_geometry import COURT_CORNERS, COURT_SEGMENTS, sample_segments, score_templates, score_templates_multi
    from .court_partial import detect_partial_court, partial_scores, refine_partial_court
else:
    from court_detector import CourtDetection, _line_mask, detect_court, _detect_at_width
    from court_geometry import COURT_CORNERS, COURT_SEGMENTS, sample_segments, score_templates, score_templates_multi
    from court_partial import detect_partial_court, partial_scores, refine_partial_court


_SAMPLES, _SEGMENT_IDS = sample_segments(0.55)
_FLOW_SETTINGS = dict(winSize=(19, 19), maxLevel=2,
                      criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 16, 0.03))


def court_view_changed(previous, current):
    """Detect edits even when the lower court keeps a similar solid color."""
    difference = cv2.absdiff(current, previous)
    mean = cv2.mean(difference)[0]
    changed = np.count_nonzero(difference > 29) / difference.size
    if mean > 21 and changed > .31:
        return True
    upper = difference[:max(1, difference.shape[0] // 2)]
    return bool(mean > 21 and changed > .20 and cv2.mean(upper)[0] > 30
                and np.count_nonzero(upper > 29) > .30 * upper.size)


class CourtTracker:
    """Follow a court verified against the markings visible in the current view."""

    def __init__(self, fps: float):
        if not isfinite(fps) or fps <= 0:
            raise ValueError("Court tracking requires a positive frame rate.")
        self.acquisitions = 0
        self.cuts = 0
        self._frame_number = -1
        self._track_interval = max(1, round(fps / 9))
        self._retry_interval = max(1, round(fps * 1.2))
        self._cut_retry_interval = max(1, round(fps * 0.25))
        self._candidate_retry_interval = max(1, round(fps * 0.12))
        self._candidate_max_age = max(1, round(fps * 2.0))
        self._pending_candidate = None
        self._pending_count = 0
        self._pending_frame = -1
        self._weak_scene = False
        self._last_acquisition = -self._retry_interval
        self._next_update = 0
        self._previous_thumbnail = None
        self._track_gray = None
        self._result = None
        self._recent = deque(maxlen=5)
        self._recent_views = deque(maxlen=3)
        self._shape = None

    def update(self, frame: np.ndarray, frame_idx: Optional[int] = None) -> Optional[CourtDetection]:
        """Schedule work in source frames, even when the caller submits sparsely.

        Omitting frame_idx preserves the consecutive-frame CLI interface. A
        source rewind starts a new stream and discards all previous geometry.
        """
        next_frame = self._frame_number + 1 if frame_idx is None else int(frame_idx)
        if next_frame < 0:
            raise ValueError("Court frame index must be non-negative.")
        if next_frame <= self._frame_number:
            self._previous_thumbnail = None
            self._recent.clear()
            self._recent_views.clear()
            self._reset_tracking()
            self._next_update = next_frame
            self._last_acquisition = next_frame - self._retry_interval
        self._frame_number = next_frame
        height, width = frame.shape[:2]
        if (height, width) != self._shape:
            self._shape = (height, width)
            self._previous_thumbnail = None
            self._recent.clear()
            self._recent_views.clear()
            self._reset_tracking()
            self._next_update = self._frame_number

        thumbnail = cv2.resize(frame, (96, 54), interpolation=cv2.INTER_LINEAR)
        thumbnail = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2GRAY)
        cut = False
        if self._previous_thumbnail is not None:
            cut = court_view_changed(self._previous_thumbnail, thumbnail)
        self._previous_thumbnail = thumbnail
        if cut:
            self.cuts += 1
            self._reset_tracking(remember=True)
            self._next_update = self._frame_number

        if self._frame_number < self._next_update:
            return self._result

        small_width = min(width, 608)
        small_height = max(1, round(height * small_width / width))
        small = cv2.resize(frame, (small_width, small_height), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        unit = max(small_width / 960, 0.4)
        markings = _line_mask(gray, unit)
        distance = cv2.distanceTransform(cv2.bitwise_not(markings), cv2.DIST_L2, cv2.DIST_MASK_3)
        scale = np.diag([small_width / width, small_height / height, 1.0])

        if self._result is not None:
            previous_result = self._result
            tracked = self._track(gray, distance, scale)
            if tracked is None and self._track_gray is not None:
                # Keep the matching last verified view for motion recovery.
                # It still has to pass the current image's line checks.
                self._recent_views.appendleft((previous_result, self._track_gray.copy()))
            self._result = tracked

        if self._result is None:
            self._result = self._recover(frame, distance, scale, cut, gray=gray)

        self._track_gray = gray if self._result is not None else None
        self._next_update = self._frame_number + self._track_interval
        return self._result

    def _reset_tracking(self, remember=False) -> None:
        if remember and self._result is not None and self._track_gray is not None:
            self._recent_views.appendleft((self._result, self._track_gray.copy()))
        self._result = None
        self._track_gray = None
        self._pending_candidate = None
        self._pending_count = 0
        self._pending_frame = -1
        self._weak_scene = False

    @staticmethod
    def _line_scores(matrices: np.ndarray, distance: np.ndarray):
        supports, visibility = score_templates_multi(
            matrices, distance, _SAMPLES, _SEGMENT_IDS, (2.0, 0.75)
        )
        support, close_support = supports
        coverage = support * visibility
        score = coverage.mean(axis=1)
        close_score = (close_support * visibility).mean(axis=1)
        pattern = (
            ((coverage[:, :4] >= 0.60).sum(axis=1) >= 3)
            & ((coverage[:, 4:8] >= 0.60).sum(axis=1) >= 3)
            & (coverage[:, 8:10].max(axis=1) >= 0.63)
        )
        return score, close_score, pattern

    @staticmethod
    def _motion(previous: np.ndarray, gray: np.ndarray, old_matrix: np.ndarray) -> Optional[np.ndarray]:
        mask = np.zeros_like(gray)
        projected = cv2.perspectiveTransform(COURT_SEGMENTS, old_matrix)
        for start, end in np.rint(projected).astype(int):
            cv2.line(mask, tuple(start), tuple(end), 255, 13)
        feature_points = cv2.goodFeaturesToTrack(previous, 60, 0.015, 7, mask=mask, blockSize=5)
        if feature_points is None or len(feature_points) < 8:
            return None

        moved, valid, error = cv2.calcOpticalFlowPyrLK(previous, gray, feature_points, None, **_FLOW_SETTINGS)
        if moved is None:
            return None
        keep = valid.ravel().astype(bool) & (error.ravel() < 27)
        if keep.sum() < 8:
            return None

        original = feature_points.reshape(-1, 2)[keep]
        current = moved.reshape(-1, 2)[keep]
        motion, inliers = cv2.estimateAffinePartial2D(original, current, method=cv2.RANSAC,
                                                     ransacReprojThreshold=1.6, maxIters=250)
        if motion is None or np.count_nonzero(inliers) < max(8, round(0.60 * len(original))):
            return None
        points = original[inliers.ravel().astype(bool)]
        if np.any(np.ptp(points, axis=0) < np.array(gray.shape[::-1]) * 0.12):
            return None
        zoom = np.hypot(*motion[0, :2])
        if not 0.87 <= zoom <= 1.15 or np.linalg.norm(motion[:, 2]) > gray.shape[1] * 0.15:
            return None
        return np.vstack((motion, [0.0, 0.0, 1.0]))

    def _track(self, gray: np.ndarray, distance: np.ndarray, scale: np.ndarray) -> Optional[CourtDetection]:
        if self._result.partial:
            return self._track_partial(gray, distance, scale)
        old_matrix = scale @ self._result.court_to_image
        scores, close, pattern = self._line_scores(old_matrix[None], distance)
        if pattern[0] and scores[0] >= 0.97 and close[0] >= 0.88:
            return self._result

        usable = _tracking_support(scores[0], close[0], pattern[0], established=True)
        aligned = _refine_matrix(old_matrix, distance, established=True)
        if aligned is None:
            aligned = _refine_projective(old_matrix, distance)
        if aligned is not None:
            adjusted, adjusted_score, adjusted_close = aligned
            if not usable or adjusted_score + .30 * adjusted_close > scores[0] + .30 * close[0] + .012:
                return self._make_result(np.linalg.solve(scale, adjusted), adjusted_score, adjusted_close)
        motion = self._motion(self._track_gray, gray, old_matrix)
        if motion is not None:
            candidate = motion @ old_matrix
            shifted, shifted_close, shifted_pattern = self._line_scores(candidate[None], distance)
            improved = shifted[0] + 0.30 * shifted_close[0] > scores[0] + 0.30 * close[0] + 0.012
            if _tracking_support(shifted[0], shifted_close[0], shifted_pattern[0], established=True) and (improved or not usable):
                return self._make_result(np.linalg.solve(scale, candidate), shifted[0], shifted_close[0])
        if not usable:
            # Low camera angles can move several pixels between sparse
            # updates. Distributed matches recover that movement when the
            # narrow line-mask optical flow has too few reliable features.
            motion = self._view_motion(self._track_gray, gray)
            if motion is not None:
                aligned = _refine_matrix(motion @ old_matrix, distance, established=True)
                if aligned is not None:
                    matrix, score, close = aligned
                    return self._make_result(np.linalg.solve(scale, matrix), score, close)
        return self._result if usable else None

    def _track_partial(self, gray, distance, scale):
        old_matrix = scale @ self._result.court_to_image
        score, close, _, valid = partial_scores(old_matrix[None], distance, established=True)
        if valid[0] and score[0] >= .96 and close[0] >= .88:
            return self._result
        aligned = refine_partial_court(old_matrix, distance, established=True)
        if aligned is None or aligned[2] < .78:
            motion = self._motion(self._track_gray, gray, old_matrix)
            if motion is None and aligned is None:
                motion = self._view_motion(self._track_gray, gray)
            if motion is not None:
                shifted = refine_partial_court(motion @ old_matrix, distance, established=True)
                if shifted is not None and (aligned is None or shifted[1] + .3 * shifted[2] > aligned[1] + .3 * aligned[2] + .02):
                    aligned = shifted
        if aligned is None:
            # Rediscover the same court if accumulated camera tilt cannot
            # be undone by a translation or by reliable point matches.
            candidate = detect_partial_court(gray, distance)
            if (candidate is None or candidate[1] < .92 or candidate[2] < .77
                    or not court_poses_consistent(candidate[0], old_matrix, gray.shape, max_shift=.019)):
                return None
            aligned = candidate
        matrix, score, close = aligned
        return self._make_result(np.linalg.solve(scale, matrix), score, close, partial=True)

    @staticmethod
    def _view_motion(previous, gray):
        """Recover a camera adjustment from distributed descriptor matches."""
        detector = cv2.ORB_create(nfeatures=800, edgeThreshold=15, fastThreshold=12)
        old_points, old_desc = detector.detectAndCompute(previous, None)
        new_points, new_desc = detector.detectAndCompute(gray, None)
        if old_desc is None or new_desc is None:
            return None
        pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(old_desc, new_desc, k=2)
        matches = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < .72 * pair[1].distance]
        if len(matches) < 16:
            return None
        source = np.float32([old_points[item.queryIdx].pt for item in matches])
        target = np.float32([new_points[item.trainIdx].pt for item in matches])
        motion, inliers = cv2.findHomography(source, target, cv2.RANSAC, 2.0, maxIters=1200, confidence=.995)
        if motion is None or inliers is None:
            return None
        keep = inliers.ravel().astype(bool)
        if keep.sum() < max(16, round(.50 * len(matches))):
            return None
        bounds = np.array(gray.shape[::-1]) * [.20, .20]
        if np.any(np.ptp(source[keep], axis=0) < bounds) or np.any(np.ptp(target[keep], axis=0) < bounds):
            return None
        height, width = gray.shape
        border = np.float32([[[0, 0], [width, 0], [width, height], [0, height]]])
        projected = cv2.perspectiveTransform(border, motion)[0]
        if not np.isfinite(projected).all() or not cv2.isContourConvex(projected):
            return None
        area = abs(cv2.contourArea(projected)) / (width * height)
        if not .50 <= area <= 2.0:
            return None
        return motion

    @staticmethod
    def _make_result(matrix: np.ndarray, score: float, close_score: float, *, partial=False) -> CourtDetection:
        quad = cv2.perspectiveTransform(COURT_CORNERS[None], matrix)[0]
        return CourtDetection(np.linalg.inv(matrix), matrix, quad, float(score), float(close_score), partial=partial)

    def _remember(self, result: CourtDetection) -> None:
        for known in self._recent:
            if np.max(np.linalg.norm(result.quad - known.quad, axis=1)) <= self._shape[1] * 0.009:
                return
        self._recent.appendleft(result)

    def _recover(self, frame: np.ndarray, distance: np.ndarray, scale: np.ndarray,
                 cut: bool, gray=None) -> Optional[CourtDetection]:
        full = [known for known in self._recent if not known.partial]
        if full:
            candidates = np.asarray([scale @ known.court_to_image for known in full])
            scores, close, pattern = self._line_scores(candidates, distance)
            usable = pattern & (scores >= 0.86) & (close >= 0.61)
            if np.any(usable):
                best = int(np.argmax(np.where(usable, scores + 0.30 * close, -np.inf)))
                return full[best]

        for known in self._recent:
            if known.partial and not cut:
                aligned = refine_partial_court(scale @ known.court_to_image, distance)
                if aligned is not None:
                    matrix, score, close = aligned
                    return self._make_result(np.linalg.solve(scale, matrix), score, close, partial=True)

        # A camera adjustment can trigger a cut without changing the court.
        # Re-establish it only from current feature correspondences and lines;
        # an unrelated close-up cannot reuse an unverified old homography.
        if gray is not None:
            for known, previous in self._recent_views:
                if previous.shape != gray.shape:
                    continue
                old_matrix = scale @ known.court_to_image
                motion = self._motion(previous, gray, old_matrix)
                if motion is None:
                    motion = self._view_motion(previous, gray)
                if motion is None:
                    continue
                candidate = motion @ old_matrix
                aligned = (refine_partial_court(candidate, distance) if known.partial
                           else _refine_matrix(candidate, distance, established=False))
                if aligned is not None and aligned[1] >= .82:
                    candidate, score, close = aligned
                    result = self._make_result(np.linalg.solve(scale, candidate), score, close, partial=known.partial)
                    self._remember(result)
                    return result

        since = self._frame_number - self._last_acquisition
        retry = (self._candidate_retry_interval if self._pending_candidate is not None or self._weak_scene
                 else self._cut_retry_interval if cut else self._retry_interval)
        if since < retry:
            return None
        self._last_acquisition = self._frame_number
        self.acquisitions += 1
        if self._pending_candidate is not None and self._pending_candidate.partial:
            result = self._confirm_cropped_court(frame.shape, gray, distance, scale)
            if result is not None or self._pending_candidate is not None:
                return result
        result = detect_court(frame, max_width=768)
        self._weak_scene = bool(result is not None and result.score >= .83 and result.validation_score >= .74)
        if result is not None and result.score >= 0.94:
            scores, close, pattern = self._line_scores(np.asarray([scale @ result.court_to_image]), distance)
            if pattern[0] and scores[0] >= 0.82 and close[0] >= 0.55 and result.validation_score >= 0.79:
                self._remember(result)
                self._pending_candidate = None
                self._pending_count = 0
                return result
        result = self._confirm_partial_court(frame, distance, scale)
        if result is not None or self._pending_candidate is not None:
            return result
        return self._confirm_cropped_court(frame.shape, gray, distance, scale)

    def _confirm_cropped_court(self, shape, gray, distance, scale):
        """Require matching fits from separate frames for the cropped-court path."""
        fit = detect_partial_court(gray, distance)
        pending = self._pending_candidate
        if fit is None and pending is not None:
            fit = refine_partial_court(scale @ pending.court_to_image, distance)
        if fit is None:
            self._pending_candidate = None
            self._pending_count = 0
            return None
        matrix, score, close = fit
        candidate = self._make_result(np.linalg.solve(scale, matrix), score, close, partial=True)
        if (pending is not None and 0 < self._frame_number - self._pending_frame <= self._candidate_max_age
                and court_poses_consistent(candidate.court_to_image, pending.court_to_image, shape)):
            self._pending_candidate = None
            self._pending_count = 0
            self._remember(candidate)
            return candidate
        self._pending_candidate = candidate
        self._pending_count = 1
        self._pending_frame = self._frame_number
        return None

    def _confirm_partial_court(self, frame, distance, scale):
        """Require independent scales and repeated frames for a weaker view.

        Court-level views can have corners outside the image. They must not
        bypass verification just because the high-angle acquisition score is
        unattainable. Single-scale, single-frame fits are never published.
        """
        height, width = frame.shape[:2]
        clusters = []
        transverse = _long_transverse_lines(distance)
        # Direct detections avoid counting a 768->512 fallback as two scales.
        for resolution in dict.fromkeys(min(width, value) for value in (320, 512, 768)):
            candidate = _detect_at_width(frame, resolution)
            if candidate is None or candidate.score < 0.83 or candidate.validation_score < 0.74:
                continue
            aligned = _refine_matrix(scale @ candidate.court_to_image, distance, established=False)
            if aligned is None or aligned[1] < .82:
                continue
            matrix, score, close = aligned
            if _baseline_beyond(matrix, transverse, distance.shape):
                continue
            candidate = self._make_result(np.linalg.solve(scale, matrix), score, close)
            entry = (candidate, score, close)
            for cluster in clusters:
                if court_poses_consistent(candidate.court_to_image, cluster[0][0].court_to_image, frame.shape):
                    cluster.append(entry)
                    break
            else:
                clusters.append([entry])
        supported = [cluster for cluster in clusters
                     if any(item[2] >= 0.52 for item in cluster)
                     and (len(cluster) >= 2 or len(clusters) == 1)]
        if not supported:
            if self._frame_number - self._pending_frame > self._candidate_max_age:
                self._pending_candidate = None
                self._pending_count = 0
            return None
        cluster = max(supported, key=lambda values: (len(values), np.mean([item[1] + .3 * item[2] for item in values])))
        candidate = max((entry for entry in cluster if entry[2] >= 0.52), key=lambda entry: entry[1] + .3 * entry[2])[0]
        consistent = (self._pending_candidate is not None
                      and 0 < self._frame_number - self._pending_frame <= self._candidate_max_age
                      and court_poses_consistent(candidate.court_to_image, self._pending_candidate.court_to_image, frame.shape))
        self._pending_count = self._pending_count + 1 if consistent else 1
        self._pending_candidate = candidate
        self._pending_frame = self._frame_number
        required = 2 if len(cluster) >= 2 else 3
        if self._pending_count < required:
            return None
        self._pending_candidate = None
        self._pending_count = 0
        self._remember(candidate)
        return candidate


def court_poses_consistent(first, second, shape, *, max_shift=.012):
    """Compare visible markings instead of unstable offscreen corner estimates."""
    height, width = shape[:2]
    a = cv2.perspectiveTransform(_SAMPLES[None, :, :2], np.asarray(first, dtype=np.float64))[0]
    b = cv2.perspectiveTransform(_SAMPLES[None, :, :2], np.asarray(second, dtype=np.float64))[0]
    visible = (np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
               & (a[:, 0] >= 0) & (a[:, 0] < width) & (a[:, 1] >= 0) & (a[:, 1] < height)
               & (b[:, 0] >= 0) & (b[:, 0] < width) & (b[:, 1] >= 0) & (b[:, 1] < height))
    if np.count_nonzero(visible) < 12:
        return False
    distance = np.percentile(np.linalg.norm(a[visible] - b[visible], axis=1), 95)
    return bool(distance <= max_shift * np.hypot(width, height))


def _long_transverse_lines(distance):
    height, width = distance.shape
    mask = (distance < .1).astype(np.uint8) * 255
    lines = cv2.HoughLinesP(mask, 1, np.pi / 360, max(20, round(width * .12)),
                            minLineLength=max(40, round(width * .25)), maxLineGap=3)
    if lines is None:
        return []
    return [line for line in lines.reshape(-1, 4)
            if abs(line[2] - line[0]) >= width * .25
            and abs(line[3] - line[1]) <= .12 * abs(line[2] - line[0])]


def _baseline_beyond(matrix, lines, shape):
    height, _ = shape
    baseline = cv2.perspectiveTransform(COURT_CORNERS[None], matrix)[0][[3, 2]]
    if not np.isfinite(baseline).all() or abs(baseline[1, 0] - baseline[0, 0]) < 1:
        return True
    slope = (baseline[1, 1] - baseline[0, 1]) / (baseline[1, 0] - baseline[0, 0])
    for x0, y0, x1, y1 in lines:
        x, y = (x0 + x1) / 2, (y0 + y1) / 2
        expected = baseline[0, 1] + slope * (x - baseline[0, 0])
        if expected + max(10, height * .05) < y < height * .92:
            return True
    return False


def _tracking_support(score, close_score, pattern, established=False):
    strict = score >= 0.77 and close_score >= 0.52
    # Only an established pose gets tolerance for subpixel aliasing/occlusion;
    # it must retain stronger overall support and the complete line pattern.
    stable = established and score >= 0.82 and close_score >= 0.45
    return bool(pattern and (strict or stable))


def _refine_matrix(matrix, distance, established=False):
    offsets = [(x, y) for y in range(-3, 4) for x in range(-3, 4)]
    offsets.sort(key=lambda item: item[0] ** 2 + item[1] ** 2)
    shifts = np.repeat(np.eye(3, dtype=np.float64)[None], len(offsets), axis=0)
    shifts[:, 0, 2] = [item[0] for item in offsets]
    shifts[:, 1, 2] = [item[1] for item in offsets]
    candidates = shifts @ matrix
    scores, close, pattern = CourtTracker._line_scores(candidates, distance)
    allowed = pattern & (((scores >= .77) & (close >= .52))
                         | (established & (scores >= .82) & (close >= .45)))
    if not np.any(allowed):
        return None
    penalty = np.array([x * x + y * y for x, y in offsets]) * 1e-5
    best = int(np.argmax(np.where(allowed, scores + .30 * close - penalty, -np.inf)))
    return candidates[best], float(scores[best]), float(close[best])


def _refine_projective(matrix, distance):
    """Align a known court after small perspective changes, using current lines.

    Translation alone cannot correct a camera tilt at court level. Bound each
    image-border adjustment to six verification pixels, and retain the same
    full-pattern acceptance criteria used for established tracking.
    """
    initial, _, _ = CourtTracker._line_scores(matrix[None], distance)
    if initial[0] < .60:
        return None
    height, width = distance.shape
    border = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    offsets = np.zeros((4, 2), np.float32)
    for step in (2., 1.):
        for _ in range(12):
            values = [offsets.copy()]
            for coordinate in range(8):
                for change in (-step, step):
                    item = offsets.copy()
                    item.flat[coordinate] = np.clip(item.flat[coordinate] + change, -6, 6)
                    values.append(item)
            values = np.asarray(values)
            matrices = np.asarray([cv2.getPerspectiveTransform(border, border + item) @ matrix for item in values])
            scores, close, pattern = CourtTracker._line_scores(matrices, distance)
            quality = scores + .30 * close - .0002 * (values ** 2).sum(axis=(1, 2))
            best = int(np.argmax(quality))
            if best == 0:
                break
            offsets = values[best]
    adjusted = cv2.getPerspectiveTransform(border, border + offsets) @ matrix
    score, close, pattern = CourtTracker._line_scores(adjusted[None], distance)
    if not _tracking_support(score[0], close[0], pattern[0], established=True):
        return None
    return adjusted, float(score[0]), float(close[0])


def refine_court_geometry(frame: np.ndarray, court_to_image: np.ndarray, established=False, *, partial=False):
    """Verify existing geometry against current markings without reacquiring it.

    A delayed background reply is not evidence of court loss. This check uses
    the tracking criteria for the previously verified type of pose.
    """
    if court_to_image is None or frame is None:
        return None
    matrix = np.asarray(court_to_image, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        return None
    height, width = frame.shape[:2]
    small_width = min(width, 608)
    small_height = max(1, round(height * small_width / width))
    small = cv2.resize(frame, (small_width, small_height), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    markings = _line_mask(gray, max(small_width / 960, 0.4))
    distance = cv2.distanceTransform(cv2.bitwise_not(markings), cv2.DIST_L2, cv2.DIST_MASK_3)
    scale = np.diag([small_width / width, small_height / height, 1.0])
    scaled = scale @ matrix
    if partial:
        aligned = refine_partial_court(scaled, distance, established=established)
    else:
        aligned = _refine_matrix(scaled, distance, established=established)
        if aligned is None and established:
            aligned = _refine_projective(scaled, distance)
    if aligned is None:
        return None
    adjusted, score, close = aligned
    return CourtTracker._make_result(np.linalg.solve(scale, adjusted), score, close, partial=partial)


def court_geometry_is_visible(frame: np.ndarray, court_to_image: np.ndarray, established=False) -> bool:
    return refine_court_geometry(frame, court_to_image, established=established) is not None
