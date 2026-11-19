"""Detect ground markings from a single frame of an upright tennis camera."""

from dataclasses import dataclass
from itertools import combinations
from typing import Optional, Tuple

import cv2
import numpy as np

if __package__:
    from .court_geometry import (
        COURT_CORNERS, COURT_SEGMENTS, LONGITUDINAL_X, SEGMENT_COUNT,
        TRANSVERSE_Y, sample_segments, score_templates,
    )
else:
    from court_geometry import (
        COURT_CORNERS, COURT_SEGMENTS, LONGITUDINAL_X, SEGMENT_COUNT,
        TRANSVERSE_Y, sample_segments, score_templates,
    )


@dataclass(frozen=True)
class CourtDetection:
    image_to_court: np.ndarray
    court_to_image: np.ndarray
    quad: np.ndarray
    score: float
    validation_score: float
    partial: bool = False


@dataclass
class LineGroup:
    points: list
    midpoint: np.ndarray
    normal: np.ndarray


def _line_mask(gray: np.ndarray, unit: float) -> np.ndarray:
    radius = max(2, round(5 * unit))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
    ridges = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    return cv2.inRange(ridges, 16, 255)


def _candidate_lines(mask: np.ndarray, unit: float) -> Tuple[np.ndarray, np.ndarray]:
    segments = cv2.HoughLinesP(
        mask, 1, np.pi / 360, max(13, round(29 * unit)),
        minLineLength=max(18, round(39 * unit)), maxLineGap=max(4, round(13 * unit)),
    )
    if segments is None:
        empty = np.empty((0, 3), dtype=np.float32)
        return empty, empty

    endpoints = segments.reshape(-1, 2, 2).astype(np.float32)
    lengths = np.linalg.norm(endpoints[:, 1] - endpoints[:, 0], axis=1)
    groups = []
    merge_distance = max(2.0, 3.8 * unit)
    for index in np.argsort(-lengths)[:340]:
        start, end = endpoints[index]
        delta = (end - start) / lengths[index]
        normal = np.array([-delta[1], delta[0]])
        midpoint = (start + end) / 2
        for group in groups:
            difference = midpoint - group.midpoint
            if (abs(np.dot(normal, group.normal)) > 0.9986
                    and abs(np.dot(difference, normal)) <= merge_distance
                    and abs(np.dot(difference, group.normal)) <= merge_distance):
                group.points.extend((start, end))
                break
        else:
            groups.append(LineGroup([start, end], midpoint, normal))

    transverse = []
    longitudinal = []
    height, width = mask.shape
    for group in groups:
        points = np.asarray(group.points)
        dx, dy, x, y = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        line = (-dy, dx, dy * x - dx * y)
        span = np.ptp(points @ np.array([dx, dy]))
        if abs(dy) <= 0.75 * abs(dx):
            middle_y = y + (width / 2 - x) * dy / dx
            if 0 <= middle_y < height:
                # Preserve candidates at different depths, even if a fence or
                # stands contribute many longer lines in one strip.
                band = int(14 * middle_y / height)
                horizontal_span = span * max(0.12, 1 - 2.1 * abs(dy))
                transverse.append((horizontal_span, line, band))
        if abs(dy) >= 0.52 * abs(dx):
            longitudinal.append((span, line))

    transverse.sort(key=lambda item: item[0], reverse=True)
    longitudinal.sort(key=lambda item: item[0], reverse=True)
    chosen = []
    band_counts = np.zeros(14, dtype=int)
    for _, line, band in transverse:
        if band_counts[band] < 2:
            chosen.append(line)
            band_counts[band] += 1
        if len(chosen) == 19:
            break
    return (
        np.array(chosen, dtype=np.float32).reshape(-1, 3),
        np.array([line for _, line in longitudinal[:14]], dtype=np.float32).reshape(-1, 3),
    )


def _candidate_quads(
    transverse: np.ndarray, longitudinal: np.ndarray, width: int, height: int
) -> np.ndarray:
    if len(transverse) < 2 or len(longitudinal) < 2:
        return np.empty((0, 4, 2), dtype=np.float32)

    grid = np.cross(transverse[:, None, :], longitudinal[None, :, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        xy = grid[:, :, :2] / grid[:, :, 2:]
    rows = np.array(list(combinations(range(len(transverse)), 2)))
    cols = np.array(list(combinations(range(len(longitudinal)), 2)))
    quads = xy[rows[:, None, :, None], cols[None, :, None, :]].reshape(-1, 2, 2, 2)
    # Filter distant intersections before calculating geometry across them.
    bounds = np.array([width, height])
    quads = quads[np.all((quads >= -0.18 * bounds) & (quads <= 1.18 * bounds), axis=(1, 2, 3))]
    reverse_rows = quads[:, 0, :, 1].mean(axis=1) > quads[:, 1, :, 1].mean(axis=1)
    reverse_cols = quads[:, :, 0, 0].mean(axis=1) > quads[:, :, 1, 0].mean(axis=1)
    quads[reverse_rows] = quads[reverse_rows, ::-1, :, :]
    quads[reverse_cols] = quads[reverse_cols, :, ::-1, :]
    quads = quads.reshape(-1, 4, 2)[:, [0, 1, 3, 2]]

    edges = np.roll(quads, -1, axis=1) - quads
    next_edges = np.roll(edges, -1, axis=1)
    turns = edges[:, :, 0] * next_edges[:, :, 1] - edges[:, :, 1] * next_edges[:, :, 0]
    well_spread = (
        (edges[:, 0, 0] > width * 0.07) & (edges[:, 2, 0] < -width * 0.07)
        & (edges[:, 1, 1] > height * 0.07) & (edges[:, 3, 1] < -height * 0.07)
        & (turns > width * height * 0.002).all(axis=1)
    )
    return quads[well_spread]


def _template_seeds() -> Tuple[np.ndarray, np.ndarray]:
    transforms = []
    unused = []
    # A pair from opposite sides of the court gives a well-conditioned fit.
    pairs = ((0, 2), (0, 3), (1, 2), (1, 3))
    for left, right in pairs:
        x, right_x = LONGITUDINAL_X[[left, right]]
        dx = right_x - x
        for top, bottom in pairs:
            y, bottom_y = TRANSVERSE_Y[[top, bottom]]
            dy = bottom_y - y
            transforms.append([[1 / dx, 0, -x / dx], [0, 1 / dy, -y / dy], [0, 0, 1]])
            held_out = np.ones(SEGMENT_COUNT, dtype=np.float32)
            held_out[[left, right, top + 4, bottom + 4]] = 0
            unused.append(held_out)
    return np.asarray(transforms, dtype=np.float32), np.asarray(unused)


SEED_TRANSFORMS, HELD_OUT_LINES = _template_seeds()
COARSE_SAMPLES = sample_segments(1.6)
DENSE_SAMPLES = sample_segments(0.23)


def _fit_candidates(quads: np.ndarray, width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
    square = np.array([(0, 0), (1, 0), (1, 1), (0, 1)], dtype=np.float32)
    square_to_image = np.empty((len(quads), 3, 3), dtype=np.float32)
    for index, quad in enumerate(quads):
        square_to_image[index] = cv2.getPerspectiveTransform(square, quad)

    candidates = (square_to_image[:, None] @ SEED_TRANSFORMS).reshape(-1, 3, 3)
    seed_indices = np.tile(np.arange(len(SEED_TRANSFORMS)), len(quads))
    corners_h = np.column_stack((COURT_CORNERS, np.ones(4, dtype=np.float32)))
    projected = candidates @ corners_h.T
    depth = projected[:, 2, :]
    no_pole = (depth > 1e-4).all(axis=1) | (depth < -1e-4).all(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        xy = projected[:, :2, :] / depth[:, None, :]
    bound = np.array([width, height])[None, :, None]
    near_frame = ((xy >= -0.25 * bound) & (xy <= 1.25 * bound)).all(axis=(1, 2))
    far_width = xy[:, 0, 1] - xy[:, 0, 0]
    near_width = xy[:, 0, 2] - xy[:, 0, 3]
    near_y = xy[:, 1, 2:4].mean(axis=1)
    court_height = near_y - xy[:, 1, :2].mean(axis=1)
    with np.errstate(invalid="ignore"):
        area = 0.5 * np.abs(np.sum(
            xy[:, 0, :] * np.roll(xy[:, 1, :], -1, axis=1)
            - xy[:, 1, :] * np.roll(xy[:, 0, :], -1, axis=1), axis=1,
        ))
    # Thin strips (fences, advertising boards) can contain several bright lines.
    plausible = (
        no_pole & near_frame & (near_y >= 0.51 * height) & (court_height >= 0.23 * height)
        & (far_width >= 0.12 * width) & (near_width >= 0.90 * far_width)
        & (area >= 0.075 * width * height)
    )
    return candidates[plausible], seed_indices[plausible]


def _measure(
    candidates: np.ndarray,
    seed_indices: np.ndarray,
    distance: np.ndarray,
    samples: Tuple[np.ndarray, np.ndarray],
    tolerance: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    supports, visibility = score_templates(candidates, distance, *samples, tolerance)
    coverage = supports * visibility
    overall = coverage.mean(axis=1)
    held_out = (coverage * HELD_OUT_LINES[seed_indices]).sum(axis=1) / (SEGMENT_COUNT - 4)
    return overall, held_out, coverage


def _best_fit(
    candidates: np.ndarray, seed_indices: np.ndarray, distance: np.ndarray, unit: float
) -> Optional[Tuple[np.ndarray, float, float]]:
    tolerance = max(1.8, 2.8 * unit)
    coarse_scores = np.empty(len(candidates), dtype=np.float32)
    for start in range(0, len(candidates), 256):
        batch = slice(start, start + 256)
        overall, held_out, _ = _measure(
            candidates[batch], seed_indices[batch], distance, COARSE_SAMPLES, tolerance,
        )
        coarse_scores[batch] = 0.65 * overall + 0.35 * held_out

    shortlist = np.argsort(-coarse_scores)[:80]
    candidates, seed_indices = candidates[shortlist], seed_indices[shortlist]
    overall, held_out, coverage = _measure(
        candidates, seed_indices, distance, DENSE_SAMPLES, tolerance,
    )
    precise, _, _ = _measure(
        candidates, seed_indices, distance, DENSE_SAMPLES, max(0.96, 1.2 * unit),
    )
    verified = (
        (overall >= 0.83) & (held_out >= 0.74)
        & ((coverage[:, :4] >= 0.48).sum(axis=1) >= 3)
        & ((coverage[:, 4:8] >= 0.48).sum(axis=1) >= 3)
        & (coverage[:, 8:10].max(axis=1) >= 0.60)
    )
    if not np.any(verified):
        return None

    rank = 0.44 * overall + 0.34 * held_out + 0.22 * precise
    winner = int(np.argmax(np.where(verified, rank, -np.inf)))
    return candidates[winner], float(overall[winner]), float(held_out[winner])


def _detect_at_width(frame: np.ndarray, width: int) -> Optional[CourtDetection]:
    original_height, original_width = frame.shape[:2]
    height = max(1, round(original_height * width / original_width))
    image = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA) if width < original_width else frame
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    unit = max(width / 960, 0.4)
    mask = _line_mask(gray, unit)
    transverse, longitudinal = _candidate_lines(mask, unit)
    quads = _candidate_quads(transverse, longitudinal, width, height)
    if len(quads) == 0:
        return None
    candidates, seed_indices = _fit_candidates(quads, width, height)
    if len(candidates) == 0:
        return None

    distance = cv2.distanceTransform(cv2.bitwise_not(mask), cv2.DIST_L2, cv2.DIST_MASK_3)
    best = _best_fit(candidates, seed_indices, distance, unit)
    if best is None:
        return None

    preview_to_original = np.diag([original_width / width, original_height / height, 1.0])
    court_to_image = preview_to_original @ best[0].astype(np.float64)
    image_to_court = np.linalg.inv(court_to_image)
    quad = cv2.perspectiveTransform(COURT_CORNERS[None], court_to_image)[0]
    return CourtDetection(image_to_court, court_to_image, quad, best[1], best[2])


def detect_court(frame: np.ndarray, *, max_width: int = 960) -> Optional[CourtDetection]:
    """Return a verified court in original image pixels, or None for insufficient evidence."""
    original_height, original_width = frame.shape[:2]
    if min(original_height, original_width) < 32 or max_width < 320:
        raise ValueError("A valid frame and max_width of at least 320 are required.")
    width = min(original_width, max_width)
    result = _detect_at_width(frame, width)
    if width < 640 or (result is not None and result.score >= 0.92 and result.validation_score >= 0.88):
        return result

    # Downsampling can suppress fine fence or compression detail enough for
    # Hough voting to recover markings that had weaker support at full size.
    alternate = _detect_at_width(frame, round(width * 2 / 3))
    if result is None:
        return alternate
    if alternate is not None and alternate.score + alternate.validation_score > result.score + result.validation_score:
        return alternate
    return result


def _paint_overlay(canvas: np.ndarray, detection: CourtDetection) -> None:
    thickness = max(1, round(canvas.shape[1] / 650))
    projected = cv2.perspectiveTransform(COURT_SEGMENTS, detection.court_to_image)
    for start, end in np.rint(projected).astype(int):
        cv2.line(canvas, tuple(start), tuple(end), (76, 230, 75, 255), thickness, cv2.LINE_AA)
    for point in np.rint(detection.quad).astype(int):
        cv2.circle(canvas, tuple(point), thickness * 2, (30, 160, 255, 255), -1, cv2.LINE_AA)


def draw_overlay(frame: np.ndarray, detection: CourtDetection) -> np.ndarray:
    """Draw all ground markings in green, and the four outer corners in orange."""
    preview = frame.copy()
    _paint_overlay(preview, detection)
    return preview


def transparent_overlay(frame: np.ndarray, detection: CourtDetection) -> np.ndarray:
    """Return the same overlay as a straight-alpha BGRA image for video compositing."""
    layer = np.zeros((*frame.shape[:2], 4), dtype=np.uint8)
    _paint_overlay(layer, detection)
    alpha = layer[:, :, 3]
    edges = (alpha > 0) & (alpha < 255)
    # Antialiasing against transparent black scales both color and alpha.
    layer[edges, :3] = np.clip(np.rint(layer[edges, :3] * (255.0 / alpha[edges, None])), 0, 255)
    return layer
