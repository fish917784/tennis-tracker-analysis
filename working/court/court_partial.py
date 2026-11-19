"""Fit a cropped court from the visible markings at the near end."""

from itertools import combinations

import cv2
import numpy as np

if __package__:
    from .court_geometry import (
        COURT_CORNERS, COURT_SEGMENTS, LONGITUDINAL_X, NET_Y, TRANSVERSE_Y,
        sample_segments, score_templates_multi,
    )
else:
    from court_geometry import (
        COURT_CORNERS, COURT_SEGMENTS, LONGITUDINAL_X, NET_Y, TRANSVERSE_Y,
        sample_segments, score_templates_multi,
    )


def _near_samples(spacing):
    points, ids = sample_segments(spacing)
    # Give the uncompressed half of each sideline its own vote. Otherwise
    # isolated lines behind the net can dominate the scores in a low view.
    keep = (ids >= 4) | (points[:, 1] >= NET_Y)
    return points[keep], ids[keep]


_COARSE = _near_samples(.77)
_DENSE = _near_samples(.28)
_NEAR_SLOTS = [0, 1, 2, 3, 6, 7]
_LINE_PAIRS = ((0, 2), (0, 3), (1, 2), (1, 3))


def partial_scores(matrices, distance, *, established=False, coarse=False):
    """Check six nearby markings and corroborate their positions past the seeds."""
    unit = distance.shape[1] / 608
    supports, visibility = score_templates_multi(
        matrices, distance, *(_COARSE if coarse else _DENSE),
        (max(1.25, 2.1 * unit), max(.92, 1.0 * unit)),
    )
    near, close = supports
    # A fraction is only meaningful if enough of that marking is on screen.
    visible = visibility[:, _NEAR_SLOTS] >= .25
    near_lines = near[:, _NEAR_SLOTS] * visible
    close_lines = close[:, _NEAR_SLOTS] * visible
    support = near_lines.mean(axis=1)
    precision = close_lines.mean(axis=1)
    extra = (near[:, [4, 5, 8, 9]] * visibility[:, [4, 5, 8, 9]])
    corroboration = np.sort(extra, axis=1)[:, -2:].mean(axis=1)
    sidelines = near_lines[:, :4]
    witness = .45 if established else .58
    with_baseline = (
        ((sidelines >= witness).sum(axis=1) >= 3)
        & (sidelines[:, :2].max(axis=1) >= .68)
        & (sidelines[:, 2:].max(axis=1) >= .68)
        & (near_lines[:, 4] >= (.64 if established else .75))
        & (near_lines[:, 5] >= (.69 if established else .80))
        & (support >= (.77 if established else .82))
        & (precision >= (.53 if established else .62))
        & (corroboration >= (.36 if established else .50))
    )
    # With the bottom line offscreen, require stronger evidence on both
    # singles sidelines plus independent support across both service boxes.
    baseline_endpoints = matrices @ np.array(
        [[*COURT_CORNERS[3], 1.], [*COURT_CORNERS[2], 1.]]
    ).T
    with np.errstate(divide="ignore", invalid="ignore"):
        baseline_y = (baseline_endpoints[:, 1] / baseline_endpoints[:, 2]).mean(axis=1)
    cropped = ((baseline_y > .978 * distance.shape[0]) & (visibility[:, 7] < .44)
               & (near_lines[:, 5] < .64))
    cropped_support = near_lines[:, :5].mean(axis=1)
    cropped_precision = close_lines[:, :5].mean(axis=1)
    far_support = extra[:, :2].max(axis=1)
    centers = extra[:, 2:]
    center_support = centers.max(axis=1)
    far_confirmed = (far_support >= (.46 if established else .60))
    compressed = ((far_support >= (.17 if established else .22))
                  & (centers.mean(axis=1) >= (.41 if established else .55))
                  & (centers.min(axis=1) >= (.25 if established else .39)))
    without_baseline = (
        cropped & (sidelines[:, 1:3].min(axis=1) >= (.67 if established else .80))
        & (sidelines[:, [0, 3]].min(axis=1) >= (.43 if established else .66))
        & (near_lines[:, 4] >= (.64 if established else .74))
        & (cropped_support >= (.79 if established else .85))
        & (cropped_precision >= (.54 if established else .62))
        & (far_confirmed | compressed)
        & (center_support >= (.43 if established else .62))
    )
    support = np.where(cropped, cropped_support, support)
    precision = np.where(cropped, cropped_precision, precision)
    close_centers = close[:, 8:] * visibility[:, 8:]
    center_fit = .4 * center_support + .3 * centers.mean(axis=1) + .3 * close_centers.mean(axis=1)
    rank = .47 * support + .24 * precision + .12 * corroboration + .13 * center_fit + .04 * extra[:, 0]
    return support, precision, rank, with_baseline | without_baseline


def refine_partial_court(matrix, distance, *, established=False):
    """Align a confirmed projection to nearby markings in the current frame."""
    offsets = [(0, 0)] + [(x, y) for y in range(-3, 4) for x in range(-3, 4) if x or y]
    shifts = np.repeat(np.eye(3)[None], len(offsets), axis=0)
    shifts[:, :2, 2] = offsets
    candidates = shifts @ matrix
    scores, close, rank, valid = partial_scores(candidates, distance, established=established)
    if not np.any(valid):
        return None
    penalty = np.square(offsets).sum(axis=1) * .00035
    best = int(np.argmax(np.where(valid, rank - penalty, -np.inf)))
    return candidates[best], float(scores[best]), float(close[best])


def _plausible(matrices, shape, *, cropped_baseline=False, oblique=False):
    height, width = shape
    points = matrices @ np.column_stack((COURT_CORNERS, np.ones(4))).T
    depth = points[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        xy = points[:, :2] / depth[:, None]
    far = xy[:, 0, 1] - xy[:, 0, 0]
    near = xy[:, 0, 2] - xy[:, 0, 3]
    near_y = xy[:, 1, 2:].mean(axis=1)
    far_rise = np.abs(xy[:, 1, 1] - xy[:, 1, 0])
    near_rise = np.abs(xy[:, 1, 2] - xy[:, 1, 3])
    near_bottom = ((near_y >= height * .975) & (near_y <= height * 1.22) if cropped_baseline
                   else (near_y >= height * .53) & (near_y <= height * 1.08))
    far_slope, near_slope = (.28, .28) if oblique else (.13, .14)
    low_and_cropped = (
        (far >= width * .13) & (far <= near * .60)
        & (near >= width * .62) & (near <= width * 2.6)
        & (far_rise <= far_slope * far) & (near_rise <= near_slope * near)
        & near_bottom
        & (near_y - xy[:, 1, :2].mean(axis=1) >= height * .16)
    )
    if cropped_baseline:
        low_and_cropped &= xy[:, 1, 2:].min(axis=1) >= .90 * height
    if oblique:
        side_depth = xy[:, 1, [3, 2]] - xy[:, 1, :2]
        tilted = (far_rise > .13 * far) | (near_rise > .14 * near)
        low_and_cropped &= tilted & (side_depth >= .10 * height).all(axis=1)
    bounded = ((xy[:, 0] >= -.85 * width) & (xy[:, 0] <= 1.85 * width)).all(axis=1)
    return ((depth > 0).all(axis=1) | (depth < 0).all(axis=1)) & bounded & low_and_cropped


def _seed_transforms(top, bottom):
    y_top, y_bottom = TRANSVERSE_Y[[top, bottom]]
    dy = y_bottom - y_top
    transforms = []
    for left, right in _LINE_PAIRS:
        x, x_right = LONGITUDINAL_X[[left, right]]
        dx = x_right - x
        transforms.append([[1 / dx, 0, -x / dx], [0, 1 / dy, -y_top / dy], [0, 0, 1]])
    return np.asarray(transforms)


_BASELINE_SEEDS = _seed_transforms(2, 3)
_SERVICE_SEEDS = np.concatenate((_seed_transforms(0, 2), _seed_transforms(1, 2)))
_SQUARE = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])


def _ground_segments(distance, rows_mask):
    """Find line segments once for both low-view searches."""
    height, width = distance.shape
    mask = (distance == 0).astype(np.uint8) * 255
    mask[:round(.43 * height)] = 0
    lines = cv2.HoughLinesP(mask, 1, np.pi / 540, max(16, round(width * .032)),
                           minLineLength=max(23, round(width * .054)),
                           maxLineGap=max(5, round(width * .024)))
    weak_rows = cv2.HoughLinesP(rows_mask, 1, np.pi / 540, max(13, round(width * .028)),
                               minLineLength=max(18, round(width * .039)),
                               maxLineGap=max(5, round(width * .024)))
    return lines, weak_rows


def _ground_lines(lines, weak_rows, shape, *, oblique=False):
    """Classify transverse markings and the two sides of the court."""
    height, width = shape
    groups = ([], [], [])
    if lines is None:
        return groups
    segments = lines.reshape(-1, 2, 2).astype(np.float32)
    row_slope = .28 if oblique else .13
    side_min_slope = .28 if oblique else .16
    if weak_rows is not None:
        weak = weak_rows.reshape(-1, 2, 2)
        delta = weak[:, 1] - weak[:, 0]
        weak = weak[np.abs(delta[:, 1]) <= row_slope * np.abs(delta[:, 0])]
        segments = np.concatenate((segments, weak))
    lengths = np.linalg.norm(np.diff(segments, axis=1)[:, 0], axis=1)
    for i in np.argsort(-lengths)[:250]:
        start, end = segments[i]
        start, end = (start, end) if start[0] <= end[0] else (end, start)
        dx, dy = end - start
        row = abs(dy) <= row_slope * dx
        side = abs(dy) > side_min_slope * dx and (oblique or abs(dy) <= 2.5 * dx)
        normal = np.array([-dy, dx], dtype=np.float64) / lengths[i]
        midpoint = (start + end) * .5
        if row and midpoint[1] >= height * .51:
            group = groups[0]
        elif side and midpoint[1] >= height * .54:
            group = groups[1 if dy < 0 else 2]
        else:
            continue
        for other, pos, points in group:
            if normal @ other > .9992 and abs((midpoint - pos) @ other) < 2.4:
                points.extend((start, end))
                break
        else:
            group.append((normal, midpoint, [start, end]))

    fitted = []
    for index, group in enumerate(groups):
        lines_by_span = []
        for _, _, points in group:
            pts = np.asarray(points)
            dx, dy, x, y = cv2.fitLine(pts, cv2.DIST_L2, 0, .01, .01).ravel()
            line = np.array([-dy, dx, dy * x - dx * y])
            lines_by_span.append((np.ptp(pts @ np.array([dx, dy])), line))
        lines_by_span.sort(key=lambda item: item[0], reverse=True)
        if index:
            fitted.append(np.asarray([line for _, line in lines_by_span[:11]]))
            continue
        rows = []
        bands = np.zeros(18, dtype=int)
        for _, line in lines_by_span:
            y = -(line[0] * width / 2 + line[2]) / line[1]
            band = int(18 * y / height)
            if 8 <= band < len(bands) and bands[band] < 3:
                rows.append(line)
                bands[band] += 1
        fitted.append(np.asarray(rows))
    return fitted


def _section_quads(transverse, left, right, shape, *, cropped_baseline=False):
    if min(len(transverse), len(left), len(right)) == 0 or len(transverse) < 2:
        return np.empty((0, 4, 2), np.float32)
    height, width = shape
    mid_y = -(transverse[:, 0] * width / 2 + transverse[:, 2]) / transverse[:, 1]
    order = np.argsort(mid_y)
    transverse, mid_y = transverse[order], mid_y[order]
    upper, lower = np.array(list(combinations(range(len(transverse)), 2))).T
    spans = (mid_y[lower] - mid_y[upper])
    keep = (spans >= .075 * height) & (spans <= .45 * height) & (mid_y[lower] >= .56 * height)
    if cropped_baseline:
        keep &= mid_y[upper] < .83 * height
    upper, lower = upper[keep], lower[keep]
    left_grid = np.cross(transverse[:, None], left[None])
    right_grid = np.cross(transverse[:, None], right[None])
    with np.errstate(divide="ignore", invalid="ignore"):
        left_xy = left_grid[..., :2] / left_grid[..., 2:]
        right_xy = right_grid[..., :2] / right_grid[..., 2:]
    a, d = np.broadcast_arrays(left_xy[upper, :, None], left_xy[lower, :, None])
    b, c = np.broadcast_arrays(right_xy[upper, None, :], right_xy[lower, None, :])
    a, b = np.broadcast_arrays(a, b)
    c, d = np.broadcast_arrays(c, d)
    quads = np.stack((a, b, c, d), axis=-2).reshape(-1, 4, 2)
    span_top = quads[:, 1, 0] - quads[:, 0, 0]
    span_bottom = quads[:, 2, 0] - quads[:, 3, 0]
    keep = (
        (span_top >= .23 * width) & (span_bottom >= span_top * 1.20)
        & (span_bottom <= span_top * 2.70)
        & ((quads[:, :, 0] >= -.70 * width) & (quads[:, :, 0] <= 1.70 * width)).all(axis=1)
    )
    return quads[keep].astype(np.float32)


def _unexplained_cross_line(matrix, rows, row_distance):
    """Reject a service-line candidate if another ground marking crosses below it."""
    height, width = row_distance.shape
    projected = cv2.perspectiveTransform(COURT_SEGMENTS[[1, 2, 6, 7]], matrix)
    points = np.concatenate((projected, np.ones((4, 2, 1))), axis=2)
    left, right, service, baseline = np.cross(points[:, 0], points[:, 1])
    y_at_middle = lambda edges: -(edges[..., 0] * width / 2 + edges[..., 2]) / edges[..., 1]
    row_y = y_at_middle(rows)
    crossing = rows[(row_y > y_at_middle(service) + .038 * height)
                    & (row_y < y_at_middle(baseline) - .030 * height)]
    if not len(crossing):
        return False
    intersections = np.cross(crossing[:, None], np.stack((left, right))[None])
    with np.errstate(divide="ignore", invalid="ignore"):
        endpoints = intersections[:, :, :2] / intersections[:, :, 2:]
    endpoints = endpoints[np.isfinite(endpoints).all(axis=(1, 2))
                          & (endpoints[:, 1, 0] - endpoints[:, 0, 0] >= .35 * width)]
    if not len(endpoints):
        return False
    samples = endpoints[:, :1] + (endpoints[:, 1:] - endpoints[:, :1]) * np.linspace(.06, .94, 50)[None, :, None]
    x, y = np.rint(samples).astype(np.intp).transpose(2, 0, 1)
    inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    matched = np.zeros_like(inside)
    matched[inside] = row_distance[y[inside], x[inside]] <= max(1.0, 1.42 * width / 608)
    count = inside.sum(axis=1)
    return bool(np.any((count >= 25) & (matched.sum(axis=1) > .60 * count)))


def detect_partial_court(gray, distance):
    """Seed from the near baseline/service line and score independent markings."""
    height, width = gray.shape
    kernel = np.ones((max(5, round(width * .013) | 1), 1), np.uint8)
    ridges = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    rows_mask = cv2.inRange(ridges, 8, 255)
    rows_mask[:round(.50 * height)] = 0
    segments, weak_rows = _ground_segments(distance, rows_mask)
    # Keep the usual near-level search first. A side view may have a much
    # steeper baseline and an almost vertical visible sideline.
    for oblique in (False, True):
        rows, left, right = _ground_lines(segments, weak_rows, distance.shape, oblique=oblique)
        for cropped in (False, True):
            fit = _fit_marking_candidates(
                rows, left, right, distance, rows_mask, cropped_baseline=cropped, oblique=oblique,
            )
            if fit is not None:
                return fit
    return None


def _fit_marking_candidates(rows, left, right, distance, rows_mask, *, cropped_baseline=False, oblique=False):
    quads = _section_quads(rows, left, right, distance.shape, cropped_baseline=cropped_baseline)
    if not len(quads):
        return None
    bases = np.asarray([cv2.getPerspectiveTransform(_SQUARE, quad) for quad in quads])
    seeds = _SERVICE_SEEDS if cropped_baseline else _BASELINE_SEEDS
    candidates = (bases[:, None] @ seeds).reshape(-1, 3, 3)
    candidates = candidates[_plausible(
        candidates, distance.shape, cropped_baseline=cropped_baseline, oblique=oblique,
    )]
    if not len(candidates):
        return None
    ranks = []
    for start in range(0, len(candidates), 192):
        _, _, rank, valid = partial_scores(candidates[start:start + 192], distance, coarse=True)
        ranks.append(rank + valid * .025)
    shortlist = candidates[np.argsort(-np.concatenate(ranks))[:36]]
    scores, close, rank, valid = partial_scores(shortlist, distance)
    if np.any(valid):
        row_distance = cv2.distanceTransform(cv2.bitwise_not(rows_mask), cv2.DIST_L2, cv2.DIST_MASK_3)
        for best in np.argsort(-rank):
            if valid[best] and not _unexplained_cross_line(shortlist[best], rows, row_distance):
                return shortlist[best], float(scores[best]), float(close[best])
    return None
