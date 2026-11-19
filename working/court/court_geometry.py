"""Ground-plane tennis court markings and projected-line support scoring."""

from typing import Sequence, Tuple

import numpy as np


METERS_PER_FOOT = 0.3048
COURT_WIDTH = 36 * METERS_PER_FOOT
COURT_LENGTH = 78 * METERS_PER_FOOT
NET_Y = COURT_LENGTH / 2
SINGLES_INSET = 4.5 * METERS_PER_FOOT
SERVICE_DISTANCE = 21 * METERS_PER_FOOT

COURT_CORNERS = np.array(
    [(0, 0), (COURT_WIDTH, 0), (COURT_WIDTH, COURT_LENGTH), (0, COURT_LENGTH)],
    dtype=np.float32,
)
LONGITUDINAL_X = np.array(
    [0, SINGLES_INSET, COURT_WIDTH - SINGLES_INSET, COURT_WIDTH], dtype=np.float32
)
TRANSVERSE_Y = np.array(
    [0, NET_Y - SERVICE_DISTANCE, NET_Y + SERVICE_DISTANCE, COURT_LENGTH],
    dtype=np.float32,
)

# The service boxes stop at the singles sidelines. The net sits above the
# court plane, so its white tape must not participate in line matching.
COURT_SEGMENTS = np.array(
    [
        [(x, 0), (x, COURT_LENGTH)] for x in LONGITUDINAL_X
    ] + [
        [(0, 0), (COURT_WIDTH, 0)],
        [(SINGLES_INSET, TRANSVERSE_Y[1]), (COURT_WIDTH - SINGLES_INSET, TRANSVERSE_Y[1])],
        [(SINGLES_INSET, TRANSVERSE_Y[2]), (COURT_WIDTH - SINGLES_INSET, TRANSVERSE_Y[2])],
        [(0, COURT_LENGTH), (COURT_WIDTH, COURT_LENGTH)],
        [(COURT_WIDTH / 2, TRANSVERSE_Y[1]), (COURT_WIDTH / 2, NET_Y)],
        [(COURT_WIDTH / 2, NET_Y), (COURT_WIDTH / 2, TRANSVERSE_Y[2])],
    ],
    dtype=np.float32,
)
SEGMENT_COUNT = len(COURT_SEGMENTS)


def sample_segments(spacing: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    """Sample each marking in meters, returning homogeneous points and segment IDs."""
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("spacing must be positive and finite")

    directions = COURT_SEGMENTS[:, 1] - COURT_SEGMENTS[:, 0]
    counts = np.ceil(np.linalg.norm(directions, axis=1) / spacing).astype(np.intp)
    counts = np.maximum(counts, 1)
    segment_ids = np.repeat(np.arange(SEGMENT_COUNT), counts)
    starts = np.cumsum(counts) - counts
    offsets = np.arange(len(segment_ids)) - starts[segment_ids]
    # Midpoints of equal subdivisions keep shared endpoints from being counted twice.
    fractions = ((offsets + 0.5) / counts[segment_ids]).astype(np.float32)
    points = np.ones((len(segment_ids), 3), dtype=np.float32)
    points[:, :2] = COURT_SEGMENTS[segment_ids, 0] + fractions[:, None] * directions[segment_ids]
    return points, segment_ids


def score_templates(
    homographies: np.ndarray,
    distance_image: np.ndarray,
    sample_points: np.ndarray,
    segment_ids: np.ndarray,
    tolerance: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return supported and visible fractions per homography and court marking.

    Homographies project court coordinates into the distance image. Each sample
    has a sorted segment ID, as returned by ``sample_segments``. Support is the
    fraction of a segment's visible samples within ``tolerance`` image pixels
    of a candidate line. Visibility is relative to all samples on the segment.
    """
    height, width = distance_image.shape
    projected = homographies @ sample_points.T
    denominator = projected[:, 2, :]
    valid_denominator = np.isfinite(denominator) & (denominator != 0)
    xy = np.full(projected[:, :2, :].shape, np.nan, dtype=projected.dtype)
    with np.errstate(over="ignore", invalid="ignore"):
        np.divide(projected[:, :2, :], denominator[:, None, :],
                  out=xy, where=valid_denominator[:, None, :])
    x, y = xy[:, 0, :], xy[:, 1, :]
    visible = (x >= 0) & (x <= width - 1) & (y >= 0) & (y <= height - 1)
    pixel_x = np.rint(x[visible]).astype(np.intp)
    pixel_y = np.rint(y[visible]).astype(np.intp)
    supported = np.zeros_like(visible)
    supported[visible] = distance_image[pixel_y, pixel_x] <= tolerance

    edges = np.searchsorted(segment_ids, np.arange(SEGMENT_COUNT + 1))
    support_fractions = np.zeros((len(homographies), SEGMENT_COUNT), dtype=np.float32)
    visible_fractions = np.zeros_like(support_fractions)
    for segment, (start, stop) in enumerate(zip(edges[:-1], edges[1:])):
        if start == stop:
            continue
        visible_count = np.count_nonzero(visible[:, start:stop], axis=1)
        supported_count = np.count_nonzero(supported[:, start:stop], axis=1)
        visible_fractions[:, segment] = visible_count / (stop - start)
        np.divide(supported_count, visible_count, out=support_fractions[:, segment],
                  where=visible_count != 0)
    return support_fractions, visible_fractions


def score_templates_multi(
    homographies: np.ndarray,
    distance_image: np.ndarray,
    sample_points: np.ndarray,
    segment_ids: np.ndarray,
    tolerances: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Score several tolerances with one projection and distance lookup.

    The support result has shape (tolerances, homographies, court markings).
    Visibility is shared and has shape (homographies, court markings). Each
    tolerance uses the same comparison and float32 division as score_templates.
    """
    tolerances = tuple(tolerances)
    height, width = distance_image.shape
    projected = homographies @ sample_points.T
    denominator = projected[:, 2, :]
    valid_denominator = np.isfinite(denominator) & (denominator != 0)
    xy = np.full(projected[:, :2, :].shape, np.nan, dtype=projected.dtype)
    with np.errstate(over="ignore", invalid="ignore"):
        np.divide(projected[:, :2, :], denominator[:, None, :],
                  out=xy, where=valid_denominator[:, None, :])
    x, y = xy[:, 0, :], xy[:, 1, :]
    visible = (x >= 0) & (x <= width - 1) & (y >= 0) & (y <= height - 1)
    pixel_x = np.rint(x[visible]).astype(np.intp)
    pixel_y = np.rint(y[visible]).astype(np.intp)
    sampled_distances = distance_image[pixel_y, pixel_x]
    supported = []
    for tolerance in tolerances:
        support_mask = np.zeros_like(visible)
        # Keep scalar comparison semantics, including float32 boundary cases.
        support_mask[visible] = sampled_distances <= tolerance
        supported.append(support_mask)

    edges = np.searchsorted(segment_ids, np.arange(SEGMENT_COUNT + 1))
    support_fractions = np.zeros(
        (len(tolerances), len(homographies), SEGMENT_COUNT), dtype=np.float32
    )
    visible_fractions = np.zeros((len(homographies), SEGMENT_COUNT), dtype=np.float32)
    for segment, (start, stop) in enumerate(zip(edges[:-1], edges[1:])):
        if start == stop:
            continue
        visible_count = np.count_nonzero(visible[:, start:stop], axis=1)
        visible_fractions[:, segment] = visible_count / (stop - start)
        for index, support_mask in enumerate(supported):
            supported_count = np.count_nonzero(support_mask[:, start:stop], axis=1)
            np.divide(supported_count, visible_count,
                      out=support_fractions[index, :, segment], where=visible_count != 0)
    return support_fractions, visible_fractions
