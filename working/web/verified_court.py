"""Use the reviewed per-frame court only for byte-identical source videos."""
import hashlib
import json
from pathlib import Path
from zipfile import BadZipFile

import numpy as np


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


class VerifiedCourtGeometry:
    def __init__(self, result, archive):
        self.fps = float(result["fps"])
        self.width = int(result["width"])
        self.height = int(result["height"])
        self.count = int(result["frame_count"])
        self.to_image = archive["court_to_image"]
        self.quad = archive["court_quad"]
        self.segment = archive["segment_id"]
        if (self.count < 1 or not np.isfinite(self.fps) or self.fps <= 0
                or min(self.width, self.height) < 32
                or self.to_image.shape != (self.count, 3, 3)
                or self.quad.shape != (self.count, 4, 2)
                or self.segment.shape != (self.count,)
                or not np.issubdtype(self.segment.dtype, np.integer)
                or np.any(self.segment[1:] < self.segment[:-1])):
            raise ValueError("Invalid verified court frame data")
        self.calibrated = np.isfinite(self.to_image).all(axis=(1, 2))
        if (not self.calibrated.any()
                or not np.array_equal(self.calibrated, np.isfinite(self.quad).all(axis=(1, 2)))
                or int(self.calibrated.sum()) != result["calibrated_frames"]):
            raise ValueError("Verified court matrices and outlines do not agree")
        self.to_court = np.full(self.to_image.shape, np.nan, np.float64)
        inv = np.linalg.inv(self.to_image[self.calibrated])
        if not np.isfinite(inv).all():
            raise ValueError("Invalid verified court transform")
        self.to_court[self.calibrated] = inv

        self.displayed = archive["displayed"] if "displayed" in archive else self.calibrated
        if (self.displayed.shape != (self.count,) or self.displayed.dtype != np.bool_
                or np.any(self.calibrated & ~self.displayed)):
            raise ValueError("Invalid verified court display data")
        self.displayed_from = np.maximum.accumulate(np.where(self.calibrated, np.arange(self.count), -1))
        if np.any(self.displayed & (self.displayed_from < 0)):
            raise ValueError("No preceding court outline for verified display")
        # The archived analysis segment is always authoritative for ball
        # tracking. Display history can remain during its archived grace.
        cut_between_verified = (self.calibrated[:-1] & self.calibrated[1:]
                                & (self.segment[:-1] != self.segment[1:]))
        self.display_segment = np.r_[0, np.cumsum(cut_between_verified | (self.displayed[1:] != self.displayed[:-1]))]

    def matches(self, fps, width, height, total):
        return (self.width == width and self.height == height and self.count == total
                and bool(np.isclose(self.fps, fps, rtol=1e-6, atol=1e-6)))

    def update(self, frame, frame_idx):
        calibrated = bool(self.calibrated[frame_idx])
        displayed = bool(self.displayed[frame_idx])
        quad = self.quad[frame_idx] if calibrated else None
        display_quad = self.quad[self.displayed_from[frame_idx]] if displayed else None
        return {
            "H": self.to_court[frame_idx] if calibrated else None,
            "H_inv": self.to_image[frame_idx] if calibrated else None,
            "court_poly": quad.astype(np.int32) if calibrated else None,
            "corners": quad,
            "method": "verified_source" if calibrated else "none",
            "calibrated": calibrated,
            "segment_id": int(self.segment[frame_idx]),
            "displayed": displayed,
            "verifying": displayed and not calibrated,
            "display_segment_id": int(self.display_segment[frame_idx]),
            "display_corners": display_quad,
            "display_court_poly": display_quad.astype(np.int32) if displayed else None,
        }


def load_verified_court(path: Path, evidence_root: Path):
    """Return a matching court record, or fall back to automatic detection."""
    evidence_root = Path(evidence_root).resolve()
    manifest_path = evidence_root / "verified-evidence.json"
    if not manifest_path.is_file():
        return None
    try:
        entries = read_json(manifest_path)
        if not isinstance(entries, dict):
            return None
        with Path(path).open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
    except (OSError, ValueError):
        return None

    for selection in entries.values():
        try:
            directory = (evidence_root / selection["result_dir"]).resolve()
            if not directory.is_relative_to(evidence_root):
                continue
            result = read_json(directory / "result.json")
            if (result["source_sha256"] != digest
                    or result["decoded_all_source_frames"] is not True
                    or result["expected_frames"] != result["frame_count"]):
                continue
            with np.load(directory / "analysis.npz", allow_pickle=False) as archive:
                return VerifiedCourtGeometry(result, archive)
        except (OSError, ValueError, KeyError, TypeError, np.linalg.LinAlgError, BadZipFile):
            continue
    return None
