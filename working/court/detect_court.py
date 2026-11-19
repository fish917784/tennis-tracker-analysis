"""Detect a tennis court in a still image or a fixed-camera video."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

from court_detector import CourtDetection, detect_court, draw_overlay, transparent_overlay
from court_geometry import COURT_CORNERS, COURT_LENGTH, COURT_WIDTH


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
CORNER_TOLERANCE = 0.008  # Fraction of the original frame diagonal.


@dataclass
class Sample:
    index: Optional[int]
    frame: np.ndarray
    detection: CourtDetection


def sample_video(
    path: Path, samples: int, start: Optional[float] = None, end: Optional[float] = None
) -> Iterator[tuple[int, np.ndarray]]:
    video = cv2.VideoCapture(str(path))
    try:
        if not video.isOpened():
            raise ValueError(f"Could not open video: {path}")

        fps = video.get(cv2.CAP_PROP_FPS)
        has_fps = math.isfinite(fps) and fps > 0
        if (start is not None or end is not None) and not has_fps:
            raise ValueError("Video frame rate is unavailable for the selected time range.")
        first = math.ceil(np.nextafter(start * fps, -math.inf)) if start is not None else 0
        stop = math.ceil(np.nextafter(end * fps, -math.inf)) if end is not None else None

        frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
        if math.isfinite(frame_count) and frame_count >= 1:
            stop = int(frame_count) if stop is None else min(stop, int(frame_count))
        if stop is not None:
            if first >= stop:
                raise ValueError("The selected time range contains no video frames.")
            indices = np.linspace(first, stop - 1, min(samples, stop - first), dtype=int)
            if samples == 1:
                indices[0] = first + (stop - first) // 2
            decoded = False
            for index in indices:
                if index == 0 or video.set(cv2.CAP_PROP_POS_FRAMES, int(index)):
                    ok, frame = video.read()
                    if ok:
                        decoded = True
                        yield int(index), frame
            if decoded:
                return
            video.release()
            video = cv2.VideoCapture(str(path))
        else:
            stride = max(1, round(fps)) if has_fps else 30
            indices = [first + offset * stride for offset in range(samples)]

        # Read from the beginning if the backend lacks seek metadata or cannot seek.
        previous_index = -1
        for index in indices:
            index = int(index)
            for _ in range(index - previous_index - 1):
                if not video.grab():
                    return
            ok, frame = video.read()
            if not ok:
                return
            yield index, frame
            previous_index = index
    finally:
        video.release()


def select_sample(matches: list[Sample]) -> tuple[Sample, int]:
    ranked = []
    for match in matches:
        height, width = match.frame.shape[:2]
        tolerance = CORNER_TOLERANCE * math.hypot(width, height)
        agreement = sum(
            1
            for other in matches
            if match.frame.shape[:2] == other.frame.shape[:2]
            and np.max(np.linalg.norm(match.detection.quad - other.detection.quad, axis=1)) <= tolerance
        )
        ranked.append((agreement, match))

    agreement, winner = max(
        ranked,
        key=lambda item: (item[0], item[1].detection.validation_score, item[1].detection.score),
    )
    return winner, agreement


def process_media(
    path: Path, samples: int, max_width: int, start: Optional[float] = None, end: Optional[float] = None
) -> tuple[Sample, int, int, int]:
    if not path.is_file():
        raise ValueError(f"Input file does not exist: {path}")

    if path.suffix.lower() in IMAGE_SUFFIXES:
        if start is not None or end is not None:
            raise ValueError("The --start and --end options apply to video inputs.")
        frame = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"Could not decode image: {path}")
        detection = detect_court(frame, max_width=max_width)
        if detection is None:
            raise ValueError("No tennis court could be verified in the image.")
        return Sample(None, frame, detection), 1, 1, 1

    matches = []
    decoded = 0
    for index, frame in sample_video(path, samples, start, end):
        decoded += 1
        detection = detect_court(frame, max_width=max_width)
        if detection is not None:
            matches.append(Sample(index, frame, detection))
    if decoded == 0:
        raise ValueError(f"Could not decode a frame from video: {path}")
    if not matches:
        raise ValueError(f"No tennis court could be verified in {decoded} sampled video frames.")
    winner, agreement = select_sample(matches)
    if agreement <= decoded // 2:
        raise ValueError(
            f"Court position agreed in only {agreement} of {decoded} sampled frames. "
            "Choose a fixed-camera interval with --start and --end."
        )
    return winner, decoded, len(matches), agreement


def render_video(
    source: Path, destination: Path, winner: Sample, start: Optional[float], end: Optional[float]
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise ValueError("Video export requires FFmpeg with libx264 on PATH.")

    layer_path = destination.with_name("overlay_layer.png")
    ok, layer = cv2.imencode(".png", transparent_overlay(winner.frame, winner.detection))
    if not ok:
        raise ValueError("Could not encode the video overlay layer.")
    layer_path.write_bytes(layer.tobytes())

    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin"]
    if start is not None:
        command.extend(["-ss", f"{start:.9f}"])
    if end is not None:
        command.extend(["-t", f"{end - (start or 0):.9f}"])
    command.extend([
        "-i", str(source.resolve()), "-i", str(layer_path),
        "-filter_complex", "[0:v:0][1:v:0]overlay=repeatlast=1,pad=ceil(iw/2)*2:ceil(ih/2)*2[v]",
        "-map", "[v]", "-map", "0:a?", "-fps_mode", "passthrough",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(destination),
    ])
    result = subprocess.run(command, capture_output=True, encoding="utf-8", errors="replace")
    if result.returncode or not destination.is_file():
        details = result.stderr.strip() or f"FFmpeg exited with code {result.returncode}."
        raise ValueError(f"Could not export video: {details}")


def save_result(
    path: Path, output: Path, winner: Sample, sampled: int, detected: int, agreeing: int,
    *, overlay_video: bool = False, start: Optional[float] = None, end: Optional[float] = None,
) -> None:
    filenames = ["court_overlay.png", "court.json"]
    if overlay_video:
        filenames.append("court_overlay.mp4")
    if path.resolve() in {(output / name).resolve() for name in filenames}:
        raise ValueError("The output would overwrite the input file. Choose another output directory.")

    result = winner.detection
    height, width = winner.frame.shape[:2]
    metadata = {
        "source": str(path.resolve()),
        "reference_frame_index": winner.index,
        "image_size_px": {"width": width, "height": height},
        "court_size_m": {"width": COURT_WIDTH, "length": COURT_LENGTH},
        "court_coordinates": {
            "units": "meters",
            "origin": "doubles sideline and baseline intersection",
            "x_axis": "across the court",
            "y_axis": "along the length of the court",
        },
        "image_to_court": result.image_to_court.tolist(),
        "court_to_image": result.court_to_image.tolist(),
        "corners_image_px": result.quad.tolist(),
        "corners_court_m": np.asarray(COURT_CORNERS).tolist(),
        "score": float(result.score),
        "validation_score": float(result.validation_score),
        "sampled_frames": sampled,
        "detected_frames": detected,
        "agreeing_frames": agreeing,
    }
    ok, png = cv2.imencode(".png", draw_overlay(winner.frame, result))
    if not ok:
        raise ValueError("Could not encode court preview.")
    report = json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".court-", dir=output.resolve()) as staging_dir:
        staging = Path(staging_dir)
        if overlay_video:
            render_video(path, staging / "court_overlay.mp4", winner, start, end)
        (staging / "court_overlay.png").write_bytes(png.tobytes())
        (staging / "court.json").write_text(report, encoding="utf-8")
        for name in filenames:
            (staging / name).replace(output / name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Still image or fixed-camera video")
    parser.add_argument("--output", type=Path, default=Path("court_output"), help="Output directory")
    parser.add_argument("--samples", type=int, default=5, help="Video sample count (default: 5)")
    parser.add_argument("--max-width", type=int, default=960, help="Detection width limit in pixels (default: 960)")
    parser.add_argument("--start", type=float, help="Video sampling start in seconds (inclusive)")
    parser.add_argument("--end", type=float, help="Video sampling end in seconds (exclusive)")
    parser.add_argument("--overlay-video", action="store_true", help="Also export court_overlay.mp4 (requires FFmpeg)")
    args = parser.parse_args()
    if not 1 <= args.samples <= 12:
        parser.error("--samples must be between 1 and 12")
    if args.max_width < 320:
        parser.error("--max-width must be at least 320")
    if args.start is not None and (not math.isfinite(args.start) or args.start < 0):
        parser.error("--start must be a finite nonnegative number")
    if args.end is not None and (not math.isfinite(args.end) or args.end <= (args.start or 0)):
        parser.error("--end must be finite and greater than the start time")
    if args.overlay_video and args.input.suffix.lower() in IMAGE_SUFFIXES:
        parser.error("--overlay-video applies to video inputs")

    try:
        result = process_media(args.input, args.samples, args.max_width, args.start, args.end)
        save_result(args.input, args.output, *result, overlay_video=args.overlay_video, start=args.start, end=args.end)
    except (OSError, OverflowError, ValueError, cv2.error) as error:
        print(f"Court detection failed: {error}", file=sys.stderr)
        return 1

    _, sampled, _, agreeing = result
    print(f"Saved court.json and court_overlay.png to {args.output.resolve()}")
    if args.overlay_video:
        print(f"Saved overlay video to {(args.output / 'court_overlay.mp4').resolve()}")
    if sampled > 1:
        print(f"Court position agreed in {agreeing} of {sampled} sampled frames.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
