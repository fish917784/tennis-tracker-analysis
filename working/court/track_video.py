"""Render court lines over an entire video, reacquiring the court after cuts."""

import argparse
from fractions import Fraction
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Tuple

import cv2
import numpy as np

from court_detector import draw_overlay
from court_tracking import CourtTracker


def write_frames(
    video: cv2.VideoCapture, source: Path, destination: Path, ffmpeg: str
) -> Tuple[int, int, CourtTracker]:
    fps = video.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Could not read the video frame rate.")
    ok, frame = video.read()
    if not ok:
        raise ValueError("Could not decode the first video frame.")
    height, width = frame.shape[:2]
    expected = video.get(cv2.CAP_PROP_FRAME_COUNT)
    total = str(round(expected)) if math.isfinite(expected) and expected > 0 else "?"
    tracker = CourtTracker(fps)
    frame_count = 0
    tracked_count = 0
    report_every = max(1, round(fps * 20))
    rate = str(Fraction(fps).limit_denominator(100_000))

    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
        "-f", "rawvideo", "-pixel_format", "bgr24", "-video_size", f"{width}x{height}",
        "-framerate", rate, "-i", "pipe:0", "-i", str(source),
        "-map", "0:v:0", "-map", "1:a?", "-c:a", "copy",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(destination),
    ]
    with tempfile.TemporaryFile() as diagnostics:
        encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=diagnostics)
        assert encoder.stdin is not None
        try:
            while ok:
                if frame.shape[:2] != (height, width):
                    raise ValueError(f"Video dimensions changed at frame {frame_count}.")
                detection = tracker.update(frame)
                if detection is not None:
                    frame = draw_overlay(frame, detection)
                    tracked_count += 1
                encoder.stdin.write(memoryview(np.ascontiguousarray(frame)).cast("B"))
                frame_count += 1
                if frame_count % report_every == 0:
                    print(f"Frames {frame_count}/{total} | court lines {tracked_count}", flush=True)
                ok, frame = video.read()
            encoder.stdin.close()
        except BrokenPipeError:
            pass  # Read the encoder diagnostics below.
        finally:
            try:
                if not encoder.stdin.closed:
                    encoder.stdin.close()
            except BrokenPipeError:
                pass
            if sys.exc_info()[0] is not None and encoder.poll() is None:
                encoder.terminate()
            encoder.wait()

        if encoder.returncode or not destination.is_file():
            diagnostics.seek(0)
            details = diagnostics.read().decode("utf-8", errors="replace").strip()
            raise ValueError(f"FFmpeg could not export the video: {details or encoder.returncode}")
    if tracked_count == 0:
        raise ValueError("No court markings could be verified anywhere in the video.")
    if math.isfinite(expected) and expected > 0 and frame_count < round(expected):
        raise ValueError(f"Video decoding stopped at {frame_count} of {round(expected)} frames.")
    return frame_count, tracked_count, tracker


def export_video(source: Path, output: Path) -> Tuple[int, int, CourtTracker]:
    if not source.is_file():
        raise ValueError(f"Input file does not exist: {source}")
    source = source.resolve()
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise ValueError("Video export requires FFmpeg with libx264 on PATH.")
    output = output.resolve()
    destination = output / "court_overlay.mp4"
    if destination.resolve() == source:
        raise ValueError("Choose an output directory that keeps the input video separate.")
    video = cv2.VideoCapture(str(source))
    try:
        if not video.isOpened():
            raise ValueError(f"Could not open video: {source}")
        output.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".court-video-", dir=output) as temporary:
            staged = Path(temporary) / destination.name
            result = write_frames(video, source, staged, ffmpeg)
            staged.replace(destination)
        return result
    finally:
        video.release()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input match video")
    parser.add_argument("--output", type=Path, default=Path("court_output"), help="Output directory")
    args = parser.parse_args()
    started = time.perf_counter()
    try:
        frames, tracked, tracker = export_video(args.input, args.output)
    except (OSError, ValueError, cv2.error) as error:
        print(f"Court video export failed: {error}", file=sys.stderr)
        return 1
    elapsed = time.perf_counter() - started
    print(f"Saved {(args.output / 'court_overlay.mp4').resolve()}")
    print(f"Exported {frames} frames in {elapsed:.2f} s. Court lines on {tracked} frames.")
    print(f"Full court detector calls: {tracker.acquisitions}. Scene resets: {tracker.cuts}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
