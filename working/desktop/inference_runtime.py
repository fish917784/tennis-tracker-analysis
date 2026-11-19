"""Shared fixed-size TrackNet preprocessing and inference.

CUDA AMP changes numerical precision. Callers can select the FP32 baseline with
TENNIS_PRECISION=fp32 or use_amp=False; output heatmaps always use float32.
"""
from contextlib import nullcontext
import os
from threading import Lock
import warnings
from weakref import WeakSet

import cv2
import numpy as np
import torch


WIDTH, HEIGHT = 512, 288
_AMP_DISABLED_MODELS = WeakSet()
_FALLBACK_LOCK = Lock()
_WARNED_AMP_FALLBACK = False


def _amp_enabled(device: torch.device, use_amp: bool) -> bool:
    precision = os.environ.get("TENNIS_PRECISION", "amp").strip().lower()
    if precision not in {"amp", "fp32"}:
        raise ValueError("TENNIS_PRECISION must be 'amp' or 'fp32'.")
    return bool(use_amp and precision == "amp" and device.type == "cuda")


def _prepare_triplet(frames, interpolation: int):
    frames = list(frames)
    if not 1 <= len(frames) <= 3:
        raise ValueError("Each prediction requires between one and three BGR frames.")
    images = []
    for frame in frames:
        if (not isinstance(frame, np.ndarray) or frame.ndim != 3
                or frame.shape[2] != 3 or min(frame.shape[:2]) < 1):
            raise ValueError("Each frame must be a non-empty H×W×3 BGR image.")
        images.append(cv2.cvtColor(
            cv2.resize(frame, (WIDTH, HEIGHT), interpolation=interpolation),
            cv2.COLOR_BGR2RGB,
        ))
    count = len(images)
    images.extend([images[-1]] * (3 - count))
    return np.concatenate(images, axis=2), count


def _warn_and_disable_amp(model):
    global _WARNED_AMP_FALLBACK
    with _FALLBACK_LOCK:
        _AMP_DISABLED_MODELS.add(model)
        if not _WARNED_AMP_FALLBACK:
            warnings.warn(
                "Non-finite CUDA AMP heatmaps; retrying in FP32 and disabling "
                "AMP for this model instance.",
                RuntimeWarning,
                stacklevel=3,
            )
            _WARNED_AMP_FALLBACK = True


def _forward_heatmaps(model, tensor: torch.Tensor, device: torch.device, amp: bool) -> np.ndarray:
    context = torch.autocast(device_type="cuda", dtype=torch.float16) if amp else nullcontext()
    with context:
        output = model(tensor)
    expected = (tensor.shape[0], 3, HEIGHT, WIDTH)
    if not isinstance(output, torch.Tensor) or tuple(output.shape) != expected:
        shape = getattr(output, "shape", type(output).__name__)
        raise ValueError(f"TrackNet must return heatmaps with shape {expected}; received {shape}.")
    # The existing CPU candidate extraction needs the full maps. Check after
    # this transfer to avoid adding a second per-call CUDA synchronization.
    return output.float().cpu().numpy()


@torch.inference_mode()
def predict_triplets(model, frame_groups, device, use_amp: bool = True,
                     interpolation: int = cv2.INTER_AREA) -> list[np.ndarray]:
    """Infer a batch of triplets while retaining each group's original length.

    The returned list preserves input order; each array has shape (N,288,512),
    where N is that group's original frame count (1–3). Batch size is chosen by
    the caller so online camera latency need not increase.
    """
    device = torch.device(device)
    groups = list(frame_groups)
    if not groups:
        raise ValueError("At least one frame group is required.")
    prepared = [_prepare_triplet(frames, interpolation) for frames in groups]
    images = [item[0] for item in prepared]
    pixels = images[0][None] if len(images) == 1 else np.stack(images, axis=0)
    tensor = torch.from_numpy(pixels).permute(0, 3, 1, 2)
    tensor = tensor.to(device, dtype=torch.float32).div_(255)
    tensor = tensor.contiguous(memory_format=torch.channels_last)
    amp = _amp_enabled(device, use_amp) and model not in _AMP_DISABLED_MODELS
    output = _forward_heatmaps(model, tensor, device, amp)
    if not np.isfinite(output).all():
        if amp:
            _warn_and_disable_amp(model)
            output = _forward_heatmaps(model, tensor, device, False)
        if not np.isfinite(output).all():
            raise ValueError("TrackNet returned non-finite heatmaps in FP32.")
    return [output[index, :count] for index, (_, count) in enumerate(prepared)]


@torch.inference_mode()
def predict_heatmaps(model, frames, device, use_amp: bool = True,
                     interpolation: int = cv2.INTER_AREA) -> np.ndarray:
    """Infer one to three frames, repeating the final input only for padding."""
    return predict_triplets(model, [frames], device, use_amp=use_amp,
                            interpolation=interpolation)[0]
