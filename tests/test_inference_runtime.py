from contextlib import nullcontext
import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "runtime_under_test", ROOT / "working/desktop/inference_runtime.py"
)
runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime)


class FrameRedChannel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, value):
        self.inputs.append(value.detach().clone())
        return value[:, [0, 3, 6]]


class NonFiniteOnce(FrameRedChannel):
    def forward(self, value):
        output = super().forward(value)
        return torch.full_like(output, float("nan")) if len(self.inputs) == 1 else output


class AlwaysNonFinite(FrameRedChannel):
    def forward(self, value):
        return torch.full_like(super().forward(value), float("inf"))


def frame(red, green=20, blue=10):
    image = np.empty((8, 12, 3), dtype=np.uint8)
    image[:] = [blue, green, red]
    return image


class InferenceRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.precision = patch.dict(os.environ, {"TENNIS_PRECISION": "amp"})
        self.precision.start()
        runtime._AMP_DISABLED_MODELS.clear()
        runtime._WARNED_AMP_FALLBACK = False

    def tearDown(self):
        self.precision.stop()

    def test_cpu_uses_fp32_and_preserves_rgb_and_temporal_channels(self):
        model = FrameRedChannel()
        with patch.object(runtime.torch, "autocast") as autocast:
            output = runtime.predict_heatmaps(model, [frame(30), frame(60), frame(90)], "cpu")
        autocast.assert_not_called()
        self.assertEqual(output.shape, (3, 288, 512))
        self.assertEqual(output.dtype, np.float32)
        value = model.inputs[0]
        self.assertEqual(value.dtype, torch.float32)
        self.assertTrue(value.is_contiguous(memory_format=torch.channels_last))
        np.testing.assert_allclose(value[0, :, 0, 0].numpy(), np.array([30, 20, 10, 60, 20, 10, 90, 20, 10]) / 255)
        np.testing.assert_allclose(output[:, 0, 0], np.array([30, 60, 90]) / 255)

    def test_one_and_two_frames_repeat_only_for_input_padding(self):
        for count in [1, 2]:
            with self.subTest(count=count):
                model = FrameRedChannel()
                output = runtime.predict_heatmaps(model, [frame(30 + 30 * i) for i in range(count)], "cpu")
                self.assertEqual(output.shape, (count, 288, 512))
                last = model.inputs[0][0, (count - 1) * 3:(count - 1) * 3 + 3]
                for index in range(count, 3):
                    torch.testing.assert_close(model.inputs[0][0, index * 3:index * 3 + 3], last)

    def test_batch_preserves_group_order_lengths_and_single_forward(self):
        model = FrameRedChannel()
        outputs = runtime.predict_triplets(model, [[frame(30), frame(60)], [frame(90)]], "cpu")
        self.assertEqual(len(model.inputs), 1)
        self.assertEqual(model.inputs[0].shape, (2, 9, 288, 512))
        self.assertEqual([value.shape[0] for value in outputs], [2, 1])
        np.testing.assert_allclose(outputs[0][:, 0, 0], np.array([30, 60]) / 255)
        np.testing.assert_allclose(outputs[1][:, 0, 0], np.array([90]) / 255)

    def test_empty_overfull_and_malformed_inputs_fail_clearly(self):
        model = FrameRedChannel()
        for frames in [[], [frame(20)] * 4, [np.zeros((8, 12), np.uint8)], [None]]:
            with self.subTest(kind=str(type(frames[0])) if frames else "empty"):
                with self.assertRaises(ValueError):
                    runtime.predict_heatmaps(model, frames, "cpu")
        with self.assertRaises(ValueError):
            runtime.predict_triplets(model, [], "cpu")
        self.assertEqual(model.inputs, [])

    def test_precision_env_and_explicit_disable(self):
        self.assertFalse(runtime._amp_enabled(torch.device("cpu"), True))
        self.assertTrue(runtime._amp_enabled(torch.device("cuda"), True))
        self.assertFalse(runtime._amp_enabled(torch.device("cuda"), False))
        with patch.dict(os.environ, {"TENNIS_PRECISION": "fp32"}):
            self.assertFalse(runtime._amp_enabled(torch.device("cuda"), True))
        with patch.dict(os.environ, {"TENNIS_PRECISION": "invalid"}):
            with self.assertRaisesRegex(ValueError, "TENNIS_PRECISION"):
                runtime.predict_heatmaps(FrameRedChannel(), [frame(30)], "cpu")

    def test_nonfinite_amp_retries_fp32_disables_instance_and_warns_once(self):
        first, second = NonFiniteOnce(), NonFiniteOnce()
        # Mock only the AMP-selection/context seam. All tensors and model
        # execution stay on CPU, so this test never initializes CUDA.
        with patch.object(runtime, "_amp_enabled", return_value=True), \
             patch.object(runtime.torch, "autocast", return_value=nullcontext()) as autocast, \
             warnings.catch_warnings(record=True) as emitted:
            warnings.simplefilter("always")
            first_output = runtime.predict_heatmaps(first, [frame(30)], "cpu")
            runtime.predict_heatmaps(first, [frame(30)], "cpu")
            second_output = runtime.predict_heatmaps(second, [frame(60)], "cpu")
        self.assertTrue(np.isfinite(first_output).all())
        self.assertTrue(np.isfinite(second_output).all())
        self.assertEqual(len(first.inputs), 3)
        self.assertEqual(len(second.inputs), 2)
        self.assertEqual(autocast.call_count, 2)
        self.assertIn(first, runtime._AMP_DISABLED_MODELS)
        self.assertEqual(len(emitted), 1)
        self.assertIn("FP32", str(emitted[0].message))

    def test_nonfinite_fp32_and_bad_model_shape_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-finite"):
            runtime.predict_heatmaps(AlwaysNonFinite(), [frame(30)], "cpu", use_amp=False)
        class BadShape(torch.nn.Module):
            def forward(self, value):
                return value[:, :1]
        with self.assertRaisesRegex(ValueError, "shape"):
            runtime.predict_heatmaps(BadShape(), [frame(30)], "cpu")


if __name__ == "__main__":
    unittest.main()
