import asyncio
import hashlib
import importlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = types.ModuleType("verified_web_tests")
PACKAGE.__path__ = [str(ROOT / "working/web")]
sys.modules[PACKAGE.__name__] = PACKAGE


class VerifiedCourtTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.evidence = self.root / "evidence"
        self.result_dir = self.evidence / "recovered" / "test"
        self.result_dir.mkdir(parents=True)
        self.path = self.root / "uploaded_by_uuid.mp4"
        self.source_bytes = b"exact original video contents"
        self.path.write_bytes(self.source_bytes)
        self.load = importlib.import_module("verified_web_tests.verified_court").load_verified_court

        self.matrix = np.array([[40., 0., 80.], [0., 15., 50.], [0., 0., 1.]])
        self.quad = np.array([[80., 50.], [519., 50.], [519., 407.], [80., 407.]], dtype=np.float32)
        self.arrays = {
            "court_to_image": np.stack([
                self.matrix, np.full((3, 3), np.nan), np.full((3, 3), np.nan), self.matrix,
            ]),
            "court_quad": np.stack([
                self.quad, np.full((4, 2), np.nan), np.full((4, 2), np.nan), self.quad,
            ]),
            "trajectory_xy": np.full((4, 2), np.nan),
            "segment_id": np.array([1, 2, 2, 3], dtype=np.int32),
            "displayed": np.array([True, True, False, True]),
        }
        self.metadata = {
            "source_path": str(self.root / "test.mp4"),
            "source_sha256": hashlib.sha256(self.source_bytes).hexdigest(),
            "frame_count": 4,
            "expected_frames": 4,
            "decoded_all_source_frames": True,
            "calibrated_frames": 2,
            "width": 640,
            "height": 480,
            "fps": 30.0,
            "court": {"method": "reference_registered", "corners": self.quad.tolist(), "has_image": True},
            "bounces": [{"frame": 0, "confidence": .72}],
        }
        self.manifest = self.evidence / "verified-evidence.json"
        self.manifest.write_text(
            json.dumps({"test": {"result_dir": "recovered/test", "court_frame": 0, "bounce_frame": 0}}),
            encoding="utf-8",
        )
        self.write_evidence()

    def write_evidence(self):
        (self.result_dir / "result.json").write_text(json.dumps(self.metadata), encoding="utf-8")
        np.savez_compressed(self.result_dir / "analysis.npz", **self.arrays)

    def test_same_content_recognized_with_a_different_upload_name(self):
        geometry = self.load(self.path, self.evidence)
        self.assertIsNotNone(geometry)
        self.assertTrue(geometry.matches(30., 640, 480, 4))
        for parameters in [(25., 640, 480, 4), (30., 641, 480, 4), (30., 640, 481, 4), (30., 640, 480, 3)]:
            with self.subTest(parameters=parameters):
                self.assertFalse(geometry.matches(*parameters))

    def test_original_filename_with_other_contents_does_not_match(self):
        same_name = self.root / "test.mp4"
        same_name.write_bytes(b"not the verified video")
        self.assertIsNone(self.load(same_name, self.evidence))

    def test_unverified_frames_never_expose_a_projection_matrix(self):
        geometry = self.load(self.path, self.evidence)
        self.assertIsNotNone(geometry)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        confirmed = geometry.update(frame, 0)
        verifying = geometry.update(frame, 1)
        absent = geometry.update(frame, 2)
        next_segment = geometry.update(frame, 3)
        self.assertTrue(confirmed["calibrated"])
        np.testing.assert_allclose(confirmed["H"], np.linalg.inv(self.matrix))
        np.testing.assert_array_equal(confirmed["H_inv"], self.matrix)
        np.testing.assert_array_equal(confirmed["court_poly"], self.quad.astype(np.int32))
        for state in (verifying, absent):
            self.assertFalse(state["calibrated"])
            self.assertIsNone(state["H"])
            self.assertIsNone(state["H_inv"])
            self.assertIsNone(state["court_poly"])
        self.assertTrue(next_segment["calibrated"])
        self.assertEqual([state["segment_id"] for state in (confirmed, verifying, absent, next_segment)], [1, 2, 2, 3])

    def test_missing_or_malformed_index_leaves_automatic_path_available(self):
        self.manifest.unlink()
        self.assertIsNone(self.load(self.path, self.evidence))
        self.manifest.write_text("{malformed", encoding="utf-8")
        self.assertIsNone(self.load(self.path, self.evidence))

    def test_bad_array_shape_and_singular_matrix_reject_the_cache(self):
        for malformed in (np.zeros((4, 3, 2)), np.zeros((4, 3, 3))):
            with self.subTest(shape=malformed.shape):
                self.arrays["court_to_image"] = malformed
                self.write_evidence()
                self.assertIsNone(self.load(self.path, self.evidence))

    def test_only_results_for_all_source_frames_are_eligible(self):
        self.metadata["decoded_all_source_frames"] = False
        self.write_evidence()
        self.assertIsNone(self.load(self.path, self.evidence))

    def test_upload_passes_verified_geometry_only_for_exact_source(self):
        import httpx

        server = importlib.import_module("verified_web_tests.app")
        uploads = self.root / "uploads"
        outputs = self.root / "outputs"
        uploads.mkdir()
        outputs.mkdir()
        unexpected_bytes = b"filename matches only"
        calls = []

        def process(in_path, out_path, progress_cb=None, *, court_debug_path,
                    limit_to_source_fps, analysis_status_cb, **options):
            upload = Path(in_path)
            payload = upload.read_bytes()
            if payload == self.source_bytes:
                self.assertIn("court_geometry", options)
                geometry = options["court_geometry"]
                self.assertTrue(geometry.matches(30., 640, 480, 4))
                self.assertTrue(geometry.update(np.zeros((480, 640, 3), np.uint8), 0)["calibrated"])
            else:
                self.assertEqual(payload, unexpected_bytes)
                self.assertNotIn("court_geometry", options)
            calls.append((payload, limit_to_source_fps()))
            analysis_status_cb(dict(source_fps=30., analysis_frames=4))
            return [], {"method": "test", "corners": [], "has_image": False}

        async def check():
            cases = (
                ("renamed.mp4", self.source_bytes, True),
                ("test.mp4", unexpected_bytes, False),
            )
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server.app), base_url="http://test"
            ) as client:
                for filename, contents, limited in cases:
                    response = await client.post(
                        "/upload",
                        data={"limit_to_source_fps": str(limited).lower()},
                        files={"file": (filename, contents, "video/mp4")},
                    )
                    self.assertEqual(response.status_code, 200)
                    key = response.json()["job_id"]
                    try:
                        state = (await client.get(f"/status/{key}")).json()
                        self.assertEqual(state["status"], "done", state.get("message"))
                        self.assertEqual(state["analysis_frames"], 4)
                        self.assertEqual(state["limit_to_source_fps"], limited)
                    finally:
                        server.job_status.pop(key, None)

        with patch.object(server, "UPLOAD_DIR", uploads), patch.object(server, "OUTPUT_DIR", outputs), \
             patch.object(server, "VERIFIED_COURT_DIR", self.evidence), patch.object(server, "process_video", process):
            asyncio.run(asyncio.wait_for(check(), timeout=15))
        self.assertEqual(calls, [(self.source_bytes, True), (unexpected_bytes, False)])
        self.assertEqual(list(uploads.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
