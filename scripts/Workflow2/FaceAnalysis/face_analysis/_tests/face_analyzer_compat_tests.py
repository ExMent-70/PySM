import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np


FACE_ANALYSIS_DIR = Path(__file__).resolve().parents[1]
FACE_ANALYSIS_ROOT = FACE_ANALYSIS_DIR.parent
for path in (FACE_ANALYSIS_DIR, FACE_ANALYSIS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from face_lib import face_analyzer as face_analyzer_module
from face_lib.face_analyzer import FaceAnalyzer, FaceAnalyzerInitError


class _Config:
    def __init__(self, values):
        self.values = values

    def get(self, key, default=None):
        return self.values.get(key, default)


class FaceAnalyzerCompatibilityTests(unittest.TestCase):
    def test_cpu_provider_is_rejected_even_when_gpu_package_is_installed(self):
        analyzer = FaceAnalyzer.__new__(FaceAnalyzer)
        analyzer.onnx_manager = SimpleNamespace(provider_name="CPUExecutionProvider")
        missing_cpu_dist = face_analyzer_module.metadata.PackageNotFoundError("onnxruntime")

        with patch.object(
            face_analyzer_module.metadata,
            "version",
            side_effect=["1.27.0", missing_cpu_dist],
        ):
            with self.assertRaisesRegex(FaceAnalyzerInitError, "GPU-провайдер"):
                analyzer._validate_gpu_runtime()

    def test_insightface_detector_size_stays_640(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_root = Path(temp_dir)
            (model_root / "models" / "buffalo_l").mkdir(parents=True)

            analyzer = FaceAnalyzer.__new__(FaceAnalyzer)
            analyzer.det_thresh = 0.5
            analyzer.onnx_manager = SimpleNamespace(
                provider_name="CUDAExecutionProvider",
                provider_options=[{"device_id": "0"}],
            )
            analyzer.config_manager = _Config({
                "paths.model_root": str(model_root),
                "model.name": "buffalo_l",
            })

            app = MagicMock()
            with patch.object(face_analyzer_module, "FaceAnalysis", return_value=app):
                result = analyzer._initialize_insightface()

        self.assertIs(result, app)
        app.prepare.assert_called_once_with(
            ctx_id=0,
            det_thresh=0.5,
            det_size=(640, 640),
        )

    def test_source_image_is_prepared_on_1280_canvas(self):
        analyzer = FaceAnalyzer.__new__(FaceAnalyzer)
        analyzer.source_canvas_size = (1280, 1280)
        analyzer.analyzer = MagicMock()
        analyzer.analyzer.get.return_value = []

        image = np.zeros((900, 1600, 3), dtype=np.uint8)
        faces, embeddings, original_shape = analyzer.analyze_image(image, "sample.jpg")

        self.assertIsNone(faces)
        self.assertIsNone(embeddings)
        self.assertEqual(original_shape, (900, 1600))
        detector_image = analyzer.analyzer.get.call_args.args[0]
        self.assertEqual(detector_image.shape, (1280, 1280, 3))


if __name__ == "__main__":
    unittest.main()
