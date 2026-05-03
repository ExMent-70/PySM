from pathlib import Path
import onnxruntime as ort
from typing import Dict, Tuple

from _common.onnx_manager import ONNXModelManager, suppress_output
from .model_downloader import ModelDownloader


class ModelLoader:
    def __init__(self, model_path: Path, provider: dict):
        ModelDownloader().ensure(model_path)

        self.manager = ONNXModelManager(provider)
        self.session, self.inputs, self.outputs = self._init(model_path)

    def _init(self, path: Path) -> Tuple[ort.InferenceSession, dict, dict]:
        with suppress_output():
            session = self.manager.get_session(path)

        if not session:
            raise RuntimeError("ONNX session failed")

        inputs = [i.name for i in session.get_inputs()]
        outputs = [o.name for o in session.get_outputs()]

        def find(candidates, pool):
            for c in candidates:
                if c in pool:
                    return c
            raise RuntimeError(f"Missing tensor {candidates}")

        return (
            session,
            {
                "pixel": find(["pixel_values", "image"], inputs),
                "ids": find(["input_ids"], inputs),
                "mask": find(["attention_mask"], inputs),
            },
            {
                "image": find(["image_embeds", "embedding"], outputs),
                "text": find(["text_embeds"], outputs),
            },
        )

    def shutdown(self):
        self.manager.shutdown()