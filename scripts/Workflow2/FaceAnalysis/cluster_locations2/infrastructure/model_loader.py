from pathlib import Path
import onnxruntime as ort

from _common.onnx_manager import ONNXModelManager, suppress_output
from .model_downloader import ModelDownloader


class ModelLoader:
    def __init__(self, model_path: Path, provider: dict):
        ModelDownloader().ensure(model_path)

        self.manager = ONNXModelManager(provider)
        self.session = self._init_session(model_path)

    def _init_session(self, path: Path) -> ort.InferenceSession:
        with suppress_output():
            session = self.manager.get_session(path)

        if not session:
            raise RuntimeError("ONNX session failed")

        return session

    def shutdown(self):
        self.manager.shutdown()
