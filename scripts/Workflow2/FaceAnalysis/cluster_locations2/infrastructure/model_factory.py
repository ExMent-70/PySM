from pathlib import Path

from cluster_locations2.infrastructure.model_loader import ModelLoader
from cluster_locations2.infrastructure.models.clip.clip_model import ClipModel


class ModelFactory:

    @staticmethod
    def create(config):
        model_root = Path(config.paths.model_root)

        model_path = model_root / config.paths.clip_model_onnx
        tokenizer_path = model_root / config.paths.tokenizer_path

        loader = ModelLoader(model_path, config.provider.model_dump())

        return ClipModel(
            model_loader=loader,
            tokenizer_path=tokenizer_path,
            input_size=tuple(config.model_params.input_size),
        )