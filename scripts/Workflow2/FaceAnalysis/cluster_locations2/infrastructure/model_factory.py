from pathlib import Path


class ModelFactory:
    _BACKENDS = {
        "clip": "_create_clip",
        "siglip2_onnx": "_create_siglip2_onnx",
    }

    @staticmethod
    def create(config):
        backend = config.model.backend.lower()
        factory_name = ModelFactory._BACKENDS.get(backend)
        if factory_name is None:
            raise ValueError(f"Unsupported model backend: {config.model.backend}")

        return getattr(ModelFactory, factory_name)(config)

    @staticmethod
    def _create_clip(config):
        from cluster_locations2.infrastructure.models.clip.clip_model import ClipModel
        from cluster_locations2.infrastructure.models.clip.clip_model_loader import (
            ClipModelLoader,
        )

        model_path = Path(config.clip.model_onnx)
        tokenizer_path = Path(config.clip.tokenizer_path)

        loader = ClipModelLoader(model_path, config.provider.model_dump())

        return ClipModel(
            model_loader=loader,
            tokenizer_path=tokenizer_path,
            input_size=tuple(config.model_params.input_size),
        )

    @staticmethod
    def _create_siglip2_onnx(config):
        from cluster_locations2.infrastructure.models.siglip2_onnx.siglip2_onnx_model import (
            Siglip2OnnxModel,
        )

        return Siglip2OnnxModel(
            model_dir=Path(config.siglip2_onnx.model_dir),
            vision_model=config.siglip2_onnx.vision_model,
            text_model=config.siglip2_onnx.text_model,
            tokenizer_path=Path(config.siglip2_onnx.tokenizer_path),
            provider=config.provider.model_dump(),
            input_size=tuple(config.model_params.input_size),
            image_output=config.siglip2_onnx.image_output,
            spatial_strategy=config.siglip2_onnx.spatial_strategy,
        )
