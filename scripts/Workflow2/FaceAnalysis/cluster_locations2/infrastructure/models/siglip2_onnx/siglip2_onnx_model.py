from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from _common.onnx_manager import ONNXModelManager, suppress_output

from .siglip2_onnx_downloader import Siglip2OnnxDownloader


class Siglip2OnnxModel:
    def __init__(
        self,
        model_dir: Path,
        vision_model: str,
        text_model: str,
        tokenizer_path: Path,
        provider: dict,
        input_size=(384, 384),
        image_output: str = "last_hidden_state",
        spatial_strategy: str = "flatten_axis1_norm",
    ):
        self.model_dir = model_dir
        self.vision_path = model_dir / vision_model
        self.text_path = model_dir / text_model
        self.tokenizer_path = tokenizer_path
        self.input_size = tuple(input_size)
        self.image_output = image_output
        self.spatial_strategy = spatial_strategy
        self.provider = dict(provider)
        self.downloader = Siglip2OnnxDownloader()
        self.downloader.ensure_vision_model(self.model_dir, vision_model)
        self.manager = ONNXModelManager(provider)
        self.session = self.manager.get_session(self.vision_path)
        if self.session is None:
            raise RuntimeError(f"SigLIP2 ONNX vision model not found: {self.vision_path}")
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self._resolve_output_name(image_output)
        self.pooler_output_name = self._resolve_output_name("pooler_output")
        self.text_manager = None
        self.text_session = None
        self.tokenizer = None
        self._closed = False

    def _resolve_output_name(self, requested: str) -> str:
        outputs = self.session.get_outputs()
        names = [output.name for output in outputs]
        if requested in names:
            return requested
        if requested == "last_hidden_state":
            return names[0]
        if requested == "pooler_output" and len(names) > 1:
            return names[1]
        raise RuntimeError(
            f"SigLIP2 ONNX output not found: {requested}; available outputs: {names}"
        )

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        image = cv2.resize(image, self.input_size, interpolation=cv2.INTER_CUBIC)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        image = (image - 0.5) / 0.5
        return image.transpose(2, 0, 1)

    def _normalize_rows(self, matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        return matrix / norms

    def _grid_pool(self, output: np.ndarray, grid_size: int) -> np.ndarray:
        if output.ndim != 3:
            return output.reshape(output.shape[0], -1)

        batch, tokens, channels = output.shape
        side = int(np.sqrt(tokens))
        if side * side != tokens:
            return output.mean(axis=1)

        grid = output.reshape(batch, side, side, channels)
        edges = np.linspace(0, side, grid_size + 1, dtype=int)
        pooled = []
        for y0, y1 in zip(edges[:-1], edges[1:]):
            for x0, x1 in zip(edges[:-1], edges[1:]):
                patch = grid[:, y0:y1, x0:x1, :]
                pooled.append(patch.mean(axis=(1, 2)))
        return np.stack(pooled, axis=1).reshape(batch, -1)

    def _postprocess(self, output: np.ndarray) -> np.ndarray:
        if self.spatial_strategy == "flatten_axis1_norm":
            if output.ndim == 3:
                norms = np.linalg.norm(output, axis=1, keepdims=True)
                norms[norms == 0] = 1e-12
                output = output / norms
            output = output.reshape(output.shape[0], -1)
            return self._normalize_rows(output).astype(np.float32)

        if self.spatial_strategy == "flatten":
            output = output.reshape(output.shape[0], -1)
            return self._normalize_rows(output).astype(np.float32)

        if self.spatial_strategy == "pooler":
            if output.ndim > 2:
                output = output.mean(axis=1)
            return self._normalize_rows(output).astype(np.float32)

        if self.spatial_strategy == "mean_std":
            if output.ndim != 3:
                output = output.reshape(output.shape[0], -1)
            else:
                output = np.concatenate(
                    [output.mean(axis=1), output.std(axis=1)],
                    axis=1,
                )
            return self._normalize_rows(output).astype(np.float32)

        if self.spatial_strategy.startswith("grid_"):
            try:
                grid_token = self.spatial_strategy.split("_", 1)[1]
                parts = grid_token.split("x")
                grid_size = int(parts[0])
                if len(parts) > 1 and int(parts[1]) != grid_size:
                    raise ValueError(grid_token)
            except ValueError as e:
                raise RuntimeError(
                    f"Invalid grid spatial_strategy: {self.spatial_strategy}"
                ) from e
            output = self._grid_pool(output, grid_size)
            return self._normalize_rows(output).astype(np.float32)

        raise RuntimeError(f"Unsupported spatial_strategy: {self.spatial_strategy}")

    def encode_images(self, images: list[np.ndarray]) -> np.ndarray:
        batch = np.stack([self._preprocess(image) for image in images])
        with suppress_output():
            output = self.session.run([self.output_name], {self.input_name: batch})[0]
        return self._postprocess(output)

    def encode_images_pooled(self, images: list[np.ndarray]) -> np.ndarray:
        batch = np.stack([self._preprocess(image) for image in images])
        with suppress_output():
            output = self.session.run(
                [self.pooler_output_name],
                {self.input_name: batch},
            )[0]
        return self._normalize_rows(output).astype(np.float32)

    def _ensure_text_model(self):
        if self.text_session is None:
            self.downloader.ensure_text_model(self.model_dir, self.text_path.name)
            self.text_manager = ONNXModelManager(self._text_provider_config())
            self.text_session = self.text_manager.get_session(self.text_path)
            if self.text_session is None:
                raise RuntimeError(f"SigLIP2 ONNX text model not found: {self.text_path}")

        if self.tokenizer is None:
            try:
                from tokenizers import Tokenizer
            except ImportError as e:
                raise RuntimeError(
                    "tokenizers is required for SigLIP2 ONNX text tokenization"
                ) from e

            tokenizer_json = self.tokenizer_path / "tokenizer.json"
            if not tokenizer_json.exists():
                self.downloader.ensure_tokenizer(self.tokenizer_path)
            if not tokenizer_json.exists():
                raise RuntimeError(f"Tokenizer file not found: {tokenizer_json}")

            self.tokenizer = Tokenizer.from_file(str(tokenizer_json))
            pad_id = self.tokenizer.token_to_id("<pad>")
            if pad_id is None:
                pad_id = 0
            self.tokenizer.enable_truncation(max_length=64)
            self.tokenizer.enable_padding(
                length=64,
                pad_id=pad_id,
                pad_token="<pad>",
            )

    def _text_provider_config(self) -> dict:
        config = dict(self.provider)
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            config["provider_name"] = "CUDAExecutionProvider"
        else:
            config["provider_name"] = "CPUExecutionProvider"
        return config

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        self._ensure_text_model()

        encodings = self.tokenizer.encode_batch([str(text) for text in texts])
        input_ids = np.array([encoding.ids for encoding in encodings], dtype=np.int64)
        attention_mask = np.array(
            [encoding.attention_mask for encoding in encodings],
            dtype=np.int64,
        )

        session_inputs = {item.name for item in self.text_session.get_inputs()}
        feed = {}
        if "input_ids" in session_inputs:
            feed["input_ids"] = input_ids
        if "attention_mask" in session_inputs:
            feed["attention_mask"] = attention_mask

        output_names = [item.name for item in self.text_session.get_outputs()]
        output_name = "pooler_output" if "pooler_output" in output_names else output_names[-1]
        with suppress_output():
            output = self.text_session.run([output_name], feed)[0]
        return self._normalize_rows(output).astype(np.float32)

    def similarity(self, image_emb: np.ndarray, text_emb: np.ndarray) -> np.ndarray:
        return image_emb @ text_emb.T

    def shutdown(self):
        if self._closed:
            return
        self.manager.shutdown()
        if self.text_manager is not None:
            self.text_manager.shutdown()
        self._closed = True

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass
