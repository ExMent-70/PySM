"""Local-files-only adapter for RMBG-2.0, BiRefNet and Lucida checkpoints."""

from __future__ import annotations

import gc
import importlib.util
import sys
import threading
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from ..config_schema import ModelName, PrecisionName
from ..model_store import ModelFiles
from ..results import SegmentationResult
from .base import AdapterLoadContext, ModelAdapter


_IMPORT_LOCK = threading.RLock()


class LocalBiRefNetAdapter(ModelAdapter):
    """Load pinned local source and weights without editing or downloading them."""

    def __init__(self, model_id: ModelName, files: ModelFiles) -> None:
        super().__init__()
        self.model_id = model_id.value
        self._files = files
        self._model: Any = None
        self._torch: Any = None
        self._device = "cpu"
        self._dtype: Any = None
        self._mean: Any = None
        self._std: Any = None
        self._autocast_enabled = False
        self._process_resolution = 1024

    def _load(self, context: AdapterLoadContext) -> None:
        import torch
        from safetensors.torch import load_file

        device = _resolve_device(torch, context.device)
        precision, dtype = _resolve_precision(torch, context.precision, device)
        model_module, config_module = _load_model_modules(self._files)
        model_class = getattr(model_module, "BiRefNet", None)
        config_class = getattr(config_module, "BiRefNetConfig", None)
        if model_class is None or config_class is None:
            raise RuntimeError("В локальном model source отсутствует контракт BiRefNet.")

        model_config = config_class(bb_pretrained=False)
        model = model_class(config=model_config)
        state_dict = load_file(str(self._files.weights), device="cpu")
        if state_dict and next(iter(state_dict)).startswith("module."):
            state_dict = {
                key.removeprefix("module."): value for key, value in state_dict.items()
            }
        model.load_state_dict(state_dict, strict=True)
        model.requires_grad_(False)
        model.eval()
        # RMBG-2.0 contains a few float32-only internal tensors which are not
        # registered as parameters/buffers. Storing the whole model as FP16
        # therefore creates Float/Half conflicts. CUDA autocast preserves the
        # requested compute precision without corrupting those tensors.
        self._autocast_enabled = _requires_cuda_autocast(
            self.model_id,
            device,
            precision,
        )
        model_dtype = torch.float32 if self._autocast_enabled else dtype
        model.to(device=device, dtype=model_dtype)

        torch.set_float32_matmul_precision("high")
        self._torch = torch
        self._model = model
        self._device = device
        self._dtype = dtype
        self._mean = torch.tensor(
            (0.485, 0.456, 0.406),
            device=device,
            dtype=torch.float32,
        ).view(1, 3, 1, 1)
        self._std = torch.tensor(
            (0.229, 0.224, 0.225),
            device=device,
            dtype=torch.float32,
        ).view(1, 3, 1, 1)
        self._precision = precision
        self._process_resolution = context.process_resolution

    def _infer(self, image_rgb: np.ndarray) -> SegmentationResult:
        return self._infer_batch((image_rgb,))[0]

    def _infer_batch(
        self,
        images_rgb: tuple[np.ndarray, ...],
    ) -> tuple[SegmentationResult, ...]:
        torch = self._torch
        model = self._model
        source_sizes = tuple(
            (image_rgb.shape[1], image_rgb.shape[0]) for image_rgb in images_rgb
        )
        prepared = tuple(
            self._prepare_gpu_tensor(image_rgb)
            for image_rgb in images_rgb
        )
        tensor = torch.cat(prepared, dim=0)

        autocast_context = (
            torch.autocast(device_type="cuda", dtype=self._dtype)
            if self._autocast_enabled
            else nullcontext()
        )
        with torch.inference_mode(), autocast_context:
            predictions = model(tensor)
            logits = predictions[-1] if isinstance(predictions, (list, tuple)) else predictions
            if not isinstance(logits, torch.Tensor):
                raise RuntimeError("BiRefNet вернул неподдерживаемую структуру выхода.")
            masks = torch.sigmoid(logits)

        metadata = {
            "device": self._device,
            "precision": self._precision.value,
            "process_resolution": self._process_resolution,
            "batch_size": len(images_rgb),
            "preprocess": "gpu_bicubic_fp32",
            "mask_resize": "gpu_bicubic_fp32",
            "autocast": self._autocast_enabled,
        }
        return tuple(
            SegmentationResult(
                mask=self._restore_mask(mask, source_size),
                source_size=source_size,
                model_id=self.model_id,
                metadata=dict(metadata),
            )
            for mask, source_size in zip(masks, source_sizes)
        )

    def _prepare_gpu_tensor(self, image_rgb: np.ndarray) -> Any:
        torch = self._torch
        writable = np.array(
            image_rgb,
            dtype=np.uint8,
            order="C",
            copy=True,
        )
        tensor = torch.from_numpy(writable).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(device=self._device, dtype=torch.float32)
        tensor = tensor / 255.0
        tensor = torch.nn.functional.interpolate(
            tensor,
            size=(self._process_resolution, self._process_resolution),
            mode="bicubic",
            align_corners=False,
        )
        tensor = (tensor - self._mean) / self._std
        target_dtype = torch.float32 if self._autocast_enabled else self._dtype
        return tensor.to(dtype=target_dtype)

    def _restore_mask(
        self,
        mask: Any,
        source_size: tuple[int, int],
    ) -> np.ndarray:
        torch = self._torch
        width, height = source_size
        restored = torch.nn.functional.interpolate(
            mask.unsqueeze(0),
            size=(height, width),
            mode="bicubic",
            align_corners=False,
        )
        restored = restored[0, 0].float().clamp_(0.0, 1.0).cpu().numpy()
        return np.ascontiguousarray(restored, dtype=np.float32)

    def _unload(self) -> None:
        torch = self._torch
        model = self._model
        if model is not None:
            model.to("cpu")
        self._model = None
        self._torch = None
        self._mean = None
        self._std = None
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()


def _load_model_modules(files: ModelFiles) -> tuple[ModuleType, ModuleType]:
    """Load remote model code in an isolated name without rewriting cache files."""

    unique = files.model_dir.name.replace("-", "_") + "_" + files.weights.stem
    package_name = f"pysm_rmbg_{unique}"
    config_name = f"{package_name}.BiRefNet_config"
    model_name = f"{package_name}.{files.model_script.stem}"
    with _IMPORT_LOCK:
        package = sys.modules.get(package_name)
        if package is None:
            package = ModuleType(package_name)
            package.__package__ = package_name
            package.__path__ = [str(files.model_dir)]
            sys.modules[package_name] = package
        config_module = _module_from_path(config_name, files.config_script)
        previous_config = sys.modules.get("BiRefNet_config")
        sys.modules["BiRefNet_config"] = config_module
        try:
            with _modern_timm_import_aliases():
                model_module = _module_from_path(model_name, files.model_script)
        finally:
            if previous_config is None:
                sys.modules.pop("BiRefNet_config", None)
            else:
                sys.modules["BiRefNet_config"] = previous_config
    return model_module, config_module


@contextmanager
def _modern_timm_import_aliases():
    """Redirect deprecated timm module paths used by pinned model sources.

    RMBG-2.0 and the lite BiRefNet source still import from
    ``timm.models.layers`` and ``timm.models.registry``. Their required public
    symbols now live in ``timm.layers`` and ``timm.models``. Temporary aliases
    avoid deprecation warnings without changing checksum-verified model files
    or leaking compatibility modules into the rest of the process.
    """

    import timm.layers
    import timm.models

    aliases = {
        "timm.models.layers": timm.layers,
        "timm.models.registry": timm.models,
    }
    missing = object()
    previous = {name: sys.modules.get(name, missing) for name in aliases}
    sys.modules.update(aliases)
    try:
        yield
    finally:
        for name, module in previous.items():
            if module is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _module_from_path(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Не удалось создать import spec для {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _resolve_device(torch: Any, requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Выбрано CUDA, но CUDA недоступна в текущем PyTorch.")
    if requested not in {"cpu", "cuda"}:
        raise ValueError(f"Неподдерживаемое устройство: {requested}")
    return requested


def _resolve_precision(
    torch: Any,
    requested: PrecisionName,
    device: str,
) -> tuple[PrecisionName, Any]:
    if requested == PrecisionName.AUTO:
        requested = PrecisionName.FP16 if device == "cuda" else PrecisionName.FP32
    if device == "cpu" and requested != PrecisionName.FP32:
        raise RuntimeError("Для CPU в первой версии поддерживается только FP32.")
    mapping = {
        PrecisionName.FP32: torch.float32,
        PrecisionName.FP16: torch.float16,
        PrecisionName.BF16: torch.bfloat16,
    }
    return requested, mapping[requested]


def _requires_cuda_autocast(
    model_id: str,
    device: str,
    precision: PrecisionName,
) -> bool:
    """Return whether reduced precision must keep model storage in FP32."""

    return (
        model_id == ModelName.RMBG_2_0.value
        and device == "cuda"
        and precision != PrecisionName.FP32
    )
