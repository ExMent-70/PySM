"""CUDA-only adapter for the pinned ComfyUI-RMBG SDMatte implementation."""

from __future__ import annotations

import gc
import importlib
import importlib.util
import sys
from contextlib import redirect_stdout
from functools import wraps
from io import StringIO
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from ..adapters.base import AdapterLoadContext
from ..config_schema import PrecisionName
from .base import MaskRefiner, RefinerDependencyError


def validate_sdmatte_runtime(
    *,
    requested_device: str,
    requested_precision: PrecisionName | str | None = None,
) -> None:
    """Fail before a 5 GB download when CUDA or optional packages are absent."""

    missing = tuple(
        name
        for name in ("diffusers", "accelerate")
        if importlib.util.find_spec(name) is None
    )
    if missing:
        raise RefinerDependencyError(
            "Для SDMatte не установлены зависимости: " + ", ".join(missing)
        )
    import torch

    if requested_device == "cpu" or not torch.cuda.is_available():
        raise RuntimeError(
            "SDMatte из ComfyUI-RMBG 3.1.0 требует доступную CUDA."
        )
    precision_value = (
        requested_precision.value
        if isinstance(requested_precision, PrecisionName)
        else requested_precision
    )
    if precision_value == PrecisionName.BF16.value:
        raise RuntimeError("SDMatte поддерживает FP16 или FP32, но не BF16.")


class SDMatteRefiner(MaskRefiner):
    """Use an initial RMBG mask as the visual prompt for SDMatte."""

    refiner_id = "sdmatte"

    def __init__(
        self,
        *,
        model_root: Path,
        weights: Path,
        transparent_object: bool,
        constraint: float,
    ) -> None:
        super().__init__()
        self._model_root = model_root
        self._weights = weights
        self._transparent_object = transparent_object
        self._constraint = constraint
        self._model: Any = None
        self._torch: Any = None
        self._precision = PrecisionName.FP16
        self._resolution = 1024

    def _load(self, context: AdapterLoadContext) -> None:
        try:
            import torch
            import diffusers  # noqa: F401
            from safetensors.torch import load_file
        except ImportError as exc:
            raise RefinerDependencyError(
                "Для SDMatte требуются пакеты diffusers и accelerate. "
                "Установите зависимости PySM и повторите запуск."
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError(
                "SDMatte в зафиксированном ComfyUI-RMBG 3.1.0 поддерживается "
                "только на CUDA. Выберите быстрый refinement для CPU."
            )
        if context.device == "cpu":
            raise RuntimeError(
                "Для SDMatte выбрано устройство CPU, но этот runtime CUDA-only."
            )
        precision = context.precision
        if precision == PrecisionName.AUTO:
            precision = PrecisionName.FP16
        if precision == PrecisionName.BF16:
            raise RuntimeError("SDMatte поддерживает FP16 или FP32, но не BF16.")

        model_class = _load_sdmatte_class(self._model_root)
        # Upstream prints internal layer-patching messages during construction.
        # Keep stdout reserved for the final machine-readable PySM response.
        with redirect_stdout(StringIO()):
            model = model_class(
                pretrained_model_name_or_path=str(self._model_root),
                load_weight=False,
                use_aux_input=True,
                aux_input="trimap",
                use_encoder_hidden_states=True,
                use_attention_mask=True,
                add_noise=False,
            )
        state_dict = load_file(str(self._weights), device="cpu")
        incompatible = model.load_state_dict(state_dict, strict=False)
        if len(incompatible.unexpected_keys) > 16:
            raise RuntimeError(
                "Веса SDMatte несовместимы с зафиксированным runtime: "
                f"unexpected_keys={len(incompatible.unexpected_keys)}."
            )
        model.requires_grad_(False)
        model.eval()
        model.to("cuda", dtype=torch.float32)
        unet = getattr(model, "unet", None)
        _install_diffusers_scale_compatibility(unet)
        if unet is not None and hasattr(unet, "set_attn_processor"):
            from diffusers.models.attention_processor import SlicedAttnProcessor

            unet.set_attn_processor(SlicedAttnProcessor(slice_size=1))
        torch.set_float32_matmul_precision("high")

        self._torch = torch
        self._model = model
        self._precision = precision
        self._resolution = context.process_resolution

    def _refine(
        self,
        image_rgb: np.ndarray,
        initial_mask: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        torch = self._torch
        height, width = initial_mask.shape
        image = torch.from_numpy(np.array(image_rgb, copy=True)).permute(2, 0, 1)
        image = image.unsqueeze(0).to(device="cuda", dtype=torch.float32) / 255.0
        image = torch.nn.functional.interpolate(
            image,
            size=(self._resolution, self._resolution),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        image = image * 2.0 - 1.0

        trimap = torch.from_numpy(
            np.ascontiguousarray(initial_mask, dtype=np.float32)
        ).unsqueeze(0).unsqueeze(0).to("cuda")
        trimap = torch.nn.functional.interpolate(
            trimap,
            size=(self._resolution, self._resolution),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        trimap = trimap * 2.0 - 1.0
        data = {
            "image": image,
            "is_trans": torch.tensor(
                [1 if self._transparent_object else 0],
                device="cuda",
            ),
            "caption": [""],
            "trimap": trimap,
            "trimap_coords": torch.tensor(
                [[0, 0, 1, 1]],
                dtype=trimap.dtype,
                device="cuda",
            ),
        }
        autocast_enabled = self._precision == PrecisionName.FP16
        with torch.inference_mode(), torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=autocast_enabled,
        ):
            predicted = self._model(data)
        restored = torch.nn.functional.interpolate(
            predicted.float(),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )[0, 0].clamp_(0.0, 1.0).cpu().numpy()
        refined = _apply_prompt_constraints(
            restored,
            initial_mask,
            self._constraint,
        )
        metadata = {
            "refiner": self.refiner_id,
            "resolution": self._resolution,
            "precision": self._precision.value,
            "transparent_object": self._transparent_object,
            "constraint": self._constraint,
            "autocast": autocast_enabled,
        }
        return refined, metadata

    def _unload(self) -> None:
        torch = self._torch
        if self._model is not None:
            self._model.to("cpu")
        self._model = None
        self._torch = None
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()


def _load_sdmatte_class(model_root: Path) -> type[Any]:
    """Import only the checksum-verified SDMatte package under an isolated name."""

    package_name = "pysm_sdmatte_runtime"
    if package_name not in sys.modules:
        package = ModuleType(package_name)
        package.__package__ = package_name
        package.__path__ = [str(model_root)]
        sys.modules[package_name] = package
    module = importlib.import_module(
        f"{package_name}.modeling.SDMatte.meta_arch"
    )
    model_class = getattr(module, "SDMatte", None)
    if model_class is None:
        raise RuntimeError("В зафиксированном runtime отсутствует класс SDMatte.")
    return model_class


def _install_diffusers_scale_compatibility(unet: Any) -> None:
    """Drop the obsolete ``scale`` kwarg passed by the pinned SDMatte UNet.

    Diffusers already ignores this argument for ordinary down/up blocks and
    plans to remove it completely in 1.0. The upstream SDMatte forward method
    still passes it, so adapt only those exact block calls without modifying
    the checksum-verified runtime files in the model store.
    """

    if unet is None:
        return
    for collection_name in ("down_blocks", "up_blocks"):
        for block in getattr(unet, collection_name, ()):
            if getattr(block, "has_cross_attention", False):
                continue
            forward = getattr(block, "forward", None)
            if not callable(forward) or getattr(
                forward,
                "_pysm_drops_deprecated_scale",
                False,
            ):
                continue

            @wraps(forward)
            def forward_without_deprecated_scale(
                *args: Any,
                __forward: Any = forward,
                **kwargs: Any,
            ) -> Any:
                kwargs.pop("scale", None)
                return __forward(*args, **kwargs)

            forward_without_deprecated_scale._pysm_drops_deprecated_scale = True
            block.forward = forward_without_deprecated_scale


def _apply_prompt_constraints(
    predicted: np.ndarray,
    initial_mask: np.ndarray,
    constraint: float,
) -> np.ndarray:
    """Preserve confident prompt regions exactly as the upstream node does."""

    prompt = np.asarray(initial_mask, dtype=np.float32)
    result = np.asarray(predicted, dtype=np.float32).copy()
    foreground = prompt > constraint
    background = prompt < (1.0 - constraint)
    unknown = ~(foreground | background)
    result[background] = 0.0
    result[foreground] = np.clip(result[foreground] * 1.2, 0.0, 1.0)
    result[(result < 0.3) & unknown] = 0.0
    return np.ascontiguousarray(np.clip(result, 0.0, 1.0), dtype=np.float32)
