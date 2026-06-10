from pathlib import Path
import logging
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import List

from pysm_lib.pysm_progress_reporter import tqdm
from _common import icon_ok, icon_warning, icon_error, icon_info, icon_save, icon_save_warning, icon_save_error


from ..config.config import AppConfig
from ..infrastructure.file_resolver import FileResolver
from ..infrastructure.image_loader import ImageLoader
from ..infrastructure.model_factory import ModelFactory
from ..domain.models import ResolvedImage, ImageEmbedding
from ..domain.classification_service import ClassificationService
from ..infrastructure.cache.text_embedding_cache import TextEmbeddingCache
from ..infrastructure.cache.image_embedding_cache import ImageEmbeddingCache
from ..infrastructure.cache.common import VALID_CACHE_MODES


logger = logging.getLogger(__name__)


class PipelineError(RuntimeError):
    pass


# ---------------- JSON ----------------

def _load_json(path: Path) -> dict:
    if not path.exists():
        #raise RuntimeError(f"{icon_error} Файл JSON не найден: {path}")
        logger.info(f"<br>{icon_error}  Файл JSON не найден: {path}<br>")
        raise PipelineError("JSON file not found")


    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        #raise RuntimeError(f"{icon_error} Ошибка чтения файла JSON: {e}")
        logger.info(f"<br>{icon_error} Ошибка чтения файла JSON: {e}<br>")
        raise PipelineError("JSON read failed")


def _save_json(path: Path, data: dict):
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp_path.replace(path)


# ---------------- BATCH ----------------

def _batch(items: List, size: int):
    if size <= 0:
        raise PipelineError("batch_size must be greater than 0")
    for i in range(0, len(items), size):
        yield items[i:i + size]


def _build_model_fingerprint(config: AppConfig, mode: str = "") -> dict:
    backend = config.model.backend.lower()
    mode_key = mode.lower()

    if backend == "siglip2_onnx":
        embedding_role = (
            "classification_pooler"
            if mode_key == "classification"
            else "clustering_spatial"
        )
        return {
            "backend": config.model.backend,
            "name": config.model.name,
            "embedding_role": embedding_role,
            "model_dir": str(Path(config.siglip2_onnx.model_dir).resolve()),
            "vision_model": config.siglip2_onnx.vision_model,
            "text_model": config.siglip2_onnx.text_model,
            "tokenizer_path": str(Path(config.siglip2_onnx.tokenizer_path).resolve()),
            "image_output": config.siglip2_onnx.image_output,
            "spatial_strategy": config.siglip2_onnx.spatial_strategy,
            "provider": config.provider.provider_name,
            "device_id": config.provider.device_id,
            "input_size": list(config.model_params.input_size),
            "image_preprocess": "siglip2_onnx_rgb_0_5_v1",
        }

    model_path = Path(config.clip.model_onnx).resolve()
    tokenizer_path = Path(config.clip.tokenizer_path).resolve()
    return {
        "backend": config.model.backend,
        "name": config.model.name,
        "model_path": str(model_path),
        "tokenizer_path": str(tokenizer_path),
        "provider": config.provider.provider_name,
        "device_id": config.provider.device_id,
        "input_size": list(config.model_params.input_size),
        "image_preprocess": "clip_openai_mean_std_rgb_v1",
    }


# ---------------- PIPELINE ----------------

def run_pipeline(
    data_dir: Path,
    config: AppConfig,
    mode: str,
    input_is_mask: bool,
    workers: int,
    batch_size: int,
    cache_mode: str = "use",
):
    if cache_mode not in VALID_CACHE_MODES:
        logger.info(f"<br>{icon_error} Unknown cache mode: {cache_mode}<br>")
        raise PipelineError(f"Unknown cache_mode: {cache_mode}")
    if batch_size <= 0:
        raise PipelineError("batch_size must be greater than 0")
    if workers <= 0:
        raise PipelineError("workers must be greater than 0")
    if (
        len(config.model_params.input_size) != 2
        or not all(isinstance(v, int) for v in config.model_params.input_size)
        or any(v <= 0 for v in config.model_params.input_size)
    ):
        raise PipelineError("input_size must contain two positive integers")
    if input_is_mask and not config.model_params.mask_suffix:
        raise PipelineError("mask_suffix must be set when processing masks")

    logger.info(f"<b>РЕЖИМ РАБОТЫ: {mode.upper()}</b>")
    if input_is_mask:
        logger.info("<i>для анализа используются изображения с масками</i><br>")
    else:    
        logger.info("<i>для анализа используются оригинальные изображения</i><br>")

    # ---------------- PATH ----------------

    if input_is_mask:
        input_dir = data_dir / "JPG" / "Masks"
    else:
        input_dir = data_dir / "JPG"

    if not input_dir.exists():
        #raise RuntimeError(f"{icon_error} Папка не найдена: {input_dir}")
        logger.info(f"<br>{icon_error} Папка не найдена: {input_dir}<br>")
        raise PipelineError("Input directory not found")
        

    # ---------------- FILE FILTER ----------------

    valid_ext = {".jpg", ".jpeg", ".png"}

    files = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in valid_ext
    )

    if not files:
        #raise RuntimeError(f"{icon_error} Нет изображений для обработки")
        logger.info(f"<br>{icon_error} Нет изображений для обработки<br>")
        raise PipelineError("No images to process")

    logger.info(f"{icon_info} количество изображений для анализа: <i>{len(files)}</i>")

    # ---------------- JSON ----------------

    json_path = data_dir / "info_faces.json"
    json_data = _load_json(json_path)

    # ---------------- SERVICES ----------------

    resolver = FileResolver(config.model_params.mask_suffix)
    loader = ImageLoader()

    # ---------------- RESOLVE ----------------

    resolved: List[ResolvedImage] = resolver.resolve(files, input_is_mask)

    if not resolved:
        #raise RuntimeError(f"{icon_error} Нет валидных изображений с масками")
        logger.info(f"<br>{icon_error} Нет валидных изображений с масками<br>")
        raise PipelineError("No valid resolved images")
        

    logger.info(f"{icon_info} количество изображений с маской: <i>{len(resolved)}</i>")

    # ---------------------------------------------------------
    # LAZY MODEL
    # ---------------------------------------------------------

    model = None

    def get_model():
        nonlocal model
        if model is None:
            logger.info("<br><b><i>Инициализация модели...</i></b>")
            try:
                model = ModelFactory.create(config)
            except Exception as e:
                raise PipelineError(f"Model initialization failed: {e}") from e
            logger.debug("<b>Модель готова</b> ✓")
        return model

    model_fingerprint = _build_model_fingerprint(config, mode)

    # ---------------- CACHE ROOT ----------------

    cache_root = data_dir / "_Cache"

    # ---------------- IMAGE CACHE ----------------

    mode_key = mode.lower()
    image_cache_name = (
        "image_classification"
        if mode_key == "classification"
        else "image_clustering"
    )
    image_cache_dir = cache_root / image_cache_name

    image_cache = ImageEmbeddingCache(
        cache_dir=image_cache_dir,
        model_fingerprint=model_fingerprint,
        input_size=tuple(config.model_params.input_size),
        use_originals=not input_is_mask,
        mask_suffix=config.model_params.mask_suffix,
    )

    file_names = [item.original_path.name for item in resolved]
    duplicate_names = sorted({name for name in file_names if file_names.count(name) > 1})
    if duplicate_names:
        logger.info(f"<br>{icon_error} Duplicate resolved image names: {duplicate_names[:5]}<br>")
        raise PipelineError(f"Duplicate resolved image names: {len(duplicate_names)}")

    file_items = [
        {
            "name": item.original_path.name,
            "input_path": str(item.input_path),
            "original_path": str(item.original_path),
        }
        for item in resolved
    ]

    def _load_image_item(item: ResolvedImage):
        img = loader.load(item.input_path)
        return item, img

    def _compute_embeddings(names: List[str]) -> np.ndarray:
        name_to_item = {item.original_path.name: item for item in resolved}
        ordered_items = [name_to_item[n] for n in names if n in name_to_item]

        embeddings: List[ImageEmbedding] = []
        model_instance = get_model()

        with tqdm(total=len(ordered_items), desc="Вычисление эмбедингов") as pbar:
            for chunk in _batch(ordered_items, batch_size):

                images = []
                valid_items = []

                if workers > 1:
                    max_workers = min(workers, len(chunk))
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        loaded_items = list(executor.map(_load_image_item, chunk))
                else:
                    loaded_items = [_load_image_item(item) for item in chunk]

                for item, img in loaded_items:
                    if img is None:
                        logger.warning(f"Ошибка чтения: {item.input_path.name}")
                        continue

                    images.append(img)
                    valid_items.append(item)

                if not images:
                    pbar.update(len(chunk))
                    continue

                if (
                    mode_key == "classification"
                    and hasattr(model_instance, "encode_images_pooled")
                ):
                    emb = model_instance.encode_images_pooled(images)
                else:
                    emb = model_instance.encode_images(images)

                for i, e in enumerate(emb):
                    embeddings.append(
                        ImageEmbedding(
                            path=valid_items[i].original_path,
                            vector=e
                        )
                    )

                pbar.update(len(chunk))

        if not embeddings:
            #raise RuntimeError("Не удалось вычислить эмбеддинги")
            logger.info(f"<br>{icon_error} Не удалось вычислить эмбеддинги<br>")
            raise PipelineError("No embeddings computed")

        name_to_vec = {e.path.name: e.vector for e in embeddings}
        missing = [n for n in names if n not in name_to_vec]
        if missing:
            logger.info(f"<br>{icon_error} Failed to compute embeddings for {len(missing)} images<br>")
            raise PipelineError(f"Image embeddings are missing for {len(missing)} images")

        return np.vstack([name_to_vec[n] for n in names])

    image_matrix = image_cache.get_or_compute(
        file_names=file_names,
        file_items=file_items,
        compute_fn=_compute_embeddings,
        cache_mode=cache_mode,
    )

    # ---------------- RESTORE OBJECTS ----------------

    embeddings: List[ImageEmbedding] = []
    for i, name in enumerate(file_names):
        embeddings.append(
            ImageEmbedding(
                path=Path(name),
                vector=image_matrix[i]
            )
        )

    logger.info(f"{icon_info} загружено эмбеддингов: <i>{len(embeddings)}</i>")

    # ---------------- MODE ----------------

    classification_scores = None

    if mode == "clustering":
        logger.info("<br><b><i>Кластеризация изображений...</i></b>")

        from ..domain.clustering_service import ClusteringService

        clustering = ClusteringService(
            eps=config.clustering.eps,
            min_samples=config.clustering.min_samples,
        )

        labels = clustering.run(embeddings)

    elif mode == "classification":
        logger.info("<br><b><i>Классификация изображений...</i></b>")

        if not config.classification.prompts:
            #raise RuntimeError(f"{icon_error} Не заданы prompts для classification")
            logger.info(f"<br>{icon_error} Для режима <b>{mode.upper()}</b> необходимо задать текстовый промпт<br>")
            raise PipelineError("Classification prompts are missing")

        text_cache = TextEmbeddingCache(
            cache_dir=cache_root / "text",
            model_fingerprint=model_fingerprint,
        )

        def _compute_text(prompts):
            model = get_model()
            return model.encode_texts(prompts)

        text_embeddings = text_cache.get_or_compute(
            prompts=config.classification.prompts,
            compute_fn=_compute_text,
            cache_mode=cache_mode,
        )

        classification = ClassificationService(
            threshold=config.classification.match_threshold,
        )
        classification_result = classification.classify_with_scores(
            embeddings,
            text_embeddings,
        )
        labels = classification_result.labels
        classification_scores = classification_result.scores

    else:
        #raise RuntimeError(f"{icon_error} Неизвестный режим работы: {mode}")
        logger.info(f"<br>{icon_error} Неизвестный режим работы: {mode}<br>")
        raise PipelineError(f"Unknown mode: {mode}")
        

    # ---------------- SAVE JSON ----------------

    updated = 0

    for row_index, (name, label) in enumerate(zip(file_names, labels)):
        if name not in json_data:
            logger.warning(f"{icon_warning} для файла {name} не найден ключа в JSON")
            continue

        json_data[name]["location_cluster"] = int(label)
        json_data[name]["location_name"] = str(label)
        if mode == "classification":
            json_data[name]["location_score"] = (
                float(classification_scores[row_index])
                if classification_scores is not None
                else None
            )
            json_data[name]["location_prompt"] = (
                config.classification.prompts[int(label)]
                if int(label) >= 0
                else None
            )
        updated += 1

    logger.info(f"<br>{icon_ok} обновлено записей в файле <i>{json_path.name}</i>: <i>{updated}</i>")
    _save_json(json_path, json_data)
    logger.info(f"{icon_save} файла <i>{json_path.name}</i> сохранён")


    # ---------------- SAVE EMBEDDINGS ----------------

    out_dir = data_dir / "_Embeddings"
    out_dir.mkdir(exist_ok=True)

    embeddings_filename = f"location_embeddings_{mode_key}.npy"
    index_filename = f"location_index_{mode_key}.json"

    np.save(out_dir / embeddings_filename, image_matrix)
    logger.info(f"{icon_save} файл <i>{embeddings_filename}<i> сохранён")


    index = {name: i for i, name in enumerate(file_names)}

    with (out_dir / index_filename).open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
        logger.info(f"{icon_save} файл <i>{index_filename}<i> сохранён")

    if mode == "classification":
        prompt_index = {
            str(i): prompt
            for i, prompt in enumerate(config.classification.prompts)
        }
        prompt_index["-1"] = None
        with (out_dir / "location_prompts.json").open("w", encoding="utf-8") as f:
            json.dump(prompt_index, f, indent=2, ensure_ascii=False)
        logger.info(f"{icon_save} location_prompts.json saved")

    logger.info("<br>")

    # ---------------- FINAL ----------------

    if model is not None:
        model.shutdown()
