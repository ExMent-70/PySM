from pathlib import Path
import logging
import json
import sys
import numpy as np
from typing import List

from pysm_lib.pysm_progress_reporter import tqdm
from _common import icon_ok, icon_warning, icon_error, icon_info, icon_save, icon_save_warning, icon_save_error


from ..config.config import AppConfig
from ..infrastructure.file_resolver import FileResolver
from ..infrastructure.image_loader import ImageLoader
from ..infrastructure.model_factory import ModelFactory
from ..domain.models import ResolvedImage, ImageEmbedding
from ..infrastructure.cache.text_embedding_cache import TextEmbeddingCache
from ..infrastructure.cache.image_embedding_cache import ImageEmbeddingCache


logger = logging.getLogger(__name__)


# ---------------- JSON ----------------

def _load_json(path: Path) -> dict:
    if not path.exists():
        #raise RuntimeError(f"{icon_error} Файл JSON не найден: {path}")
        logger.info(f"<br>{icon_error}  Файл JSON не найден: {path}<br>")
        sys.exit(-1)           


    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        #raise RuntimeError(f"{icon_error} Ошибка чтения файла JSON: {e}")
        logger.info(f"<br>{icon_error} Ошибка чтения файла JSON: {e}<br>")
        sys.exit(-1)           


def _save_json(path: Path, data: dict):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ---------------- BATCH ----------------

def _batch(items: List, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


# ---------------- PIPELINE ----------------

def run_pipeline(
    data_dir: Path,
    config: AppConfig,
    mode: str,
    input_is_mask: bool,
    workers: int,
    batch_size: int,
):
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
        sys.exit(-1)           
        

    # ---------------- FILE FILTER ----------------

    valid_ext = {".jpg", ".jpeg", ".png"}

    files = [
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in valid_ext
    ]

    if not files:
        #raise RuntimeError(f"{icon_error} Нет изображений для обработки")
        logger.info(f"<br>{icon_error} Нет изображений для обработки<br>")
        sys.exit(-1)  

    logger.info(f"{icon_info} количество изображений для анализа: <i>{len(files)}</i>")

    # ---------------- JSON ----------------

    json_path = data_dir / "info_faces.json"
    json_data = _load_json(json_path)

    sync_state_path = data_dir / "sync_state.json"

    # ---------------- SERVICES ----------------

    resolver = FileResolver(config.model_params.mask_suffix)
    loader = ImageLoader(tuple(config.model_params.input_size))

    # ---------------- RESOLVE ----------------

    resolved: List[ResolvedImage] = resolver.resolve(files, input_is_mask)

    if not resolved:
        #raise RuntimeError(f"{icon_error} Нет валидных изображений с масками")
        logger.info(f"<br>{icon_error} Нет валидных изображений с масками<br>")
        sys.exit(-1)  
        

    logger.info(f"{icon_info} количество изображений с маской: <i>{len(resolved)}</i>")

    # ---------------------------------------------------------
    # LAZY MODEL
    # ---------------------------------------------------------

    model = None

    def get_model():
        nonlocal model
        if model is None:
            logger.info("<br><b><i>Инициализация модели...</i></b>")
            model = ModelFactory.create(config)
            logger.debug("<b>Модель готова</b> ✓")
        return model

    model_path = (Path(config.paths.model_root) / config.paths.clip_model_onnx).resolve()
    tokenizer_path = (Path(config.paths.model_root) / config.paths.tokenizer_path).resolve()

    # ---------------- CACHE ROOT ----------------

    cache_root = data_dir / "_Cache"

    # ---------------- IMAGE CACHE ----------------

    image_cache = ImageEmbeddingCache(
        cache_dir=cache_root / "image",
        sync_state_path=sync_state_path,
        data_dir=data_dir,
        model_path=model_path,
        input_size=tuple(config.model_params.input_size),
        use_originals=not input_is_mask,
        mask_suffix=config.model_params.mask_suffix,
    )

    file_names = [item.original_path.name for item in resolved]

    def _compute_embeddings(names: List[str]) -> np.ndarray:
        name_to_item = {item.original_path.name: item for item in resolved}
        ordered_items = [name_to_item[n] for n in names if n in name_to_item]

        embeddings: List[ImageEmbedding] = []

        with tqdm(total=len(ordered_items), desc="Вычисление эмбедингов") as pbar:
            for chunk in _batch(ordered_items, batch_size):

                images = []
                valid_items = []

                for item in chunk:
                    img = loader.load(item.input_path)
                    if img is None:
                        logger.warning(f"Ошибка чтения: {item.input_path.name}")
                        continue

                    images.append(img)
                    valid_items.append(item)

                if not images:
                    pbar.update(len(chunk))
                    continue

                emb = get_model().encode_images(images)

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
            sys.exit(-1)              

        name_to_vec = {e.path.name: e.vector for e in embeddings}

        return np.vstack([name_to_vec[n] for n in names if n in name_to_vec])

    image_matrix = image_cache.get_or_compute(
        file_names=file_names,
        compute_fn=_compute_embeddings,
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
            sys.exit(-1)           

        text_cache = TextEmbeddingCache(
            cache_dir=cache_root / "text",
            model_path=model_path,
            tokenizer_path=tokenizer_path,
        )

        def _compute_text(prompts):
            model = get_model()
            return model.encode_texts(prompts)

        text_embeddings = text_cache.get_or_compute(
            prompts=config.classification.prompts,
            compute_fn=_compute_text,
        )

        def _similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
            b = b / np.linalg.norm(b, axis=1, keepdims=True)
            return a @ b.T

        scores = _similarity(image_matrix, text_embeddings)        

        idx = np.argmax(scores, axis=1)
        vals = np.max(scores, axis=1)

        labels = np.where(
            vals >= config.classification.match_threshold,
            idx,
            -1
        )

    else:
        #raise RuntimeError(f"{icon_error} Неизвестный режим работы: {mode}")
        logger.info(f"<br>{icon_error} Неизвестный режим работы: {mode}<br>")
        sys.exit(-1)           
        

    # ---------------- SAVE JSON ----------------

    updated = 0

    for name, label in zip(file_names, labels):
        if name not in json_data:
            logger.warning(f"{icon_warning} для файла {name} не найден ключа в JSON")
            continue

        json_data[name]["location_cluster"] = int(label)
        updated += 1

    logger.info(f"<br>{icon_ok} обновлено записей в файле <i>{json_path.name}</i>: <i>{updated}</i>")
    _save_json(json_path, json_data)
    logger.info(f"{icon_save} файла <i>{json_path.name}</i> сохранён")


    # ---------------- SAVE EMBEDDINGS ----------------

    out_dir = data_dir / "_Embeddings"
    out_dir.mkdir(exist_ok=True)

    np.save(out_dir / "location_embeddings.npy", image_matrix)
    logger.info(f"{icon_save} файл <i>location_embeddings.npy<i> сохранён")


    index = {name: i for i, name in enumerate(file_names)}

    with (out_dir / "location_index.json").open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
        logger.info(f"{icon_save} файл <i>location_index.json<i> сохранён")

    logger.info("<br>")

    # ---------------- FINAL ----------------

    if model is not None:
        model.shutdown()

