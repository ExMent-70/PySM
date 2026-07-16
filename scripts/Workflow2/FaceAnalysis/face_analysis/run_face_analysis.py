# analize/analyze_faces/run_analyze_faces.py

"""   
    Скрипт поддерживает режим Smart Sync (--a_af_mode sync) для точечного
    обновления координат лиц на фотографиях, откадрированных в Photoshop,
    без потери привязки к кластерам и именам.
"""

print("<b>ПОИСК ЛИЦ НА ФОТОГРАФИЯХ</b>")
print("<i>Инициализация...</i><br>")

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message="`rcond` parameter will change to the default of machine precision", module="insightface.utils.transform")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*SimilarityTransform\\.from_estimate.*", module="insightface.utils.face_align")



import argparse
import logging
import os
import sys
import json
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set

try:
    current_script_path = Path(__file__).resolve()
    project_root = current_script_path.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    from _common.face_storage import FaceStorageManager
    from face_analysis.face_lib import ConfigManager, FaceAnalyzer, FaceAnalyzerInitError
    from face_analysis.face_lib.result_writer import AnalysisResultWriter    
    
    from pysm_lib import pysm_context
    from pysm_lib.pysm_context import ConfigResolver
    from pysm_lib.pysm_progress_reporter import tqdm
    from pysm_lib.pysm_report_api import ResourceNode, StandardTreeBuilder

    IS_MANAGED_RUN = True
except ImportError as e:
    print(f"Критическая ошибка импорта: {e}", file=sys.stderr)
    sys.exit(1)

from _common import icon_ok, icon_warning, icon_error, icon_info, icon_save, icon_save_warning, icon_save_error

logger = logging.getLogger(__name__)


def save_json_atomic(path: Path, data: Any, indent: int = 2) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp")
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
    tmp_path.replace(path)


class SmartSyncManager:
    """Управляет точечным обновлением геометрии лиц с использованием биометрического матчинга."""
    
    def __init__(self, output_dir: Path, embeddings_path: Path):
        self.output_dir = output_dir
        self.embeddings_path = embeddings_path
        self.index_path = self.embeddings_path.parent / "faces_index.json"
        self.global_embeddings = None
        self.faces_index: Dict[str, List[int]] = dict()
        
        if self.embeddings_path.exists():
            try:
                # Читаем файл с диска без полной загрузки в RAM
                self.global_embeddings = np.load(self.embeddings_path, mmap_mode='r')
            except Exception as e:
                logger.error(f"Не удалось загрузить глобальные эмбеддинги для синхронизации: {e}")

        if self.index_path.exists():
            try:
                with open(self.index_path, 'r', encoding='utf-8') as f:
                    raw_index = json.load(f)
                if isinstance(raw_index, dict):
                    self.faces_index = {
                        str(filename): [int(idx) for idx in indices]
                        for filename, indices in raw_index.items()
                        if isinstance(indices, list)
                    }
            except Exception as e:
                logger.error(f"Не удалось загрузить индекс эмбеддингов для синхронизации: {e}")

    def _get_global_embedding_index(self, filename: str, face: Dict, position: int) -> Optional[int]:
        file_indices = self.faces_index.get(filename)
        if not file_indices:
            return None

        local_idx = face.get("face_index")
        if isinstance(local_idx, int) and 0 <= local_idx < len(file_indices):
            return file_indices[local_idx]

        if 0 <= position < len(file_indices):
            return file_indices[position]

        return None

    def _load_old_embeddings(self, filename: str, old_faces: List[Dict]) -> Optional[List[np.ndarray]]:
        if self.global_embeddings is None:
            logger.error(f"Пропуск {filename}: файл эмбеддингов не загружен, биометрический матчинг невозможен.")
            return None
        if not self.faces_index:
            logger.error(f"Пропуск {filename}: индекс эмбеддингов faces_index.json не загружен.")
            return None

        old_embs = list()
        for position, face in enumerate(old_faces):
            idx = self._get_global_embedding_index(filename, face, position)
            if idx is None or idx < 0 or idx >= self.global_embeddings.shape[0]:
                logger.error(f"Пропуск {filename}: некорректный индекс эмбеддинга для лица #{position}.")
                return None
            old_embs.append(self.global_embeddings[idx])

        return old_embs

    def apply_updates(self, updates: List[Tuple[str, List[Dict], List[np.ndarray], Optional[Tuple[int, int]]]]) -> Set[str]:
        if not updates: 
            return set()

        logger.info("<br><b>Синхронизация: биометрический матчинг и обновление JSON...</b>")
        updated_filenames: Set[str] = set()

        files_to_patch = dict()
        # ИСПРАВЛЕНО: Убран избыточный вызов list() для уже заполненных списков
        files_to_patch[self.output_dir / "info_faces.json"] = ["bbox", "original_bbox", "kps"]
        files_to_patch[self.output_dir / "info_faces_landmarks.json"] =["landmark_2d_106", "landmark_3d_68"]

        for filepath, allowed_keys in files_to_patch.items():
            if not filepath.exists(): 
                continue
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # ИСПРАВЛЕНО: Оптимизация O(N) поиска через Lookup Dictionary вместо вложенного цикла
                data_dict = dict()
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and "filename" in item:
                            data_dict[item["filename"]] = item
                elif isinstance(data, dict):
                    data_dict = data

                changed = False
                for filename, new_meta, new_embs, original_shape in updates:
                    # Быстрый поиск файла за O(1)
                    entry = data_dict.get(filename)
                    if not entry:
                        continue

                    faces_list = None
                    if isinstance(entry, dict) and "faces" in entry:
                        faces_list = entry["faces"]
                    elif isinstance(entry, list):
                        faces_list = entry

                    if faces_list and isinstance(faces_list, list):
                        old_faces = faces_list
                        
                        # 1. Извлекаем старые векторы по глобальным индексам из _Embeddings/faces_index.json
                        old_embs = self._load_old_embeddings(filename, old_faces)
                        if old_embs is None:
                            continue
                                
                        # 2. Жадный биометрический матчинг (Cosine Similarity)
                        matches = dict() # new_idx -> old_idx
                        used_old = set()
                        
                        for new_idx, new_emb in enumerate(new_embs):
                            best_old_idx = -1
                            max_sim = -1.0
                            
                            norm_new = np.linalg.norm(new_emb)
                            if norm_new == 0: continue
                                
                            for old_idx, old_emb in enumerate(old_embs):
                                if old_idx in used_old:
                                    continue
                                
                                norm_old = np.linalg.norm(old_emb)
                                if norm_old == 0: continue
                                
                                # Формула косинусного сходства (от -1.0 до 1.0)
                                sim = np.dot(new_emb, old_emb) / (norm_new * norm_old)
                                
                                if sim > max_sim:
                                    max_sim = sim
                                    best_old_idx = old_idx
                            
                            if best_old_idx != -1:
                                matches[new_idx] = best_old_idx
                                used_old.add(best_old_idx)

                        # 3. Применяем хирургическое обновление координат
                        for new_idx, old_idx in matches.items():
                            old_face = old_faces[old_idx]
                            new_face = new_meta[new_idx]
                            
                            for key in allowed_keys:
                                if key in new_face:
                                    old_face[key] = new_face[key]
                                    changed = True
                                    updated_filenames.add(filename)
                        
                        if matches and filepath.name == "info_faces.json" and isinstance(entry, dict) and original_shape is not None:
                            entry["original_shape"] = list(original_shape)
                            changed = True
                            updated_filenames.add(filename)

                if changed:
                    save_json_atomic(filepath, data, indent=2)
                    logger.info(f"{icon_save} Координаты в <i>{filepath.name}</i> успешно обновлены.")

            except Exception as e:
                logger.error(f"Ошибка при обновлении {filepath.name}: {e}", exc_info=True)

        return updated_filenames

    def close(self):
        mmap_obj = getattr(self.global_embeddings, "_mmap", None)
        if mmap_obj is not None:
            try:
                mmap_obj.close()
            except Exception:
                pass
        self.global_embeddings = None


def get_config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Анализ лиц на изображениях.")
    
    default_config_path = Path(__file__).parent / "config.toml"

    parser.add_argument("--a_af_config_file", type=str, dest="a_af_config_file", default=str(default_config_path))
    parser.add_argument(f"--a_af_output_dir", type=str, required=True)
    parser.add_argument(f"--a_af_input_dir", type=str, required=False, default=None)
    parser.add_argument("--all_threads", type=int, dest="all_threads", default=0)
    
    # --- Перенесенные параметры из конфига ---
    parser.add_argument("--a_af_model_name", type=str, dest="a_af_model_name", default="buffalo_l")
    parser.add_argument(
        "--a_af_det_size",
        type=int,
        dest="a_af_det_size",
        default=1280,
        help="Размер рабочей копии исходного кадра; вход SCRFD фиксирован на 640x640.",
    )
    parser.add_argument("--a_af_det_thresh", type=float, dest="a_af_det_thresh", default=0.5)
    parser.add_argument(
        "--a_af_tasks", 
        type=str, 
        nargs='*', 
        dest="a_af_tasks", 
        default=list(["gender", "emotion", "age", "beauty", "eyeblink"])
    )
    # Режим работы Smart Sync
    parser.add_argument("--a_af_mode", type=str, dest="a_af_mode", choices=list(['create', 'sync']), default='create')
    
    if IS_MANAGED_RUN and ConfigResolver:
        return ConfigResolver(parser).resolve_all()
    return parser.parse_args()


def load_and_process_task(analyzer: FaceAnalyzer, image_path: Path) -> Tuple[str, Any]:
    try:
        with open(image_path, "rb") as f:
            img_buffer = np.frombuffer(f.read(), np.uint8)
        img = cv2.imdecode(img_buffer, cv2.IMREAD_COLOR)
        
        if img is None:
            logger.warning(f"Не удалось декодировать изображение: {image_path.name}")
            return image_path.name, (None, None, None)
            
        return image_path.name, analyzer.analyze_image(img, image_path.name)
    except Exception as e:
        logger.error(f"Ошибка загрузки файла {image_path.name}: {e}")
        return image_path.name, (None, None, None)


def main():
    log_level = pysm_context.get("sys_log_level", "INFO") if IS_MANAGED_RUN and pysm_context else "INFO"
    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO), format="%(message)s", stream=sys.stdout)

    cli_config = get_config()

    try:
        config_manager = ConfigManager(Path(cli_config.a_af_config_file))
    except Exception as e:
        logger.critical(f"Ошибка конфига: {e}")
        sys.exit(1)


    # --- ПЕРЕОПРЕДЕЛЕНИЕ ПАРАМЕТРОВ ИЗ CLI ---
    config_manager.config['model']['name'] = cli_config.a_af_model_name
    config_manager.config['model']['det_size'] = list([cli_config.a_af_det_size, cli_config.a_af_det_size])
    if cli_config.a_af_det_thresh is not None:
        config_manager.config['model']['det_thresh'] = cli_config.a_af_det_thresh

    # Трансформация списка строк в булевы флаги
    tasks = cli_config.a_af_tasks if cli_config.a_af_tasks is not None else list()
    config_manager.config['task_flags']['analyze_gender'] = 'gender' in tasks
    config_manager.config['task_flags']['analyze_emotion'] = 'emotion' in tasks
    config_manager.config['task_flags']['analyze_age'] = 'age' in tasks
    config_manager.config['task_flags']['analyze_beauty'] = 'beauty' in tasks
    config_manager.config['task_flags']['analyze_eyeblink'] = 'eyeblink' in tasks


    output_dir = Path(cli_config.a_af_output_dir)    
    if cli_config.a_af_input_dir is not None:
        input_dir = Path(cli_config.a_af_input_dir)
    else:    
        input_dir = output_dir / "JPG"

    if not input_dir or not output_dir or not input_dir.is_dir():
        logger.critical(f"Проблема с путями: input={input_dir}, output={output_dir}")
        sys.exit(1)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(f"<br>Вход: <i>{input_dir.resolve()}</i>")
    logger.debug(f"Выход: <i>{output_dir.resolve()}</i><br>")

    state_file = output_dir / "sync_state.json"
    old_state = dict()
    if cli_config.a_af_mode == 'sync' and state_file.exists():
        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                old_state = json.load(f)
        except Exception:
            pass

    image_files = sorted([p for p in input_dir.glob("*.jpg") if p.is_file()])
    if not image_files:
        logger.warning("JPEG-файлы не найдены.")
        sys.exit(0)

    files_to_process = list()
    modified_files_set = set(list())
    new_state = dict()
    new_state.update(old_state)

    for p in image_files:
        try:
            st = p.stat()
            curr_mtime = st.st_mtime
            curr_size = st.st_size
            
            if cli_config.a_af_mode == 'sync':
                old_info = old_state.get(p.name)
                if not old_info:
                    files_to_process.append(p) 
                elif old_info.get("mtime") != curr_mtime or old_info.get("size") != curr_size:
                    modified_files_set.add(p.name)
                    files_to_process.append(p) 
                else:
                    pass 
            else:
                files_to_process.append(p) 
        except Exception:
            files_to_process.append(p)

    if not files_to_process:
        logger.info(f"{icon_ok} Режим <b>Sync</b>: Изменений в файлах не обнаружено. Анализ не требуется.")
        sys.exit(0)

    num_workers = cli_config.all_threads or (os.cpu_count() or 4)
    logger.info(f"Потоков: <b>{num_workers}</b>. Изображений для обработки: <b>{len(files_to_process)}</b> из {len(image_files)}.")

    try:
        face_analyzer = FaceAnalyzer(config_manager, output_dir_override=output_dir)
    except FaceAnalyzerInitError as e:
        logger.critical(str(e))
        sys.exit(1)

    # ИСПРАВЛЕНИЕ: Передаем хранилищу флаг полной очистки, если выбран режим 'create'
    is_create_mode = (cli_config.a_af_mode == 'create')
    storage_manager = FaceStorageManager(output_dir, clear_existing=is_create_mode)
    
    result_writer = AnalysisResultWriter(storage_manager, batch_size=50)    
    face_analyzer.prepare_models()
    
    processed_count = 0
    faces_found_total = 0
    skipped_files = list()
    is_finalized = False
    
    updates_buffer = list()
    sync_updated_files: Set[str] = set()

    try:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(load_and_process_task, face_analyzer, path): path for path in files_to_process}
            progress = tqdm(futures.items(), total=len(files_to_process), desc="Анализ изображений")
            
            for future, path in progress:
                filename, (result_meta, result_embeddings, original_shape) = future.result()
                
                if result_meta and result_embeddings:
                    if filename in modified_files_set:
                        updates_buffer.append((filename, result_meta, result_embeddings, original_shape))
                    else:
                        result_writer.add_result(filename, result_meta, result_embeddings, original_shape)
                    
                    faces_found_total += len(result_meta)
                    
                    st = path.stat()
                    new_state[filename] = dict(mtime=st.st_mtime, size=st.st_size)
                else:
                    skipped_files.append(filename)
                    if cli_config.a_af_mode == 'sync' and original_shape is not None and filename not in modified_files_set:
                        st = path.stat()
                        new_state[filename] = dict(mtime=st.st_mtime, size=st.st_size)
                
                processed_count += 1

        result_writer.close()

        if storage_manager.finalize():
            is_finalized = True
            
            if updates_buffer:
                emb_path = storage_manager.embeddings_dir / "faces_embeddings.npy"
                sync_manager = SmartSyncManager(output_dir, emb_path)
                try:
                    sync_updated_files = sync_manager.apply_updates(updates_buffer)
                finally:
                    sync_manager.close()
                failed_sync_files = modified_files_set - sync_updated_files
                for filename in failed_sync_files:
                    if filename in old_state:
                        new_state[filename] = old_state[filename]
                    else:
                        new_state.pop(filename, None)
                if failed_sync_files:
                    logger.warning(f"{icon_warning} Sync: не удалось безопасно обновить файлов: <b>{len(failed_sync_files)}</b>. Они будут повторно проверены при следующем запуске.")

            try:
                save_json_atomic(state_file, new_state, indent=4)
            except Exception as e:
                logger.error(f"Не удалось сохранить состояние файлов: {e}")

            if skipped_files:
                skipped_path = output_dir / "skipped_images.json"
                try:
                    save_json_atomic(skipped_path, skipped_files, indent=4)
                    logger.info(f"{icon_save_warning} файл <i>skipped_images.json</i> сохранен (пропущено <b>{len(skipped_files)}</b> изображений)<br>")
                except Exception as e:
                    pass

            logger.debug(f"<br>Анализ завершен. Обработано файлов: <b>{processed_count}</b>. Найденных лиц: <b>{faces_found_total}</b>.<br>")
        else:
            is_finalized = True
            logger.error(f"<br>{icon_save_error} Анализ завершен с ошибками при сохранении данных<br>")

    except KeyboardInterrupt:
        logger.warning("\nПрерывание пользователем.")
        if not is_finalized:
            result_writer.close()
            storage_manager.finalize()
        raise
    finally:
        face_analyzer.shutdown()

    script_dir = Path(__file__).resolve().parent
    tv_builder = StandardTreeBuilder(icon_size=28)

    root_node_config = ResourceNode("config.toml", Path(script_dir) / "config.toml", "file", "Файл конфигурации")
    root_node_target = ResourceNode("Исходная<br>папка", Path(input_dir), "folder", "Исходная папка")
    root_node = ResourceNode("Рабочая<br>папка", Path(output_dir), "folder", "Результаты анализа")
    root_node.children.append(ResourceNode("info_faces.json", Path(output_dir) / "info_faces.json", "code", ""))
    root_node.children.append(ResourceNode("sync_state.json", Path(output_dir) / "sync_state.json", "code", ""))
    
    tv_builder.add_section("",[root_node_config, root_node_target, root_node])
    pysm_context.log_html(tv_builder.get_html())

if __name__ == "__main__":
    main()
