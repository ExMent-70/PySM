# analize/cluster_editor/_lib/data_manager.py

import logging
import json
import shutil
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from .data_models import ImageRecord, Face
from .strategies import get_strategy

# Опциональный импорт загрузчика эмбеддингов для старого кода (если понадобится)
try:
    from _common._shared import EmbeddingLoader
except ImportError:
    EmbeddingLoader = None

logger = logging.getLogger(__name__)

class ClusterDataManager:
    def __init__(self, working_dir: Path, reference_dir: Optional[Path] = None, mode: str = "face"):
        self.working_dir = working_dir
        self.reference_dir = reference_dir if reference_dir else working_dir
        
        self.info_json_path = self.working_dir / "info_faces.json"
        self.embeddings_dir = self.working_dir / "_Embeddings"

        # Основное хранилище данных: {filename: ImageRecord}
        self.records: Dict[str, ImageRecord] = {}
        
        # Кэш векторов: {key: np.array}
        self.vector_cache: Dict[str, np.ndarray] = {}
        
        # Список новых пустых кластеров (созданных кнопкой "Создать кластер")
        # Структура: [{"id": "10", "name": "New Cluster"}]
        self.newly_created_clusters: List[Dict] = []
        
        # Ручные обложки для локаций (legacy/context support)
        self.manual_covers: Dict[str, str] = {}
        self._has_unsaved_covers = False

        # Инициализация стратегии
        try:
            self.strategy = get_strategy(mode)
            logger.info(f"<b>РЕЖИМ РАБОТЫ: {self.strategy.mode_name.upper()}</b>")
        except ValueError as e:
            logger.critical(f"Failed to initialize strategy: {e}")
            raise

    def load_data(self) -> tuple[bool, str]:
        """Загружает JSON и вектора."""
        if not self.info_json_path.exists():
            return False, f"Файл не найден: {self.info_json_path}"
        
        self.records.clear()
        self.vector_cache.clear()
        self.manual_covers.clear()
        self._has_unsaved_covers = False

        # 1. Загрузка векторов
        try:
            if EmbeddingLoader and self.embeddings_dir.exists():
                loader = EmbeddingLoader(self.embeddings_dir)
                vecs, idx_map = loader.load("faces")
                if vecs is not None and idx_map is not None:
                    for fname, indices in idx_map.items():
                        for i, row_idx in enumerate(indices):
                            if row_idx < len(vecs):
                                # ВАЖНО: Ключ в кэше строится как "filename::index_in_npy"
                                # Для Способа Б индексы в NPY должны соответствовать face_index
                                self.vector_cache[f"{fname}::{i}"] = vecs[row_idx]
                                if len(indices) == 1 and i == 0:
                                    self.vector_cache[fname] = vecs[row_idx]
        except Exception as e:
            logger.error(f"Vector load error: {e}")

        # 2. Загрузка JSON
        try:
            with open(self.info_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for filename, file_data in data.items():
                    record = ImageRecord.from_dict(filename, file_data)
                    
                    if record.face_count == 1: 
                        record.image_type = 'portrait'
                    else: 
                        record.image_type = 'group'
                    
                    record.original_image_type = record.image_type
                    
                    # --- ИСПРАВЛЕНИЕ: Привязка через face_index (Способ Б) ---
                    # Если face_index существует, используем его для поиска вектора.
                    # Если нет (старый файл), откатываемся на enumerate(i).
                    
                    all_faces = record.faces + record.removed_faces
                    
                    for i, face in enumerate(all_faces):
                        # Определяем индекс для поиска вектора
                        # Приоритет: face_index > порядковый номер
                        target_idx = i
                        if face.face_index is not None:
                            target_idx = face.face_index
                        
                        # Формируем ключ
                        if record.face_count == 1 and len(all_faces) == 1:
                             # Для одиночных портретов часто используется просто имя файла
                             # Но если есть face_index, лучше быть точным
                             face.embedding_key = filename
                        else:
                             face.embedding_key = f"{filename}::{target_idx}"
                        
                        face.commit_changes()
                        
                    record.commit_changes()
                    self.records[filename] = record
        except Exception as e:
            return False, f"JSON load error: {e}"
        
        # 3. Загрузка эталонов (если мы в режиме matches и папки отличаются)
        if self.strategy.mode_name == 'matches' and self.reference_dir != self.working_dir:
            self._load_reference_clusters()

        return True, ""


    def switch_working_session(self, new_json_path: Path):
        """
        Переключает рабочую сессию на другую папку (на лету).
        Reference directory остается прежней.
        """
        self.working_dir = new_json_path.parent
        self.info_json_path = new_json_path
        self.embeddings_dir = self.working_dir / "_Embeddings"
        
        # Если reference_dir не был задан явно (был равен working_dir),
        # то при смене working_dir мы НЕ должны менять reference_dir,
        # если мы хотим сохранить эталоны.
        # В текущей логике reference_dir задается один раз при init.
        # Если мы меняем working_dir, reference_dir остается старым (что нам и нужно).
        
        logger.info(f"Switched working session to: {self.working_dir}")


    def _load_reference_clusters(self):
        """
        Загружает эталоны из reference_dir.
        Добавляет их в self.records, но помечает как 'is_reference'.
        """
        ref_json = self.reference_dir / "info_faces.json"
        if not ref_json.exists(): 
            return

        try:
            with open(ref_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                for filename, file_data in data.items():
                    # Нас интересуют только портреты (кластеры)
                    if file_data.get("face_count") != 1: continue
                    
                    # Чтобы не было конфликта имен файлов, если они совпадают с working_dir
                    # (хотя в реальных кейсах имена обычно разные или уникальные)
                    # Мы добавляем их в records как есть. Strategy должна уметь искать файл в ref dir.
                    
                    if filename in self.records:
                        continue # Приоритет у локальных файлов (хотя это странная ситуация)

                    record = ImageRecord.from_dict(filename, file_data)
                    record.image_type = 'portrait'
                    
                    # Помечаем, что это эталон (не для редактирования)
                    for face in record.faces:
                        face.extra_data['is_reference'] = True
                        # Ключи эмбеддингов здесь не нужны, т.к. мы не будем искать вектора reference
                        # (если только не захотим рисовать их, но vector_cache для reference не грузится)
                    
                    self.records[filename] = record
                    
        except Exception as e:
            logger.error(f"Error loading reference JSON: {e}")

    def get_clusters(self, mode_config: Dict = None) -> Dict[str, List[Face]]:
        """
        Возвращает сгруппированные данные.
        Делегирует логику стратегии, затем добавляет пустые (новые) кластеры.
        """
        # Получаем основные кластеры от стратегии
        clusters = self.strategy.get_clusters(self.records)
        
        # Добавляем "пустышки", созданные вручную (еще без файлов)
        for new_c in self.newly_created_clusters:
            cid = new_c["id"]
            if cid not in clusters:
                # Создаем фейковое лицо-плейсхолдер
                f = Face(bbox=[], child_name=new_c["name"])
                f.effective_name = new_c["name"]
                clusters[cid] = [f]
        return clusters

    def get_files_for_cluster(self, mode_config: Dict, cluster_id: str) -> List[str]:
        """Возвращает список файлов в кластере через стратегию."""
        return self.strategy.get_files_for_cluster(cluster_id, self.records)

    def get_group_matches_for_cluster(self, cluster_id: str) -> List[str]:
        """Специфичный метод для Matches mode (Legacy wrapper)."""
        if self.strategy.mode_name == 'matches':
            return self.strategy.get_files_for_cluster(cluster_id, self.records)
        return []

    def move_images_to_cluster(self, mode_config: Dict, target_id: str, target_name: str, 
                               filenames: List[str], face_selection_map: Dict[str, int] = None):
        """Перемещение изображений (Drag & Drop)."""
        self.strategy.move_images(
            source_id="", # Source ID часто не важен для логики изменения, важен target
            target_id=target_id,
            filenames=filenames,
            records=self.records,
            face_selection_map=face_selection_map,
            target_name=target_name
        )

    def rename_cluster(self, mode_config: Dict, cluster_id: str, new_name: str):
        """Переименование."""
        # Обновляем в стратегии (данные в записях)
        self.strategy.rename_cluster(cluster_id, new_name, self.records)
        
        # Обновляем в списке новых (если это пустой кластер)
        for c in self.newly_created_clusters:
            if c["id"] == cluster_id:
                # Учитываем префикс стратегии
                prefix = self.strategy.get_name_prefix(cluster_id)
                # Удаляем старый префикс если он был в новом имени (защита от дублей)
                clean_new_name = new_name
                if prefix and new_name.startswith(prefix):
                    clean_new_name = new_name[len(prefix):]
                c["name"] = prefix + clean_new_name

    def create_cluster(self, mode_config: Dict, new_name: str):
        """
        Создание нового пустого кластера.
        Генерирует новый ID на основе существующих.
        """
        # Собираем все существующие ID
        clusters = self.strategy.get_clusters(self.records)
        existing_ids = set(clusters.keys())
        for c in self.newly_created_clusters: 
            existing_ids.add(c["id"])
        
        # Ищем максимальный числовой ID
        max_id = 0
        for cid in existing_ids:
            if cid.isdigit(): # Только простые числа
                val = int(cid)
                if val > max_id: max_id = val
                
        new_id = str(max_id + 1)
        prefix = self.strategy.get_name_prefix(new_id)
        
        self.newly_created_clusters.append({
            "id": new_id, 
            "name": prefix + new_name.strip()
        })

    def delete_newly_created_cluster(self, cluster_id: str):
        self.newly_created_clusters = [c for c in self.newly_created_clusters if c['id'] != cluster_id]

    def is_cluster_changed(self, mode: str, cluster_id: str) -> bool:
        """
        Проверка на наличие несохраненных изменений в кластере.
        (Упрощенная логика: проверяем records, относящиеся к кластеру).
        """
        if any(c['id'] == cluster_id for c in self.newly_created_clusters): 
            return True
        
        # Получаем файлы этого кластера
        # ВАЖНО: Это немного дорогостоящая операция, но надежная
        files = self.strategy.get_files_for_cluster(cluster_id, self.records)
        for fname in files:
            rec = self.records.get(fname)
            if rec and rec.is_changed:
                return True
        return False

    def has_changes(self) -> bool:
        """Есть ли вообще какие-либо изменения."""
        if self.newly_created_clusters: return True
        if self._has_unsaved_covers: return True
        # Проверяем записи
        return any(rec.is_changed for rec in self.records.values())

    # --- Saving Logic ---

    def _standard_json_save(self) -> bool:
        """Стандартное сохранение JSON (используется большинством режимов)."""
        output_data = {}
        for filename, record in self.records.items():
            # Не сохраняем Reference записи в наш JSON
            if any(f.extra_data.get('is_reference') for f in record.faces):
                continue
                
            record.face_count = len(record.faces)
            output_data[filename] = record.to_dict()
        try:
            # Бэкап
            if self.info_json_path.exists():
                shutil.copy(self.info_json_path, self.info_json_path.with_suffix(".json.bak"))
            with open(self.info_json_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            # Коммит изменений (сброс флагов is_changed)
            for record in self.records.values():
                record.commit_changes()
            self.newly_created_clusters = []
            self._has_unsaved_covers = False
            return True
        except Exception as e:
            logger.critical(f"Standard Save error: {e}")
            return False

    def save_data(self) -> bool:
        """
        Главный метод сохранения.
        Оркестрирует вызов стратегии и стандартного сохранения.
        """
        # Конфиг путей для стратегий
        paths_config = {
            "json_path": self.info_json_path,
            "embeddings_dir": self.embeddings_dir,
            "vector_cache": self.vector_cache # Нужно для Cleaning mode
        }
        
        # 1. Вызываем сохранение стратегии
        # ВАЖНО: Для Cleaning mode это выполнит ВСЮ работу (удаление + дамп) и вернет True.
        # Для Matches mode это сохранит matches.json и вернет False.
        # Для Face/Location это вернет True (по контракту "Standard Save").
        
        strategy_handled_completely = False
        
        # СПЕЦИФИЧНАЯ ЛОГИКА (из-за разницы контрактов в Cleaning и остальных):
        if self.strategy.mode_name == 'cleaning':
            # Cleaning mode делает все сам
            if self.strategy.save(self.records, paths_config):
                # После cleaning нужно перезагрузить индексы или почистить records от удаленных
                # Но проще всего перечитать данные, если приложение продолжает работу.
                # В текущей реализации Cleaning удаляет данные из records и vector_cache "на лету" внутри save?
                # Нет, Cleaning.save пишет файлы, но не чистит self.records.
                # Поэтому мы должны вернуть True, и UI обычно перезагружает данные.
                return True
            return False
        else:
            # Остальные режимы: сначала side-effects стратегии, потом стандартный JSON
            self.strategy.save(self.records, paths_config)
            return self._standard_json_save()

    def save_cleaned_data(self) -> bool:
        """Legacy wrapper for UI compatibility."""
        return self.save_data()
        
    def save_matches_mode_data(self, *args) -> tuple[bool, str]:
        """Legacy wrapper."""
        self.save_data()
        return True, "Saved via strategy"

    # --- Location Specific Helpers (Legacy/Context) ---
    # Эти методы нужны, так как MainWindow обращается к ним напрямую.
    # В идеале их тоже нужно убрать в стратегию, но пока оставим как мост.

    def ingest_location_covers(self, context_covers: Dict[str, str]):
        """Загрузка обложек из контекста (передается извне)."""
        if not context_covers: return
        # Строим карту Имя -> ID
        name_to_id = {}
        for record in self.records.values():
            if record.location_cluster is not None and record.location_name:
                name_to_id[record.location_name] = str(record.location_cluster)
        
        for loc_name, filename in context_covers.items():
            if loc_name in name_to_id:
                cid = name_to_id[loc_name]
                if filename in self.records:
                    self.manual_covers[cid] = filename

    def set_location_cover(self, cluster_id: str, filename: str):
        if self.strategy.mode_name == 'location':
            self.manual_covers[cluster_id] = filename
            self._has_unsaved_covers = True

    def get_representative_file(self, mode_config: Dict, cluster_id: str) -> Optional[str]:
        """Выбор обложки для кластера."""
        # 1. Проверяем ручные обложки (только для локаций)
        if self.strategy.mode_name == 'location' and cluster_id in self.manual_covers:
            cover = self.manual_covers[cluster_id]
            # Валидация: файл все еще в этом кластере?
            record = self.records.get(cover)
            if record and str(record.location_cluster) == cluster_id:
                return cover
            else:
                del self.manual_covers[cluster_id]

        # 2. Дефолтная логика стратегии
        # Для этого нам нужны файлы и лица.
        files = self.strategy.get_files_for_cluster(cluster_id, self.records)
        if not files: return None
        
        # Получаем объекты лиц из первого файла для передачи в хук стратегии
        # (Хук get_preview_image принимает список лиц)
        first_file = files[0]
        faces = self.records[first_file].faces
        return self.strategy.get_preview_image(cluster_id, faces, self.records)

    def get_location_covers_dict(self) -> Dict[str, str]:
        """Возвращает словарь {LocationName: Filename} для сохранения в контекст."""
        result = {}
        if self.strategy.mode_name != 'location': return result
        clusters = self.get_clusters()
        for cid, faces in clusters.items():
            if not faces: continue
            # effective_name заполняется стратегией
            loc_name = faces[0].effective_name 
            cover_file = self.get_representative_file({}, cid)
            if cover_file and loc_name:
                result[loc_name] = cover_file
        return result
    
    # --- Manual Match Helpers (Legacy Wrapper) ---
    def assign_manual_match(self, filename: str, target_cluster_id: str, target_cluster_name: str, face_index: int):
        # Перенаправляем в move_images, так как логика там уже реализована
        self.strategy.move_images(
            source_id="", 
            target_id=target_cluster_id, 
            filenames=[filename], 
            records=self.records,
            face_selection_map={filename: face_index},
            target_name=target_cluster_name
        )
        
    def unassign_manual_match(self, filename: str, current_cluster_id: str):
        # Аналогично unassign через move_images (target="error_matches" или спец логика)
        # В стратегии matches move_images с target_id="error_matches" снимает матч
        self.strategy.move_images(
            source_id=current_cluster_id,
            target_id="error_matches",
            filenames=[filename],
            records=self.records
        )