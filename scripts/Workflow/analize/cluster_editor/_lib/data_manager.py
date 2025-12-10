# analize/cluster_editor/_lib/data_manager.py

import logging
import ijson
import json
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import defaultdict

from .data_models import ImageRecord, Face

logger = logging.getLogger(__name__)

class ClusterDataManager:
    def __init__(self, portrait_json_path: Path, group_json_path: Optional[Path] = None):
        """
        Инициализирует менеджер данных.

        Args:
            portrait_json_path: Путь к JSON с данными о портретах (эталон).
            group_json_path (Optional): Путь к JSON с данными о группах.
                                        Необязателен для режима просмотра совпадений.
        """
        self.portrait_json_path = portrait_json_path
        self.group_json_path = group_json_path

        self.records: Dict[str, ImageRecord] = {}
        self.newly_created_clusters: List[Dict] = []
        self._cluster_indices: Dict[str, Dict[str, List[str]]] = {
            'face': defaultdict(list),
            'location': defaultdict(list)
        }
        self.matches_index: Dict[str, List[tuple[str, float]]] = defaultdict(list)
        self._cluster_id_to_name_cache: Dict[str, str] = {}

    def _build_indices(self, after_load: bool = False):
        self._cluster_indices['face'].clear()
        self._cluster_indices['location'].clear()
        if after_load:
            self._cluster_id_to_name_cache.clear()

        for record in self.records.values():
            if record.image_type == 'group':
                face_cluster_id = "group"
            elif record.faces:
                face = record.faces[0]
                # --- ИСПРАВЛЕНИЕ: Явная проверка на None, чтобы 0 не превращался в -1 ---
                face_cluster_id = str(face.cluster_label if face.cluster_label is not None else -1)
                if after_load and face_cluster_id not in self._cluster_id_to_name_cache and face.child_name:
                    self._cluster_id_to_name_cache[face_cluster_id] = face.child_name
            else:
                face_cluster_id = "-1"
            self._cluster_indices['face'][face_cluster_id].append(record.filename)

            # --- ИСПРАВЛЕНИЕ: Аналогичная проверка для location_cluster ---
            location_cluster_id = str(record.location_cluster if record.location_cluster is not None else -1)
            self._cluster_indices['location'][location_cluster_id].append(record.filename)

    def _build_matches_index(self):
        self.matches_index.clear()
        temp_matches = defaultdict(list)
        for record in self.records.values():
            if record.image_type != 'group': continue
            for face in record.faces:
                label = face.extra_data.get('matched_portrait_cluster_label')
                distance = face.extra_data.get('match_distance')
                if label is not None and distance is not None:
                    temp_matches[str(label)].append((record.filename, float(distance)))
        self.matches_index = {
            cid: sorted(pairs, key=lambda x: x[1]) for cid, pairs in temp_matches.items()
        }

    def has_changes(self) -> bool:
        if self.newly_created_clusters:
            return True
        return any(record.is_changed for record in self.records.values())

    def load_data(self) -> tuple[bool, str]:
# --- НАЧАЛО ИЗМЕНЕНИЯ ---
        if not self.portrait_json_path.is_file():
            return False, "Эталонный JSON-файл ('info_portrait_faces.json') не найден."
        
        self.records.clear()
        try:
            # Загрузка портретов обязательна всегда
            with open(self.portrait_json_path, 'r', encoding='utf-8') as f:
                items = ijson.kvitems(f, '', use_float=True)
                for filename, data in items:
                    self.records[filename] = ImageRecord.from_dict(filename, 'portrait', dict(data))
            
            # Загрузка групп опциональна и зависит от наличия пути
            if self.group_json_path and self.group_json_path.is_file():
                with open(self.group_json_path, 'r', encoding='utf-8') as f:
                    items = ijson.kvitems(f, '', use_float=True)
                    for filename, data in items:
                        self.records[filename] = ImageRecord.from_dict(filename, 'group', dict(data))
# --- КОНЕЦ ИЗМЕНЕНИЯ ---

        except Exception as e:
            return False, f"Ошибка при потоковом чтении данных:\n\n{e}\n\n{traceback.format_exc()}"
        
        self._build_indices(after_load=True)
        # В режиме 'matches' индекс будет построен позже, при загрузке файла.
        # В других режимах он построится из 'info_group_faces.json'.
        self._build_matches_index() 
        logger.info(f"Загружено {len(self.records)} записей об изображениях. Построены индексы.")
        return True, ""


# analize/cluster_editor/_lib/data_manager.py -> class ClusterDataManager

    # --- НАЧАЛО НОВОГО МЕТОДА ---
    def reload_group_data(self, group_json_path: Path) -> bool:
        """
        Перезагружает данные о групповых фотографиях из нового источника.

        Args:
            group_json_path: Путь к новому файлу 'info_group_faces.json'.

        Returns:
            True в случае успеха, иначе False.
        """
        if not group_json_path.is_file():
            logger.error(f"Файл групповых данных не найден: {group_json_path}")
            return False

        # Шаг 1: Удаляем все старые записи, относящиеся к группам
        filenames_to_delete = [
            fname for fname, record in self.records.items() 
            if record.original_image_type == 'group'
        ]
        for fname in filenames_to_delete:
            del self.records[fname]
        
        logger.debug(f"Удалено {len(filenames_to_delete)} старых записей о групповых фото.")

        # Шаг 2: Загружаем новые данные из указанного файла
        try:
            with open(group_json_path, 'r', encoding='utf-8') as f:
                items = ijson.kvitems(f, '', use_float=True)
                new_records_count = 0
                for filename, data in items:
                    self.records[filename] = ImageRecord.from_dict(filename, 'group', dict(data))
                    new_records_count += 1
            
            logger.info(f"Загружено {new_records_count} новых записей о групповых фото.")

        except Exception as e:
            logger.error(f"Ошибка при чтении нового файла групповых данных: {e}")
            # В случае ошибки, лучше полностью очистить записи, чтобы избежать несоответствий
            self.records.clear()
            return False

        # Шаг 3: Полностью перестраиваем все индексы
        self._build_indices(after_load=True)
        self._build_matches_index()
        logger.info("Все внутренние индексы успешно перестроены.")
        return True
    # --- КОНЕЦ НОВОГО МЕТОДА ---



    def save_data(self) -> bool:
        try:
            with open(self.portrait_json_path, 'r', encoding='utf-8') as f:
                full_portrait_data = json.load(f)
            with open(self.group_json_path, 'r', encoding='utf-8') as f:
                full_group_data = json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"Не удалось прочитать исходные JSON для слияния: {e}")
            return False

        for record in self.records.values():
            if not record.is_changed: continue
            
            target_dict = full_portrait_data if record.original_image_type == 'portrait' else full_group_data
            
            if record.image_type != record.original_image_type:
                source_dict = full_group_data if record.original_image_type == 'group' else full_portrait_data
                if record.filename in source_dict:
                    del source_dict[record.filename]
                
                new_target_dict = full_portrait_data if record.image_type == 'portrait' else full_group_data
                new_target_dict[record.filename] = record.to_dict()
            elif record.filename in target_dict:
                target_dict[record.filename].update(record.to_dict())

        try:
            with open(self.portrait_json_path, 'w', encoding='utf-8') as f:
                json.dump(full_portrait_data, f, ensure_ascii=False, indent=2)
            with open(self.group_json_path, 'w', encoding='utf-8') as f:
                json.dump(full_group_data, f, ensure_ascii=False, indent=2)
        except IOError as e:
            logger.critical(f"Критическая ошибка при записи JSON: {e}")
            return False
        
        for record in self.records.values():
            record.commit_changes()
        self.newly_created_clusters = []
        logger.info("Изменения успешно сохранены.")
        return True


    def _strip_name_prefix(self, name: str) -> str:
        """
        Удаляет числовой префикс 'NN-' из имени, если он есть.
        Пример: '01-Иванов Иван' -> 'Иванов Иван'
        """
        if name and '-' in name and name.split('-', 1)[0].isdigit():
            return name.split('-', 1)[1]
        return name


    def move_images_to_cluster(self, mode_config: Dict, target_id: str, target_name: str, filenames: List[str]):
        is_face_mode = mode_config["mode_name"] == 'face'
        new_id_val = int(target_id) if target_id.isdigit() else None
        
        # "Чистим" имя от UI-префикса перед сохранением в модель
        clean_target_name = self._strip_name_prefix(target_name)
        
        for filename in filenames:
            record = self.records.get(filename)
            if not record: continue
            
            if is_face_mode:
                if target_id == "group":
                    if record.image_type == 'portrait':
                        record.image_type = 'group'
                        if record.faces: record.faces[0].cluster_label = None
                else:
                    if record.image_type == 'group':
                        record.image_type = 'portrait'
                    if record.faces:
                        record.faces[0].cluster_label = new_id_val
                        record.faces[0].child_name = clean_target_name
            else:
                record.location_cluster = new_id_val
                record.location_name = target_name
        
        self._build_indices()



    def rename_cluster(self, mode_config: Dict, cluster_id: str, new_name: str):
        is_face_mode = mode_config["mode_name"] == 'face'
        
        files_to_rename = self.get_files_for_cluster(mode_config, cluster_id)
        for filename in files_to_rename:
            record = self.records[filename]
            if is_face_mode:
                if record.faces: 
                    # Присваиваем "чистое" имя, которое пришло в аргументе
                    record.faces[0].child_name = new_name
            else:
                # Для локаций префикса нет, поэтому логика остается прежней
                record.location_name = new_name

    
    def is_cluster_changed(self, mode_name: str, cluster_id: str) -> bool:
        if any(c['id'] == cluster_id for c in self.newly_created_clusters):
            return True
        is_face_mode = mode_name == 'face'
        
        for record in self.records.values():
            if not record.is_changed:
                continue

            current_id, original_id = None, None
            
            if is_face_mode:
                current_id = "group" if record.image_type == 'group' else str(record.faces[0].cluster_label if record.faces[0].cluster_label is not None else -1)
                original_id = "group" if record.original_image_type == 'group' else str(record.faces[0].original_cluster_label if record.faces[0].original_cluster_label is not None else -1)
            else:
                current_id = str(record.location_cluster if record.location_cluster is not None else -1)
                original_id = str(record.original_location_cluster if record.original_location_cluster is not None else -1)

            if cluster_id == current_id or cluster_id == original_id:
                return True
                
        return False
    
    def get_all_location_names(self) -> List[str]:
        return sorted({r.location_name for r in self.records.values() if r.location_name})


    def generate_and_save_matches_json(self, output_path: Path) -> tuple[bool, str]:
        output_data = {}
        
        # Шаг 1: Получаем ПОЛНЫЙ список всех портретных кластеров (кроме служебных)
        all_portrait_cluster_ids = [
            cid for cid in self._cluster_indices['face'].keys() 
            if cid not in ["-1", "group"]
        ]
        
        try:
            # Сортируем ID как числа, если это возможно, для красивого вывода
            sorted_cluster_ids = sorted(all_portrait_cluster_ids, key=int)
        except ValueError:
            sorted_cluster_ids = sorted(all_portrait_cluster_ids)

        # Шаг 2: Итерируемся по ВСЕМ портретным кластерам
        for cluster_id in sorted_cluster_ids:
            # Получаем совпадения (может быть пустым списком)
            matches = self.matches_index.get(cluster_id, [])
            
            # Получаем имя кластера (логика остается прежней)
            child_name = self._cluster_id_to_name_cache.get(cluster_id)
            if not child_name:
                files = self._cluster_indices['face'].get(cluster_id, [])
                if files and self.records.get(files[0]) and self.records[files[0]].faces:
                    child_name = self.records[files[0]].faces[0].child_name
            if not child_name: 
                child_name = f"Кластер {cluster_id}"
            
            # Формируем список групповых фото. Если совпадений нет, он будет пустым.
            group_photos = [{"filename": fn, "min_distance": dist, "num_faces": 1} for fn, dist in matches]
            
            # Создаем запись в итоговом файле ВСЕГДА
            output_data[cluster_id] = {
                "child_name": child_name.split('-', 1)[-1] if child_name.startswith("0") else child_name,
                "group_photos": group_photos
            }

        # Шаг 3: Сохраняем результат
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            return True, f"Файл совпадений успешно сгенерирован и сохранен:\n{output_path}"
        except (IOError, TypeError) as e:
            return False, f"Ошибка при сохранении файла совпадений: {e}"

    def get_clusters(self, mode_config: Dict) -> Dict[str, List[Face]]:
        clusters: Dict[str, List[Face]] = defaultdict(list)
        is_face_mode = mode_config["mode_name"] == 'face'
        
        for record in self.records.values():
            face = record.faces[0] if record.faces else Face(bbox=[])
            if is_face_mode:
                # --- ИСПРАВЛЕНИЕ: Явная проверка на None для cluster_label ---
                cluster_id = "group" if record.image_type == 'group' else str(face.cluster_label if face.cluster_label is not None else -1)
            else:
                cluster_id = str(record.location_cluster if record.location_cluster is not None else -1)

            if not clusters[cluster_id]:
                if is_face_mode:
                    if record.image_type == 'group': cluster_name = "_Group_Photos"
                    else:
                        cluster_name = face.child_name or f"Кластер {cluster_id}"
                        if cluster_id == "-1": cluster_name = "99-Noise"
                        elif cluster_name.startswith("Unknown"):
                             if not cluster_name.startswith("98-"): cluster_name = f"98-{cluster_name}"
                        elif cluster_id not in ["-1", "group"]:
                            prefix = mode_config['name_prefix_logic'](cluster_id)
                            cluster_name = prefix + cluster_name.split('-', 1)[-1]
                else:
                    cluster_name = record.location_name or f"Локация {cluster_id}"
                face.effective_name = cluster_name
            
            face.filename = record.filename
            clusters[cluster_id].append(face)
        return dict(clusters)

    def get_files_for_cluster(self, mode_config: Dict, cluster_id: str) -> List[str]:
        index_key = 'face' if mode_config["mode_name"] == 'face' else 'location'
        return sorted(self._cluster_indices[index_key].get(cluster_id, []))

    def get_group_matches_for_cluster(self, cluster_id: str) -> List[str]:
        return [filename for filename, _ in self.matches_index.get(cluster_id, [])]
    
    def create_cluster(self, mode_config: Dict, new_name: str):
        index_key = 'face' if mode_config["mode_name"] == 'face' else 'location'
        existing_ids = set(self._cluster_indices[index_key].keys())
        for cluster_data in self.newly_created_clusters:
            existing_ids.add(cluster_data["id"])
        numeric_ids = {int(cid) for cid in existing_ids if cid.isdigit()}
        numeric_ids.add(0)

        new_id = max(numeric_ids) + 1
        new_id_str = str(new_id)

        prefix = mode_config['name_prefix_logic'](new_id_str)
        final_new_name = prefix + new_name.strip()
        self.newly_created_clusters.append({"id": new_id_str, "name": final_new_name})

    def delete_newly_created_cluster(self, cluster_id: str):
        self.newly_created_clusters = [c for c in self.newly_created_clusters if c['id'] != cluster_id]