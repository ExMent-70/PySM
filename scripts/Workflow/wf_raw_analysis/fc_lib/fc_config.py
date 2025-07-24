# fc_lib/fc_config.py

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal

import toml
from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from .fc_messages import get_message

logger = logging.getLogger(__name__)


# --- Модели Pydantic ---

class TaskConfig(BaseModel):
    run_image_analysis_and_clustering: bool = True
    keypoint_analysis: bool = True
    create_xmp_file: bool = True
    move_files_to_claster: bool = False
    generate_html: bool = True
    analyze_gender: bool = True
    analyze_emotion: bool = True
    analyze_age: bool = True
    analyze_beauty: bool = False
    analyze_eyeblink: bool = True

class PathsConfig(BaseModel):
    folder_path: str
    output_path: str
    model_root: str
    children_file: Optional[str] = None
    tensorRT_cache_path: str = "TensorRT_cache"

class ModelConfig(BaseModel):
    name: str
    det_thresh: float = 0.5
    det_size: List[int] = [1280, 1280]
    gender_model_filename: Optional[str] = "models/FACEONNX/gender_efficientnet_b2.onnx"
    emotion_model_filename: Optional[str] = "models/FACEONNX/emotion_cnn.onnx"
    age_model_filename: Optional[str] = "models/FACEONNX/age_efficientnet_b2.onnx"
    beauty_model_filename: Optional[str] = "models/FACEONNX/beauty_resnet18.onnx"
    eyeblink_model_filename: Optional[str] = "models/FACEONNX/eye_blink_cnn.onnx"
    emotion_labels: List[str] = ["Neutral", "Happiness", "Surprise", "Sadness", "Anger", "Disgust", "Fear"]
    eyeblink_labels: List[str] = ["Closed", "Open"]
    eyeblink_threshold: float = 0.5

class ClusteringPortraitConfig(BaseModel):
    algorithm: str = "HDBSCAN"
    eps: float = 0.5
    min_samples: int = 5
    metric: str = "cosine"
    min_samples_param: Optional[int] = None
    cluster_selection_epsilon: float = 0.0
    allow_single_cluster: bool = False

class ClusteringGroupConfig(BaseModel):
    algorithm: str = "HDBSCAN"
    eps: float = 0.5
    min_samples: int = 5
    metric: str = "cosine"
    visualize: bool = False
    use_auto_eps: bool = False
    percentile: int = 95

class ClusteringConfig(BaseModel):
    portrait: ClusteringPortraitConfig = Field(default_factory=ClusteringPortraitConfig)
    group: ClusteringGroupConfig = Field(default_factory=ClusteringGroupConfig)

class MatchingConfig(BaseModel):
    match_threshold: float = 0.5
    use_auto_threshold: bool = False
    percentile: int = 10

class ProcessingConfig(BaseModel):
    select_image_type: Literal["RAW", "JPEG", "PSD"] = "RAW"
    target_size: List[int] = [640, 640]
    raw_extensions: List[str] = [".arw", ".cr2", ".cr3", ".nef", ".dng"]
    psd_extensions: List[str] = [".psd", ".psb"]
    save_jpeg: bool = True
    min_preview_size: int = 2048
    max_workers: Optional[int] = None
    max_workers_limit: Optional[int] = 16
    max_concurrent_xmp_tasks: Optional[int] = 50
    block_size: int = 0

    @field_validator("raw_extensions", "psd_extensions", mode="before")
    @classmethod
    def normalize_extensions(cls, v):
        if isinstance(v, list):
            if not all(isinstance(item, str) for item in v):
                print(f"WARNING [Config Validator]: Ожидался список строк для расширений, получено: {type(v)}.")
                return v
            return [ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in v]
        return v

class MovingConfig(BaseModel):
    move_or_copy_files: bool = False
    file_extensions_to_action: List[str] = [".jpg", ".jpeg", ".xmp"]

    @field_validator("file_extensions_to_action", mode="before")
    @classmethod
    def normalize_move_extensions(cls, v):
        if isinstance(v, list):
            normalized = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in v}
            return sorted(list(normalized))
        return v

class TsneConfig(BaseModel):
    perplexity: int = 30
    max_iter: int = 1000
    random_state: int = 42

class PcaConfig(BaseModel):
    n_components: int = 2
    random_state: int = 42

class HeadPoseThresholds(BaseModel):
    yaw_thresholds: List[float] = [-30.0, -15.0, -5.0, 5.0, 15.0, 30.0]
    pitch_thresholds: List[float] = [-25.0, -10.0, -5.0, 5.0, 10.0, 25.0]
    roll_thresholds: List[float] = [-25.0, -10.0, -5.0, 5.0, 10.0, 25.0]

class KeypointAnalysisReportConfig(BaseModel):
    eye_2d_ratio_thresholds: List[float] = [0.15, 0.25, 0.35]
    eye_z_diff_threshold: float = 0.1
    mouth_2d_ratio_thresholds: List[float] = [0.2, 0.5, 0.7]
    mouth_z_diff_thresholds: List[float] = [1.5, 2.5, 3.5]
    head_pose_thresholds: HeadPoseThresholds = Field(default_factory=HeadPoseThresholds)

class ReportConfig(BaseModel):
    thumbnail_size: int = 200
    visualization_method: str = "t-SNE"
    tsne: TsneConfig = Field(default_factory=TsneConfig)
    pca: PcaConfig = Field(default_factory=PcaConfig)
    keypoint_analysis: Optional[KeypointAnalysisReportConfig] = Field(default_factory=KeypointAnalysisReportConfig)

class ProviderConfig(BaseModel):
    provider_name: Optional[str] = None
    device_id: str = "0"
    trt_fp16_enable: bool = True
    trt_max_workspace_size: str = "1073741824"
    gpu_mem_limit: Optional[int] = None

class DebugConfig(BaseModel):
    save_analyzed_kps_images: bool = False

class XmpConfig(BaseModel):
    exclude_fields: List[str] = Field(default_factory=list)

    @field_validator("exclude_fields", mode="before")
    @classmethod
    def ensure_list_of_strings(cls, v):
        if isinstance(v, list) and all(isinstance(item, str) for item in v):
            return v
        raise ValueError(f"xmp.exclude_fields должен быть списком строк, получено: {type(v)}")

class Config(BaseModel):
    logging_level: str = "INFO"
    paths: PathsConfig
    task: TaskConfig = Field(default_factory=TaskConfig)
    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    model: ModelConfig
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    moving: MovingConfig = Field(default_factory=MovingConfig)
    clustering: ClusteringConfig = Field(default_factory=ClusteringConfig)
    matching: MatchingConfig = Field(default_factory=MatchingConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)
    xmp: XmpConfig = Field(default_factory=XmpConfig)
    debug: DebugConfig = Field(default_factory=DebugConfig)

    @field_validator("logging_level")
    @classmethod
    def logging_level_must_be_valid(cls, v):
        if v.upper() not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ValueError(f"Недопустимый уровень логирования: {v}.")
        return v.upper()


class ConfigManager:
    """Управляет загрузкой, валидацией и доступом к конфигурации."""

    def __init__(self, config_path: str = "face_config.toml"):
        self.config_path = Path(config_path)
        if not self.config_path.is_file():
            msg = f"КРИТИЧЕСКАЯ ОШИБКА: Файл конфигурации не найден: {self.config_path.resolve()}"
            print(msg)
            raise FileNotFoundError(msg)
        try:
            self.config = self._load_config()
            self.session_name_str: Optional[str] = None
            
        except ValidationError as e:
            print(f"КРИТИЧЕСКАЯ ОШИБКА: Ошибка валидации конфигурации в {self.config_path.resolve()}:\n{e}")
            raise
        except Exception as e:
            print(f"КРИТИЧЕСКАЯ ОШИБКА при инициализации ConfigManager: {e}")
            raise

    def _resolve_paths_in_section(self, config_section: Dict[str, Any], base_dir: Path):
        """Преобразует все значения в словаре в абсолютные пути."""
        if not isinstance(config_section, dict):
            return
            
        for key, value in config_section.items():
            if value and isinstance(value, str):
                path_obj = Path(value)
                if not path_obj.is_absolute():
                    resolved_path = (base_dir / path_obj).resolve()
                    config_section[key] = str(resolved_path)
                    logger.debug(f"Путь '[paths].{key}' преобразован: '{value}' -> '{resolved_path}'")

    def _load_config(self) -> dict:
        """Загружает, валидирует и обрабатывает пути в конфигурации из TOML."""
        try:
            with self.config_path.open("r", encoding="utf-8") as f:
                config_data = toml.load(f)

            validated_config = Config(**config_data).model_dump(mode="python")
            logger.debug("Конфигурация успешно прошла Pydantic валидацию.")

            base_dir = self.config_path.parent
            logger.debug(f"Базовая директория для относительных путей: {base_dir.resolve()}")
            
            if "paths" in validated_config:
                self._resolve_paths_in_section(validated_config["paths"], base_dir)
            
            logger.debug(get_message("INFO_CONFIG_LOADED", config_path=self.config_path.resolve()))
            return validated_config

        except FileNotFoundError:
            raise
        except ValidationError:
            raise
        except Exception as e:
            print(f"КРИТИЧЕСКАЯ ОШИБКА: {get_message('ERROR_CONFIG_LOAD', config_path=self.config_path.resolve(), exc=e)}")
            raise

    def get(self, section: str, key: Optional[str] = None, default: Any = None) -> Any:
        """Возвращает значение из конфигурации."""
        try:
            section_data = self.config.get(section)
            if section_data is None:
                return default
            if key:
                if isinstance(section_data, dict):
                    value = section_data.get(key)
                    return default if value is None else value
                else:
                    logger.warning(f"Попытка получить ключ '{key}' из не-словаря '{section}'.")
                    return default
            else:
                return section_data.copy() if isinstance(section_data, dict) else section_data
        except Exception as e:
            logger.error(f"Ошибка доступа к конфигурации [{section}]{f'[{key}]' if key else ''}: {e}", exc_info=True)
            return default
            
    def override_config(self, overrides: Dict[str, Any]):
        """
        Применяет словарь переопределений к существующей конфигурации.
        Поддерживает вложенность через точечную нотацию (e.g., 'paths.folder_path').
        """
        if not overrides:
            return

        logger.debug(f"Переопределение конфигурации для: {list(overrides.keys())}")
        validated_config = self.config

        for key_path, value in overrides.items():
            keys = key_path.split('.')
            if not keys:
                continue

            current_level = validated_config
            for i, key in enumerate(keys[:-1]):
                if not isinstance(current_level.get(key), dict):
                    logger.debug(f"Создание вложенного словаря для ключа '{key}' по пути '{key_path}'")
                    current_level[key] = {}
                current_level = current_level[key]

            final_key = keys[-1]
            current_level[final_key] = value
            logger.debug(f"Параметр '{key_path}' переопределен значением '{value}'")

        try:
            logger.debug("Повторная валидация конфигурации после переопределения...")
            validated_config = Config(**validated_config).model_dump(mode="python")
            self.config = validated_config
            logger.info("Конфигурация успешно загружена, переопределена и провалидирована.")
        except ValidationError as e:
            logger.error(f"Ошибка валидации конфигурации ПОСЛЕ применения переопределений: {e}")