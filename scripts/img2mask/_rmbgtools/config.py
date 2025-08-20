# ======================================================================================
# Блок 1: Импорты
# ======================================================================================
import os
import tomllib
from typing import Literal, List, Optional
from pydantic import BaseModel, Field, validator
from . import logger

# ======================================================================================
# Блок 2: Pydantic модели для постобработки
# ======================================================================================

class BasePostprocessConfig(BaseModel):
    """Общие параметры постобработки для всех команд."""
    # ИСПРАВЛЕНИЕ: Переименовано с background_type на background
    background: Literal["Alpha", "Color"] = Field("Alpha", description="Тип фона: Alpha (прозрачный) или Color (цветной).")
    background_color: str = Field("#000000", description="Цвет фона в HEX (#RRGGBB или #RRGGBBAA).")
    mask_blur: int = Field(0, ge=0, description="Радиус размытия маски. 0 - без размытия.")
    mask_offset: int = Field(0, description="Смещение краев маски. >0 расширяет, <0 сужает.")
    smooth: float = Field(0.0, ge=0.0, description="Сила сглаживания краев маски.")
    fill_holes: bool = Field(False, description="Заполнять ли 'дыры' в маске.")
    invert_output: bool = Field(False, description="Инвертировать ли финальную маску.")

    @validator('background_color')
    def validate_hex_color(cls, v):
        """Проверяет, что строка является корректным HEX-цветом."""
        if not v.startswith('#') or not all(c in '0123456789abcdefABCDEF' for c in v[1:]):
            raise ValueError(f"'{v}' is not a valid HEX color.")
        if len(v) not in (7, 9): # #RRGGBB or #RRGGBBAA
             raise ValueError(f"HEX color '{v}' must be in #RRGGBB or #RRGGBBAA format.")
        return v


class RemovePostprocessConfig(BasePostprocessConfig):
    """Параметры постобработки, специфичные для команды 'remove'."""
    refine_foreground: bool = Field(True, description="Уточнять ли цвета на границах объекта.")


# ======================================================================================
# Блок 3: Pydantic модели для команд
# ======================================================================================

class RmbgSpecificConfig(BaseModel):
    """Параметры, уникальные для модели RMBG-2.0."""
    sensitivity: float = Field(1.0, ge=0.0, le=1.0, description="Чувствительность детекции маски (0.0 до 1.0).")
    process_res: int = Field(1024, ge=256, le=2048, description="Разрешение обработки изображения.")

class RemoveConfig(BaseModel):
    """Конфигурация для команды 'remove'."""
    model_name: str = Field("RMBG-2.0", description="Модель для удаления фона.")
    rmbg_specific: RmbgSpecificConfig = Field(default_factory=RmbgSpecificConfig)
    postprocess: RemovePostprocessConfig = Field(default_factory=RemovePostprocessConfig)

class SegmentConfig(BaseModel):
    """Конфигурация для команды 'segment'."""
    sam_model_name: str = Field("SAM2-Hiera-L", description="Модель SAM2 для использования.")
    dino_model_name: str = Field("GroundingDINO-B", description="Модель GroundingDINO для использования.")
    threshold: float = Field(0.35, ge=0.0, le=1.0, description="Порог уверенности для детекции DINO.")
    postprocess: BasePostprocessConfig = Field(default_factory=BasePostprocessConfig)
    
# ======================================================================================
# Блок 4: Глобальная конфигурация и функция-загрузчик
# ======================================================================================

class GlobalConfig(BaseModel):
    """Глобальные настройки приложения."""
    model_dir: str = Field("models", description="Директория для кэширования моделей.")
    device: str = Field("auto", description="Устройство для вычислений ('auto', 'cpu', 'cuda').")
    num_threads: int = Field(4, ge=1, description="Количество потоков для параллельной обработки.")

class AppConfig(BaseModel):
    """Корневая модель конфигурации, объединяющая все секции."""
    global_settings: GlobalConfig = Field(alias="global", default_factory=GlobalConfig)
    remove_settings: RemoveConfig = Field(alias="remove", default_factory=RemoveConfig)
    segment_settings: SegmentConfig = Field(alias="segment", default_factory=SegmentConfig)

    class Config:
        populate_by_name = True

DEFAULT_CONFIG_CONTENT = """
# ===============================================
# Глобальные настройки rmbgtools
# ===============================================
[global]
# Путь для кэширования всех моделей
model_dir = "models"
# Устройство для вычислений: "auto", "cpu", "cuda"
device = "auto"
# Количество параллельных потоков для обработки
num_threads = 4

# ===============================================
# Настройки для команды "remove" (удаление фона)
# ===============================================
[remove]
# Модель по умолчанию для удаления фона.
# Доступные: "RMBG-2.0", "BiRefNet-general", и т.д.
model_name = "RMBG-2.0"

# --- Параметры, специфичные для RMBG-2.0 ---
[remove.rmbg_specific]
sensitivity = 1.0
process_res = 1024

# --- Общие параметры постобработки для "remove" ---
[remove.postprocess]
refine_foreground = true
background = "Alpha"  # "Alpha" или "Color"
background_color = "#000000"
mask_blur = 0
mask_offset = 0
smooth = 0.0
fill_holes = false
invert_output = false

# ===============================================
# Настройки для команды "segment" (сегментация)
# ===============================================
[segment]
# Модели по умолчанию для сегментации
sam_model_name = "SAM2-Hiera-L"
dino_model_name = "GroundingDINO-B"
threshold = 0.35

# --- Параметры постобработки для "segment" ---
[segment.postprocess]
background = "Alpha"
background_color = "#000000"
mask_blur = 2
mask_offset = 0
smooth = 1.0
fill_holes = true
invert_output = false
"""

def load_config(path: Optional[str] = None) -> AppConfig:
    """
    Загружает конфигурацию из TOML-файла.
    Если путь не указан, ищет 'config.toml' в текущей директории.
    Если файл не найден, создает его с настройками по умолчанию.

    Args:
        path (str, optional): Путь к файлу config.toml.

    Returns:
        AppConfig: Валидированный объект конфигурации.
    """
    if path is None:
        path = "config.toml"

    if not os.path.exists(path):
        logger.warning(f"Config file not found at '{path}'. Creating a default config.toml.")
        with open(path, "w", encoding="utf-8") as f:
            f.write(DEFAULT_CONFIG_CONTENT)

    try:
        with open(path, "rb") as f:
            toml_data = tomllib.load(f)
        
        config = AppConfig.parse_obj(toml_data)
        logger.info(f"Successfully loaded configuration from '{path}'.")
        return config

    except tomllib.TOMLDecodeError as e:
        logger.error(f"Error decoding TOML file '{path}': {e}")
        raise
    except Exception as e: # Pydantic's ValidationError
        logger.error(f"Invalid configuration in '{path}': {e}")
        raise

# ======================================================================================
# Блок 5: Тестовый блок
# ======================================================================================
if __name__ == '__main__':
    print("--- Testing config.py ---")
    
    test_config_path = "temp_config.toml"
    
    # 1. Тест создания конфига по умолчанию
    if os.path.exists(test_config_path):
        os.remove(test_config_path)
    
    print(f"\n1. Testing default config creation at '{test_config_path}'...")
    config = load_config(test_config_path)
    assert os.path.exists(test_config_path)
    assert config.global_settings.device == "auto"
    assert config.remove_settings.model_name == "RMBG-2.0"
    assert config.segment_settings.postprocess.fill_holes is True
    print("Default config creation and loading successful.")

    # 2. Тест чтения существующего конфига
    print("\n2. Testing loading an existing config...")
    config_read = load_config(test_config_path)
    assert config_read.global_settings.num_threads == 4
    print("Existing config loading successful.")

    # 3. Тест валидации pydantic (ошибка)
    print("\n3. Testing validation error...")
    bad_config_content = "[global]\nnum_threads = 0" # ge=1
    bad_config_path = "bad_config.toml"
    with open(bad_config_path, "w") as f:
        f.write(bad_config_content)
    
    try:
        load_config(bad_config_path)
    except Exception as e:
        print(f"Successfully caught expected validation error: {e}")
        assert "validation error" in str(e)

    # Очистка
    os.remove(test_config_path)
    os.remove(bad_config_path)
    print("\n--- config.py tests passed! ---")