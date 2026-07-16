import json
import re
import sys
import logging
from typing import Optional, Dict, Tuple, Any

from .context_variable_ops import initial_value_as_text, write_context_value

logger = logging.getLogger(__name__)

# ==============================================================================
# VALIDATION PRESETS (перенесены из GUI/CLI)
# ==============================================================================
VALIDATION_PRESETS: Dict[str, Dict[str, str]] = {
    "not_empty": {
        "pattern": r".+",
        "description": "Требуется любой непустой текст.",
    },
    "integer": {
        "pattern": r"^-?\d+$",
        "description": "Требуется целое число.",
    },
    "positive_integer": {
        "pattern": r"^\d+$",
        "description": "Требуется положительное целое число или ноль.",
    },
    "float": {
        "pattern": r"^-?\d+(\.\d+)?$",
        "description": "Требуется число с плавающей точкой.",
    },
    "email": {
        "pattern": r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$",
        "description": "Требуется корректный email.",
    },
    "filename_txt": {
        "pattern": r"^[^\\/:*?\"<>|]+\.txt$",
        "description": "Требуется имя файла с расширением .txt.",
    },
    "dirname_windows": {
        "pattern": r"(?i)^(?!^(con|prn|aux|nul|com[1-9]|lpt[1-9])$)[^\\/:*?\"<>|]+[^\\/:*?\"<>| .]$",
        "description": "Требуется корректное имя каталога Windows",
    },    
}


class InputProcessor:
    def __init__(self, config, pysm_context=None, managed=False):
        self.config = config
        self.pysm_context = pysm_context
        self.managed = managed

        self.pattern: Optional[str] = None
        self.error_desc: str = ""

        self._setup_validation()

    # ==============================================================================
    # VALIDATION SETUP
    # ==============================================================================
    def _setup_validation(self):
        # CLI
        valid_type = getattr(self.config, "set_valid_type", None)

        # GUI
        if valid_type is None:
            valid_type = getattr(self.config, "dlg_input_valid_type", "none")

        if valid_type == "none":
            return

        if valid_type == "custom":
            self.pattern = (
                getattr(self.config, "set_custom_regexp", None)
                or getattr(self.config, "dlg_input_custom_regexp", None)
            )

            self.error_desc = (
                getattr(self.config, "set_custom_regexp_desc", None)
                or getattr(self.config, "dlg_input_custom_regexp_desc", None)
                or "Значение не соответствует формату."
            )

            return

        # 👉 PRESET
        if valid_type in VALIDATION_PRESETS:
            preset = VALIDATION_PRESETS[valid_type]
            self.pattern = preset["pattern"]
            self.error_desc = preset["description"]

    # ==============================================================================
    # VALIDATION
    # ==============================================================================
    def validate(self, text: str) -> Tuple[bool, str]:
        if not self.pattern:
            return True, ""

        try:
            if re.fullmatch(self.pattern, text, re.IGNORECASE):
                return True, ""
            return False, self.error_desc
        except re.error as e:
            return False, f"Ошибка Regex: {e}"

    # ==============================================================================
    # CONVERSION
    # ==============================================================================
    def convert(self, value_str: str, target_type: str) -> Any:
        if target_type == "string":
            return value_str

        if target_type == "int":
            return int(value_str)

        if target_type == "float":
            return float(value_str)

        if target_type == "bool":
            return value_str.lower() in ("true", "1", "yes", "y", "on")

        if target_type == "json":
            return json.loads(value_str)

        if target_type == "auto":
            val_lower = value_str.lower()

            if val_lower in ("true", "yes", "on"):
                return True
            if val_lower in ("false", "no", "off"):
                return False

            for cast_func in (int, float):
                try:
                    return cast_func(value_str)
                except ValueError:
                    pass

            return value_str

        raise ValueError(f"Неизвестный тип: {target_type}")

    # ==============================================================================
    # CONTEXT READ
    # ==============================================================================
    def get_initial_value(self, var_name: str, default: str) -> str:
        if self.managed and self.pysm_context:
            return initial_value_as_text(self.pysm_context, var_name, default)

        return default

    # ==============================================================================
    # SAVE
    # ==============================================================================
    def save(self, var_name: str, value: Any, var_type: Optional[str] = None):
        if self.managed and self.pysm_context:
            try:
                write_context_value(self.pysm_context, var_name, value, var_type=var_type)
            except Exception as e:
                logger.critical(f"Ошибка сохранения: {e}")
                sys.exit(1)
        else:
            logger.info(f"[LOCAL] {var_name} = {value}")

    # ==============================================================================
    # PIPELINE
    # ==============================================================================
    def process(self, raw_value: str, var_name: str, value_type: str):
        is_valid, err = self.validate(raw_value)
        if not is_valid:
            raise ValueError(err)

        final_value = self.convert(raw_value, value_type)

        self.save(var_name, final_value)

        return final_value
