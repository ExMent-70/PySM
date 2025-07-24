# pysm_lib/locale_manager.py
import json
import logging
import pathlib
from typing import Dict, Any

logger = logging.getLogger(f"PyScriptManager.{__name__}")


class LocaleManager:
    """
    Manages loading and retrieving localized strings for the application.
    """

    def __init__(self, language_code: str = "ru_RU"):
        self.locales_dir = pathlib.Path(__file__).parent / "locales"
        self.strings: Dict[str, Any] = {}
        self.language_code = language_code
        self._load_strings()

    def _load_strings(self):
        """Loads the strings from the corresponding JSON file."""
        file_path = self.locales_dir / f"{self.language_code}.json"
        self.strings = {}
        if not file_path.is_file():
            logger.error(
                f"Locale file not found: {file_path}. UI will use fallback keys."
            )
            # Попробовать загрузить en_US как запасной вариант
            file_path = self.locales_dir / "en_US.json"
            if not file_path.is_file():
                return

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                self.strings = json.load(f)
            logger.info(f"Successfully loaded locale file: {file_path}")
        except (json.JSONDecodeError, Exception) as e:
            logger.critical(
                f"Failed to load or parse locale file {file_path}: {e}", exc_info=True
            )

    # === БЛОК 1: Метод get (ИЗМЕНЕН) ===
    def get(self, key_string: str, **kwargs: Any) -> str:
        """
        Retrieves a string by its key.
        The key uses dot notation to access nested JSON objects, e.g., "main_window.title".
        If kwargs are provided, it formats the string with them.
        """
        # Параметр переименован с 'key' на 'key_string', чтобы избежать конфликта
        # с возможным kwargs['key'] при форматировании.
        keys = key_string.split(".")
        value = self.strings
        try:
            for k in keys:
                value = value[k]

            if isinstance(value, str):
                if kwargs:
                    return value.format(**kwargs)
                return value
            else:
                logger.warning(
                    f"Locale key '{key_string}' resolved to a non-string value. Type: {type(value)}"
                )
                return str(value)
        except KeyError:
            logger.warning(
                f"Locale key not found: '{key_string}'. Falling back to key name."
            )
            return key_string
        except Exception as e:
            logger.error(f"Error retrieving locale key '{key_string}': {e}")
            return key_string

    def switch_language(self, new_language_code: str):
        """Switches the current language and reloads the strings."""
        logger.info(
            f"Switching language from '{self.language_code}' to '{new_language_code}'."
        )
        self.language_code = new_language_code
        self._load_strings()

    def get_available_languages(self) -> Dict[str, str]:
        """
        Scans the locales directory and returns a dict of available languages.
        Returns: Dict[language_code, display_name]
        """
        languages = {}
        if not self.locales_dir.is_dir():
            return languages

        for file_path in self.locales_dir.glob("*.json"):
            lang_code = file_path.stem
            display_name = lang_code  # Fallback name
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if (
                        isinstance(data.get("_meta"), dict)
                        and "language_name" in data["_meta"]
                    ):
                        display_name = data["_meta"]["language_name"]
            except (json.JSONDecodeError, Exception):
                logger.warning(f"Could not read metadata from locale file: {file_path}")

            languages[lang_code] = display_name

        return languages
