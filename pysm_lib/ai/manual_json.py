"""Core helpers for manual AI prompt and JSON-response workflows.

The module deliberately does not call an AI service.  It builds a prompt from
an editable template, extracts JSON from a manually returned answer, and
delegates domain validation to the caller.  This keeps credentials, provider
selection, and business rules out of the shared PySM API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Callable, Generic, Mapping, TypeVar


T = TypeVar("T")
ResponseValidator = Callable[[Any], T]
ExpectedJsonType = type[Any] | tuple[type[Any], ...] | None

_TEMPLATE_TOKEN_PATTERN = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


class AiJsonError(ValueError):
    """Base error for the manual AI JSON workflow."""


class PromptTemplateError(AiJsonError):
    """The prompt template cannot be rendered safely."""


class AiJsonDecodeError(AiJsonError):
    """A manually supplied AI answer does not contain the expected JSON."""


def format_prompt_value(value: Any) -> str:
    """Render one template value without losing Unicode JSON data.

    Plain strings are inserted verbatim so that user-entered source text stays
    readable.  All other values are encoded as indented JSON, which is useful
    for reference lists, dictionaries, and response examples.
    """

    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, indent=2)
    except (TypeError, ValueError) as exc:
        raise PromptTemplateError(
            f"Не удалось преобразовать значение шаблона в JSON: {exc}"
        ) from exc


def render_prompt_template(template: str, values: Mapping[str, Any]) -> str:
    """Replace ``{{TOKEN}}`` placeholders using strings or JSON values.

    Missing values are reported before a prompt can be copied.  This prevents a
    request with a literal unresolved placeholder from reaching an AI service.
    """

    if not isinstance(template, str) or not template.strip():
        raise PromptTemplateError("Шаблон AI-промпта должен быть непустым текстом.")

    missing_tokens: set[str] = set()

    def replace_token(match: re.Match[str]) -> str:
        token = match.group(1)
        if token not in values:
            missing_tokens.add(token)
            return match.group(0)
        return format_prompt_value(values[token])

    rendered = _TEMPLATE_TOKEN_PATTERN.sub(replace_token, template)
    if missing_tokens:
        missing = ", ".join(sorted(missing_tokens))
        raise PromptTemplateError(
            f"Для шаблона AI-промпта не переданы значения: {missing}."
        )
    return rendered


def load_prompt_template(
    path: str | Path,
    *,
    default_text: str | None = None,
    create_default: bool = False,
    encoding: str = "utf-8",
) -> str:
    """Load an editable prompt template, optionally creating a default one.

    ``default_text`` makes a script self-contained on the first run.  When it
    is omitted, a missing template is an explicit error rather than a silent
    empty prompt.
    """

    template_path = Path(path)
    if template_path.is_file():
        try:
            return template_path.read_text(encoding=encoding)
        except (OSError, UnicodeError) as exc:
            raise PromptTemplateError(
                f"Не удалось прочитать шаблон AI-промпта {template_path}: {exc}"
            ) from exc

    if template_path.exists():
        raise PromptTemplateError(
            f"Путь шаблона AI-промпта не является файлом: {template_path}."
        )
    if default_text is None:
        raise FileNotFoundError(f"Шаблон AI-промпта не найден: {template_path}")
    if not create_default:
        return default_text

    try:
        template_path.parent.mkdir(parents=True, exist_ok=True)
        template_path.write_text(default_text, encoding=encoding)
    except (OSError, UnicodeError) as exc:
        raise PromptTemplateError(
            f"Не удалось создать шаблон AI-промпта {template_path}: {exc}"
        ) from exc
    return default_text


def _type_name(value: Any) -> str:
    names = {
        dict: "JSON-объект",
        list: "JSON-массив",
        str: "JSON-строка",
        int: "JSON-число",
        float: "JSON-число",
        bool: "JSON-логическое значение",
        type(None): "JSON null",
    }
    return names.get(type(value), type(value).__name__)


def _matches_expected_type(value: Any, expected_type: ExpectedJsonType) -> bool:
    return expected_type is None or isinstance(value, expected_type)


def _expected_type_name(expected_type: type[Any]) -> str:
    names = {
        dict: "JSON-объект",
        list: "JSON-массив",
        str: "JSON-строка",
        int: "JSON-число",
        float: "JSON-число",
        bool: "JSON-логическое значение",
        type(None): "JSON null",
    }
    return names.get(expected_type, expected_type.__name__)


def extract_json_value(
    text: str,
    *,
    expected_type: ExpectedJsonType = None,
) -> Any:
    """Extract one JSON value from an AI answer or Markdown code fence.

    A full JSON response is preferred.  Otherwise the function scans JSON
    object and array starts with :class:`json.JSONDecoder`, so explanatory text
    and fenced Markdown around the JSON do not make the import fail.
    """

    if not isinstance(text, str) or not text.strip():
        raise AiJsonDecodeError("Вставьте непустой JSON-ответ AI.")

    source = text.lstrip("\ufeff")
    decoder = json.JSONDecoder()
    decoded_values: list[Any] = []

    try:
        full_value = json.loads(source)
    except json.JSONDecodeError:
        full_value = None
    else:
        decoded_values.append(full_value)
        if _matches_expected_type(full_value, expected_type):
            return full_value

    for index, character in enumerate(source):
        if character not in "{[":
            continue
        try:
            candidate, _ = decoder.raw_decode(source[index:])
        except json.JSONDecodeError:
            continue
        decoded_values.append(candidate)
        if _matches_expected_type(candidate, expected_type):
            return candidate

    if decoded_values and expected_type is not None:
        actual = _type_name(decoded_values[0])
        expected = " или ".join(
            _expected_type_name(candidate_type)
            for candidate_type in (
                expected_type if isinstance(expected_type, tuple) else (expected_type,)
            )
        )
        raise AiJsonDecodeError(f"Найден {actual}, но ожидался {expected}.")
    raise AiJsonDecodeError("В ответе AI не найден валидный JSON.")


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract a JSON object, the usual shape of a structured AI response."""

    payload = extract_json_value(text, expected_type=dict)
    return require_json_object(payload)


def parse_and_validate_ai_response(
    text: str,
    validator: ResponseValidator[T],
    *,
    expected_type: ExpectedJsonType = dict,
) -> T:
    """Extract JSON and pass it to a caller-owned domain validator."""

    if not callable(validator):
        raise TypeError("validator должен быть вызываемой функцией.")
    payload = extract_json_value(text, expected_type=expected_type)
    return validator(payload)


def require_json_object(payload: Any, *, subject: str = "Ответ AI") -> dict[str, Any]:
    """Return a JSON object or raise one consistent validation error."""

    if not isinstance(payload, dict):
        raise AiJsonError(f"{subject} должен быть JSON-объектом.")
    return payload


def get_json_array(
    payload: Mapping[str, Any],
    field_name: str,
    *,
    required: bool = False,
) -> list[Any]:
    """Read an array response field, preserving optional empty-field behavior."""

    if field_name not in payload:
        if required:
            raise AiJsonError(f"В ответе AI отсутствует обязательное поле {field_name}.")
        return []
    value = payload[field_name]
    if not isinstance(value, list):
        raise AiJsonError(f"Поле {field_name} должно быть JSON-массивом.")
    return value


def require_json_object_item(
    value: Any,
    *,
    field_name: str,
    index: int,
) -> dict[str, Any]:
    """Validate an object item inside a named response array."""

    if not isinstance(value, dict):
        raise AiJsonError(f"{field_name}[{index}] должен быть JSON-объектом.")
    return value


@dataclass(frozen=True)
class AiJsonRequest(Generic[T]):
    """Configuration for one reusable manual AI JSON interaction.

    ``response_validator`` is intentionally application-owned.  It must check
    IDs, number formats, permissions, or any other domain contract before the
    caller applies the returned value to its data model.
    """

    prompt_template: str
    response_validator: ResponseValidator[T]
    prompt_values: Mapping[str, Any] = field(default_factory=dict)
    raw_text_token: str | None = "RAW_TEXT"
    raw_text_label: str = "Неструктурированный исходный текст:"
    raw_text_placeholder: str = ""
    raw_text_required: bool = True
    response_expected_type: ExpectedJsonType = dict
    title: str = "AI JSON"
    response_label: str = "JSON-ответ AI:"
    response_file_filter: str = "JSON (*.json);;Все файлы (*)"
    success_message: Callable[[T], str] | None = None
    show_success_message: bool = True

    def build_prompt(self, raw_text: str = "") -> str:
        """Build the final prompt without mutating the supplied context."""

        values = dict(self.prompt_values)
        if self.raw_text_token is not None:
            if self.raw_text_required and not raw_text.strip():
                raise PromptTemplateError("Введите исходный текст для AI-промпта.")
            values[self.raw_text_token] = raw_text
        return render_prompt_template(self.prompt_template, values)

    def validate_response(self, text: str) -> T:
        """Extract and domain-validate a manually supplied AI response."""

        return parse_and_validate_ai_response(
            text,
            self.response_validator,
            expected_type=self.response_expected_type,
        )


__all__ = [
    "AiJsonDecodeError",
    "AiJsonError",
    "AiJsonRequest",
    "ExpectedJsonType",
    "PromptTemplateError",
    "ResponseValidator",
    "extract_json_object",
    "extract_json_value",
    "format_prompt_value",
    "get_json_array",
    "load_prompt_template",
    "parse_and_validate_ai_response",
    "render_prompt_template",
    "require_json_object",
    "require_json_object_item",
]
