"""Provider-neutral building blocks for AI-assisted PySM workflows.

This package contains no Qt dependencies. Import dialogs from
:mod:`pysm_lib.gui.ai` when a script needs the ready-made manual JSON UI.
"""

from .manual_json import (
    AiJsonDecodeError,
    AiJsonError,
    AiJsonRequest,
    ExpectedJsonType,
    PromptTemplateError,
    ResponseValidator,
    extract_json_object,
    extract_json_value,
    format_prompt_value,
    get_json_array,
    load_prompt_template,
    parse_and_validate_ai_response,
    render_prompt_template,
    require_json_object,
    require_json_object_item,
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
