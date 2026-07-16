"""Public compatibility facade for the visual JSON editor API.

The Qt implementation lives in :mod:`pysm_lib.gui.json_editor`. Keeping this
module small preserves the documented import used by PySM scripts.
"""

from .gui.json_editor import (
    JsonEditor,
    JsonEditorResult,
    JsonEditorStatus,
    JsonModel,
    NewItemDialog,
    Node,
    SchemaIndex,
    ValueDelegate,
    build_default_schema,
    create_json_editor,
    default_value_for_type,
    edit_json_variable,
    is_schema_variable,
    json_type_name,
    load_context_variable,
    load_schema,
    schema_var_name_for,
    show_json_editor_error,
    target_name_for_schema,
    validate_variable,
)

__all__ = [
    "JsonEditor",
    "JsonEditorResult",
    "JsonEditorStatus",
    "JsonModel",
    "NewItemDialog",
    "Node",
    "SchemaIndex",
    "ValueDelegate",
    "build_default_schema",
    "create_json_editor",
    "default_value_for_type",
    "edit_json_variable",
    "is_schema_variable",
    "json_type_name",
    "load_context_variable",
    "load_schema",
    "schema_var_name_for",
    "show_json_editor_error",
    "target_name_for_schema",
    "validate_variable",
]
