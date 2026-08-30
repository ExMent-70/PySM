# pysm_lib/gui/tooltip_generator.py

import re
from typing import Callable, Optional

from ..models import ScriptInfoModel, ScriptSetEntryModel
from ..locale_manager import LocaleManager
from ..theme_manager import ThemeManager
from .gui_utils import resolve_themed_text

TOOLTIP_ARGUMENT_VALUE_MAX_LENGTH = 70


def _truncate_tooltip_argument_value(value: object) -> str:
    """Сокращает только многострочные значения для подсказки."""
    value_text = str(value)
    if "\n" not in value_text and "\r" not in value_text:
        return value_text

    single_line_value = re.sub(r"\s+", " ", value_text).strip()
    visible_length = TOOLTIP_ARGUMENT_VALUE_MAX_LENGTH - 3
    preview = single_line_value[:visible_length].rstrip()
    return f"{preview}..."


def _escape_tooltip_argument_value(value: str) -> str:
    """Экранирует символы, значимые для HTML-подсказки."""
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _format_tooltip_argument_value(
    script_info: ScriptInfoModel,
    argument_name: str,
    value: object,
    instance_name_resolver: Optional[Callable[[str], Optional[str]]],
) -> str:
    """Форматирует значение аргумента для HTML-подсказки."""
    argument_meta = (script_info.command_line_args_meta or {}).get(argument_name)
    if argument_meta and argument_meta.type == "instance":
        raw_instance_ids = (
            value if isinstance(value, list) else str(value).split(",")
        )
        instance_ids = [str(instance_id).strip() for instance_id in raw_instance_ids]
        instance_ids = [instance_id for instance_id in instance_ids if instance_id]

        if instance_ids:
            instance_lines = []
            for instance_id in instance_ids:
                display_name = (
                    instance_name_resolver(instance_id)
                    if instance_name_resolver
                    else None
                )
                display_text = (
                    f"{display_name} ({instance_id})" if display_name else instance_id
                )
                escaped_text = _escape_tooltip_argument_value(
                    _truncate_tooltip_argument_value(display_text)
                )
                instance_lines.append(
                    f'<span style="{{theme.tooltip_arg_value}}">\'{escaped_text}\'</span>'
                )
            return "<br>" + "<br>".join(instance_lines)

    escaped_value = _escape_tooltip_argument_value(
        _truncate_tooltip_argument_value(value)
    )
    return f'<span style="{{theme.tooltip_arg_value}}">\'{escaped_value}\'</span>'


# --- 1. БЛОК: Функция _generate_header_script_html (ЛОГИКА НЕ ИЗМЕНЕНА) ---
def _generate_header_script_html(
    script_info: ScriptInfoModel, locale_manager: LocaleManager
) -> str:
    """Генерирует заголовок с HTML-информацией о скрипте."""
    parts = []
    parts.append(
        f"<div>{locale_manager.get('tooltips.script.label_format_bold', label=locale_manager.get('tooltips.script.label_script'), value=script_info.name + '.py')}</div>"
    )
    parts.append(
        f"<div>{locale_manager.get('tooltips.script.label_format_bold', label=locale_manager.get('tooltips.script.label_path'), value=script_info.folder_abs_path)}</div>"
    )
    return "".join(parts)


# --- 2. БЛОК: Функция _generate_base_script_html (ИЗМЕНЕНА) ---
def _generate_base_script_html(
    script_info: ScriptInfoModel, locale_manager: LocaleManager
) -> str:
    """Генерирует основную HTML-информацию о скрипте."""
    parts = []
    if script_info.is_raw:
        parts.append(
            f"<div>{locale_manager.get('tooltips.script.label_format_bold_orange', label=locale_manager.get('tooltips.script.label_warning'), value=locale_manager.get('tooltips.script.raw_text'))}</div>"
        )
    elif not script_info.passport_valid:
        error_text = script_info.passport_error or locale_manager.get(
            "tooltips.script.invalid_passport_text"
        )
        parts.append(
            f"<div>{locale_manager.get('tooltips.script.label_format_bold_red', label=locale_manager.get('tooltips.script.label_error'), value=error_text)}</div>"
        )
    else:
        if script_info.description:
            # Описание не обрабатываем здесь, так как оно может содержать пользовательский HTML
            desc_html = script_info.description.replace("\n", "<br>")
            parts.append(
                f"<div style='margin-top:5px;'><b>{locale_manager.get('tooltips.script.label_description')}</b><div style='padding-left: 10px;'>{desc_html}</div></div>"
            )

    if script_info.command_line_args_meta:
        # КОММЕНТАРИЙ: Здесь мы вставляем плейсхолдер.
        # Он будет заменен на реальный CSS-стиль позже.
        arg_parts = [
            f"<div style='margin-top:10px; {{theme.tooltip_script_args_block}}'><b>{locale_manager.get('tooltips.instance.params_header')}</b>"
        ]
        arg_list_parts = []
        for name, meta in script_info.command_line_args_meta.items():
            desc = _truncate_tooltip_argument_value(meta.description or "...")
            arg_list_parts.append(
                f"<div style='white-space: nowrap;'><b>--{name}:</b> {desc}</div>"
            )

        arg_parts.append(
            f"<div style='padding-left: 10px;'>"
            f"{''.join(arg_list_parts)}</div></div>"
        )
        parts.extend(arg_parts)

    return "".join(parts)


# --- 3. БЛОК: Функция _generate_end_script_html (ЛОГИКА НЕ ИЗМЕНЕНА) ---
def _generate_end_script_html(
    script_info: ScriptInfoModel, locale_manager: LocaleManager
) -> str:
    """Генерирует основную HTML-информацию о скрипте без специфичных для подсказок элементов."""
    final_parts = []

    if script_info.author and script_info.author != locale_manager.get(
        "models.script_info.default_author"
    ):
        final_parts.append(
            f"<div style='margin-top:10px;'>{locale_manager.get('tooltips.script.label_format_bold', label=locale_manager.get('tooltips.script.label_author'), value=script_info.author)}</div>"
        )
    if script_info.version:
        final_parts.append(
            f"<div style='margin-top:5px;'>{locale_manager.get('tooltips.script.label_format_bold', label=locale_manager.get('tooltips.script.label_version'), value=script_info.version)}</div>"
        )
    return "".join(final_parts)

def generate_script_tooltip_html(
    script_info: ScriptInfoModel, 
    locale_manager: LocaleManager,
    theme_manager: ThemeManager # <--- ИЗМЕНЕННЫЙ АРГУМЕНТ
) -> str:
    """Генерирует HTML-разметку для всплывающей подсказки скрипта."""
    if not script_info:
        return locale_manager.get("tooltips.script.no_info_available")

    header_html = _generate_header_script_html(script_info, locale_manager)
    base_html = _generate_base_script_html(script_info, locale_manager)
    end_html = _generate_end_script_html(script_info, locale_manager)

    final_html = f"""
        {header_html}
        {base_html}
        {end_html}        
        <hr>
        <b>{locale_manager.get("tooltips.script.double_click_hint")}</b>
    """
    # КОММЕНТАРИЙ: Вызываем новую утилиту, передавая ей config_manager
    return resolve_themed_text(final_html, theme_manager)


def generate_instance_tooltip_html(
    script_info: ScriptInfoModel,
    instance_entry: ScriptSetEntryModel,
    locale_manager: LocaleManager,
    theme_manager: ThemeManager,
    instance_name_resolver: Optional[Callable[[str], Optional[str]]] = None,
) -> str:
    """Генерирует HTML-разметку для всплывающей подсказки экземпляра скрипта."""
    if not script_info:
        return f"<b>{locale_manager.get('tooltips.instance.label_instance_id')}</b> {instance_entry.instance_id}<br><b style='color:red;'>{locale_manager.get('tooltips.script.label_error')}</b> {locale_manager.get('tooltips.instance.script_not_found', id=instance_entry.id)}"

    header_html = _generate_header_script_html(script_info, locale_manager)
    end_html = _generate_end_script_html(script_info, locale_manager)

    overridden_desc_html = ""
    if instance_entry.description:
        # Описание не обрабатываем здесь, так как оно может содержать пользовательский HTML
        desc_html = instance_entry.description.replace("\n", "<br>")
        overridden_desc_html = f"<div style='margin-top:5px;'><b>{locale_manager.get('tooltips.script.label_description')}</b><div style='padding-left: 10px;'>{desc_html}</div></div>"
    
    overridden_params_html = ""
    active_args = {
        k: v for k, v in instance_entry.command_line_args.items() if v.enabled
    }
    if active_args:
        param_lines = []
        for name, entry_value in active_args.items():
            # КОММЕНТАРИЙ: Вставляем плейсхолдер для цвета значения аргумента
            display_value = (
                _format_tooltip_argument_value(
                    script_info,
                    name,
                    entry_value.value,
                    instance_name_resolver,
                )
                if entry_value.value is not None
                else locale_manager.get("tooltips.instance.flag_present_text")
            )
            param_text = locale_manager.get(
                "tooltips.instance.param_format", name=name, value=display_value
            )
            param_lines.append(
                f"<div style='white-space: nowrap;'>{param_text}</div>"
            )

        # КОММЕНТАРИЙ: Вставляем плейсхолдер для фона всего блока
        overridden_params_html = f"""
        <div style='margin-top:10px; {{theme.tooltip_instance_args_block}}'>
            <b>{locale_manager.get("tooltips.instance.label_overridden_params")}</b>
            <div style='padding-left: 10px;'>{"".join(param_lines)}</div>
        </div>
        """

    final_html = f"""
    {header_html}
    <hr>
    <div><b>{locale_manager.get("tooltips.instance.label_instance_name")}</b> {instance_entry.name or script_info.name}</div>
    <div><b>{locale_manager.get("tooltips.instance.label_instance_id")}</b> {instance_entry.instance_id}</div>
    {overridden_desc_html}
    {overridden_params_html}
    <hr>
    <b>{locale_manager.get("tooltips.instance.double_click_hint")}</b>
    """
    
    return resolve_themed_text(final_html, theme_manager)
    
def generate_favorite_tooltip_html(
    script_info: ScriptInfoModel,
    instance_entry: ScriptSetEntryModel,
    locale_manager: LocaleManager,
    theme_manager: ThemeManager,
    instance_name_resolver: Optional[Callable[[str], Optional[str]]] = None,
) -> str:
    """Генерирует HTML-разметку для всплывающей подсказки экземпляра скрипта."""
    if not script_info:
        return f"<b>{locale_manager.get('tooltips.instance.label_instance_id')}</b> {instance_entry.instance_id}<br><b style='color:red;'>{locale_manager.get('tooltips.script.label_error')}</b> {locale_manager.get('tooltips.instance.script_not_found', id=instance_entry.id)}"

    #header_html = _generate_header_script_html(script_info, locale_manager)
    end_html = _generate_end_script_html(script_info, locale_manager)

    overridden_desc_html = ""
    if instance_entry.description:
        # Описание не обрабатываем здесь, так как оно может содержать пользовательский HTML
        desc_html = instance_entry.description.replace("\n", "<br>")
        overridden_desc_html = f"<div style='margin-top:5px;'><b>{locale_manager.get('tooltips.script.label_description')}</b><div style='padding-left: 10px;'>{desc_html}</div></div>"
    
    overridden_params_html = ""
    active_args = {
        k: v for k, v in instance_entry.command_line_args.items() if v.enabled
    }
    if active_args:
        param_lines = []
        for name, entry_value in active_args.items():
            # КОММЕНТАРИЙ: Вставляем плейсхолдер для цвета значения аргумента
            display_value = (
                _format_tooltip_argument_value(
                    script_info,
                    name,
                    entry_value.value,
                    instance_name_resolver,
                )
                if entry_value.value is not None
                else locale_manager.get("tooltips.instance.flag_present_text")
            )
            param_text = locale_manager.get(
                "tooltips.instance.param_format", name=name, value=display_value
            )
            param_lines.append(
                f"<div style='white-space: nowrap;'>{param_text}</div>"
            )

        # КОММЕНТАРИЙ: Вставляем плейсхолдер для фона всего блока
        overridden_params_html = f"""
        <div style='margin-top:10px; {{theme.tooltip_instance_args_block}}'>
            <b>{locale_manager.get("tooltips.instance.label_overridden_params")}</b>
            <div style='padding-left: 10px;'>{"".join(param_lines)}</div>
        </div>
        """

    final_html = f"""
    <div><b>{instance_entry.name or script_info.name}</b></div>
    {overridden_desc_html}
    {overridden_params_html}
    <hr>
    <b>{locale_manager.get("tooltips.instance.context_menu_hint")}</b>
    """
    
    return resolve_themed_text(final_html, theme_manager)
