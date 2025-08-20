# pysm_lib/gui/tooltip_generator.py

import re # <--- НОВЫЙ ИМПОРТ
from typing import Dict # <--- НОВЫЙ ИМПОРТ
from ..models import ScriptInfoModel, ScriptSetEntryModel
from ..locale_manager import LocaleManager
from ..theme_manager import ThemeManager
from .gui_utils import resolve_themed_text

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
            desc = meta.description or "..."
            arg_list_parts.append(f"<b>--{name}:</b> {desc}")

        arg_parts.append(f"<div style='padding-left: 10px;'>{'<br>'.join(arg_list_parts)}</div></div>")
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
    theme_manager: ThemeManager # <--- ИЗМЕНЕННЫЙ АРГУМЕНТ
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
            value_str = str(entry_value.value) if entry_value.value is not None else ""
            escaped_value = value_str.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            
            # КОММЕНТАРИЙ: Вставляем плейсхолдер для цвета значения аргумента
            display_value = (
                f"<span style=\"{{theme.tooltip_arg_value}}\">'{escaped_value}'</span>"
                if entry_value.value is not None
                else locale_manager.get("tooltips.instance.flag_present_text")
            )
            param_lines.append(
                locale_manager.get(
                    "tooltips.instance.param_format", name=name, value=display_value
                )
            )

        # КОММЕНТАРИЙ: Вставляем плейсхолдер для фона всего блока
        overridden_params_html = f"""
        <div style='margin-top:10px; {{theme.tooltip_instance_args_block}}'>
            <b>{locale_manager.get("tooltips.instance.label_overridden_params")}</b>
            <div style='padding-left: 10px;'>{"<br>".join(param_lines)}</div>
        </div>
        """

    final_html = f"""
    {header_html}
    <hr>
    <div><b>{locale_manager.get("tooltips.instance.label_instance_name")}</b> {instance_entry.name or script_info.name}</div>
    <div><b>{locale_manager.get("tooltips.instance.label_instance_id")}</b> {instance_entry.instance_id}</div>
    {overridden_desc_html}
    {overridden_params_html}
    {end_html}
    <hr>
    <b>{locale_manager.get("tooltips.instance.double_click_hint")}</b>
    """
    
    return resolve_themed_text(final_html, theme_manager)