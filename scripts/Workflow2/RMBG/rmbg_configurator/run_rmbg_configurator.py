#!/usr/bin/env python3
"""Configure the RMBG processing profile and save it to the PySM context."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SUBSYSTEM_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
for import_path in (PROJECT_ROOT, SUBSYSTEM_ROOT, SCRIPT_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

try:
    from pysm_lib import pysm_context, theme_api
    from pysm_lib.pysm_context import ConfigResolver

    IS_MANAGED_RUN = getattr(pysm_context, "_context_file_path", None) is not None
except ImportError:
    pysm_context = None
    theme_api = None
    ConfigResolver = None
    IS_MANAGED_RUN = False

from _common.context_config import (
    DEFAULT_CONFIG_VAR,
    load_context_settings,
    save_context_settings,
)
from _common.manifests import load_upstream_lock
from _common.model_registry import create_model_registry


def get_config() -> argparse.Namespace:
    """Define arguments and return the fully resolved script configuration."""

    parser = argparse.ArgumentParser(
        description="Настраивает RMBG-профиль и сохраняет его в контекст PySM."
    )
    parser.add_argument(
        "--config_var",
        default=DEFAULT_CONFIG_VAR,
        help="Имя JSON-переменной контекста с настройками RMBG.",
    )
    parser.add_argument(
        "--background_dir",
        default="",
        help=(
            "Папка изображений, из которой в GUI выбирается единый фон для "
            "пакетной обработки."
        ),
    )
    if ConfigResolver is not None:
        return ConfigResolver(parser).resolve_all()
    return parser.parse_args()


def main() -> int:
    config = get_config()
    if not IS_MANAGED_RUN or pysm_context is None:
        print(
            "RMBG Configurator необходимо запускать внутри PySM, "
            "чтобы сохранить настройки в контекст.",
            file=sys.stderr,
        )
        return 2

    try:
        from PySide6.QtWidgets import QApplication, QDialog

        from _lib.dialog import RmbgConfiguratorDialog
        from _lib.window_state import RmbgWindowStateStore

        settings = load_context_settings(
            pysm_context,
            config.config_var,
            use_defaults_if_missing=True,
        )
        upstream = load_upstream_lock()
        upstream_label = (
            f"PySM schema v1 • ComfyUI-RMBG {upstream['release_version']} "
            f"• {upstream['commit'][:12]} • {upstream['source_status']}"
        )

        app = QApplication.instance() or QApplication(sys.argv)
        if theme_api is not None:
            theme_api.apply_theme_to_app(app)
        window_state_store = RmbgWindowStateStore(pysm_context)
        dialog = RmbgConfiguratorDialog(
            settings,
            create_model_registry(),
            upstream_label=upstream_label,
            background_dir=(
                Path(config.background_dir) if config.background_dir else None
            ),
            test_root=PROJECT_ROOT.parents[1] / "tmp" / "Masks",
            window_state_store=window_state_store,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            print("Настройки RMBG не изменены.")
            return 0

        updated = dialog.accepted_settings

        save_context_settings(
            pysm_context,
            updated,
            config.config_var,
            commit=True,
        )
        summary = {
            "config_var": config.config_var,
            "profile_name": updated.profile_name,
            "model": updated.resolved_model_name().value,
            "model_dir": updated.model.model_dir,
            "background_dir": config.background_dir or None,
            "background_image": updated.output.background_image or None,
            "config_hash": updated.stable_hash(),
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"Не удалось сохранить настройки RMBG: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
