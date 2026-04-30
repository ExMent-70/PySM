#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_cli_var_json_set.py
====================

Утилита для установки значения переменной контекста без GUI.

Теперь использует единый InputProcessor.
"""

# ==============================================================================
# 1. ИМПОРТЫ
# ==============================================================================
import argparse
import logging
import sys

try:
    from pysm_lib import pysm_context
    from pysm_lib.pysm_context import ConfigResolver
    from pysm_lib.input_processor import InputProcessor, VALIDATION_PRESETS
    IS_MANAGED_RUN = True
except ImportError:
    pysm_context = None
    ConfigResolver = None
    InputProcessor = None
    IS_MANAGED_RUN = False

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


# ==============================================================================
# 2. CONFIG
# ==============================================================================
def get_config():
    parser = argparse.ArgumentParser(
        description="Устанавливает значение переменной контекста (поддержка JSON и вложенности)."
    )

    parser.add_argument(
        "--set_var_name",
        type=str,
        required=True,
        help="Имя переменной (поддерживает dot-notation).",
    )

    parser.add_argument(
        "--set_var_value",
        type=str,
        default="",
        help="Значение (строка).",
    )

    parser.add_argument(
        "--set_value_type",
        type=str,
        default="auto",
        choices=["auto", "string", "int", "float", "bool", "json"],
    )

    parser.add_argument(
        "--set_valid_type",
        type=str,
        default="none",
        choices=["none", "custom"] + list(VALIDATION_PRESETS.keys()),
    )

    parser.add_argument("--set_custom_regexp", type=str)
    parser.add_argument("--set_custom_regexp_desc", type=str)

    if IS_MANAGED_RUN and ConfigResolver:
        return ConfigResolver(parser).resolve_all()

    return parser.parse_args()


# ==============================================================================
# 3. MAIN
# ==============================================================================
def main():
    config = get_config()

    processor = InputProcessor(config, pysm_context, IS_MANAGED_RUN)

    try:
        result = processor.process(
            raw_value=config.set_var_value,
            var_name=config.set_var_name,
            value_type=config.set_value_type,
        )

        logger.info(f"✅ <b>{config.set_var_name}</b> = <i>{result}</i>\n")

    except Exception as e:
        logger.error(f"❌ Ошибка: {e}")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()