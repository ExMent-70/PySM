"""Диагностика прямого API TensorRT и его регистрации в ONNX Runtime."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts_utility.SYSTEM._common.gpu_doctor_report import (  # noqa: E402
    DiagnosticReport,
    build_help_parser,
    compact_python_version,
)


def build_parser():
    """Создаёт CLI диагностического скрипта."""
    return build_help_parser(
        "Проверить прямой API TensorRT и провайдер TensorRT в ONNX Runtime."
    )


def check_direct_api(report: DiagnosticReport) -> Any | None:
    """Импортирует Python API TensorRT и возвращает модуль при успехе."""
    report.section(2, "Прямой Python API TensorRT")
    try:
        import tensorrt as trt
    except ImportError:
        report.warning("Python-пакет tensorrt не установлен в текущем окружении.")
        report.line(
            "Это не исключает работу TensorRT через нативный провайдер ONNX Runtime.",
            indent=1,
        )
        return None
    except Exception as error:
        report.exception("Python-пакет tensorrt найден, но не загрузился.", error)
        return None

    report.detail("Версия TensorRT", getattr(trt, "__version__", "не указана"))
    try:
        trt.Logger(trt.Logger.WARNING)
    except Exception as error:
        report.exception("Не удалось создать логгер TensorRT.", error)
        return None

    report.success("Прямой Python API TensorRT импортирован и инициализирован.")
    return trt


def check_onnx_runtime(report: DiagnosticReport) -> tuple[bool, bool]:
    """Проверяет регистрацию TensorRT и CUDA в текущем ONNX Runtime."""
    report.section(3, "Провайдеры ONNX Runtime")
    try:
        import onnxruntime as ort
    except ImportError:
        report.warning("ONNX Runtime не установлен в текущем окружении.")
        return False, False
    except Exception as error:
        report.exception("ONNX Runtime найден, но не загрузился.", error)
        return False, False

    report.detail("Версия ONNX Runtime", ort.__version__)
    try:
        providers = ort.get_available_providers()
    except Exception as error:
        report.exception("Не удалось получить список провайдеров ONNX Runtime.", error)
        return False, False

    report.detail("Доступные провайдеры", ", ".join(providers) or "не обнаружены")
    tensorrt_registered = "TensorrtExecutionProvider" in providers
    cuda_registered = "CUDAExecutionProvider" in providers

    if tensorrt_registered:
        report.success("TensorrtExecutionProvider зарегистрирован в ONNX Runtime.")
        report.info(
            "Регистрация провайдера подтверждает его доступность, но не гарантирует обработку каждой модели."
        )
    else:
        report.warning("TensorrtExecutionProvider отсутствует в ONNX Runtime.")

    if cuda_registered:
        report.success("CUDAExecutionProvider зарегистрирован как возможный резервный провайдер.")
    else:
        report.warning("CUDAExecutionProvider отсутствует; GPU-резерв для неподдержанных узлов недоступен.")
    return tensorrt_registered, cuda_registered


def build_test_engine(report: DiagnosticReport, trt: Any | None) -> bool | None:
    """Собирает минимальный движок в памяти через прямой API TensorRT."""
    report.section(4, "Сборка минимального движка TensorRT")
    if trt is None:
        report.info("Тест пропущен: прямой Python API TensorRT недоступен.")
        return None

    try:
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(explicit_batch)
        config = builder.create_builder_config()
        input_tensor = network.add_input("input", trt.float32, (1, 3, 16, 16))
        if input_tensor is None:
            raise RuntimeError("TensorRT не создал входной тензор")
        identity = network.add_identity(input_tensor)
        if identity is None:
            raise RuntimeError("TensorRT не создал тестовый слой")
        network.mark_output(identity.get_output(0))

        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise RuntimeError("сборщик TensorRT вернул пустой результат")

        engine_size = len(bytes(serialized_engine))
        report.detail("Размер тестового движка", f"{engine_size} байт")
        report.success("Минимальный движок TensorRT успешно собран в памяти.")
        return True
    except Exception as error:
        report.exception("Не удалось собрать минимальный движок TensorRT.", error)
        return False


def main() -> int:
    """Запускает независимые проверки TensorRT и формирует заключение."""
    build_parser().parse_args()
    report = DiagnosticReport("Диагностика TensorRT")
    report.begin()

    report.section(1, "Окружение Python")
    report.detail("Версия Python", compact_python_version(sys.version))
    report.detail("Интерпретатор", sys.executable)

    trt = check_direct_api(report)
    trt_registered, cuda_registered = check_onnx_runtime(report)
    engine_built = build_test_engine(report, trt)

    if engine_built and trt_registered:
        conclusion = "Прямой API TensorRT работает, а провайдер TensorRT зарегистрирован в ONNX Runtime."
    elif trt_registered and cuda_registered:
        conclusion = (
            "ONNX Runtime видит провайдеры TensorRT и CUDA; проверьте рабочую модель в целевом сценарии."
        )
    elif engine_built:
        conclusion = "Прямой API TensorRT работает, но интеграция TensorRT с ONNX Runtime не подтверждена."
    elif trt_registered:
        conclusion = "Провайдер TensorRT зарегистрирован, но резервный CUDA-провайдер не обнаружен."
    else:
        conclusion = "Работоспособность TensorRT не подтверждена."
    report.finish(conclusion)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
