"""Сводная диагностика драйвера NVIDIA, GPU, CUDA и PyTorch."""

from __future__ import annotations

import json
import os
import platform
import shutil
import sys
import warnings
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts_utility.SYSTEM._common.gpu_doctor_report import (  # noqa: E402
    DiagnosticReport,
    build_help_parser,
    compact_python_version,
    format_bytes,
)


def build_parser():
    """Создаёт CLI диагностического скрипта."""
    return build_help_parser("Собрать сводный отчёт о драйвере NVIDIA, GPU, CUDA и PyTorch.")


def _text(value: Any) -> str:
    """Нормализует строки, которые NVML разных версий возвращает как str или bytes."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _optional_nvml(call: Any, *args: Any) -> Any | None:
    """Возвращает значение необязательного NVML-показателя или None."""
    try:
        return call(*args)
    except Exception:
        return None


def _find_portable_cuda() -> tuple[Path | None, str | None, str | None]:
    """Ищет portable CUDA PySM и читает её версию из version.json."""
    candidates: list[Path] = []
    env_path = os.environ.get("PYSM_CUDA_PATH")
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(PROJECT_ROOT.parent.parent / "ps_env" / "CUDA")

    executable_path = Path(sys.executable).resolve()
    if len(executable_path.parents) > 2:
        candidates.append(executable_path.parents[2] / "ps_env" / "CUDA")

    seen: set[Path] = set()
    for cuda_path in candidates:
        normalized_path = cuda_path.resolve()
        if normalized_path in seen:
            continue
        seen.add(normalized_path)

        version_file = normalized_path / "version.json"
        if not version_file.is_file():
            continue
        try:
            data = json.loads(version_file.read_text(encoding="utf-8"))
            version = data.get("cuda", {}).get("version")
            return normalized_path, str(version) if version else None, None
        except (OSError, TypeError, ValueError) as error:
            return normalized_path, None, f"не удалось прочитать {version_file}: {error}"
    return None, None, None


def report_environment(report: DiagnosticReport) -> None:
    """Показывает Python, ОС и обнаруженные системные инструменты CUDA."""
    report.section(1, "Окружение Python и CUDA")
    report.detail("Версия Python", compact_python_version(sys.version))
    report.detail("Интерпретатор", sys.executable)
    report.detail("Операционная система", platform.platform())
    report.detail("CUDA_PATH", os.environ.get("CUDA_PATH") or "не задан")
    report.detail("PYSM_CUDA_PATH", os.environ.get("PYSM_CUDA_PATH") or "не задан")

    portable_path, portable_version, portable_error = _find_portable_cuda()
    if portable_path is None:
        report.detail("Portable CUDA PySM", "не найдена")
    else:
        report.detail("Portable CUDA PySM", portable_path)
        report.detail("Версия portable CUDA", portable_version or "не указана")
        portable_nvcc = portable_path / "bin" / "nvcc.exe"
        report.detail(
            "Компилятор portable CUDA",
            portable_nvcc if portable_nvcc.is_file() else "не найден",
        )
    if portable_error:
        report.warning(f"Сведения о portable CUDA неполны: {portable_error}")

    report.detail("Утилита nvidia-smi", shutil.which("nvidia-smi") or "не найдена в PATH")
    report.detail("Системный nvcc из PATH", shutil.which("nvcc") or "не найден")


def report_nvml(report: DiagnosticReport) -> tuple[bool, int]:
    """Собирает сведения о драйвере и физических GPU через NVML."""
    report.section(2, "Драйвер NVIDIA и физические устройства")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            import pynvml
    except ImportError:
        report.warning("Модуль NVML для Python не установлен; системная часть отчёта пропущена.")
        report.line("Для этого раздела нужен пакет nvidia-ml-py.", indent=1)
        return False, 0

    initialized = False
    try:
        pynvml.nvmlInit()
        initialized = True
        driver_version = _text(pynvml.nvmlSystemGetDriverVersion())
        report.detail("Версия драйвера NVIDIA", driver_version)

        cuda_version = _optional_nvml(pynvml.nvmlSystemGetCudaDriverVersion)
        if cuda_version is None:
            report.detail("CUDA, поддерживаемая драйвером", "не удалось определить")
        else:
            report.detail(
                "CUDA, поддерживаемая драйвером",
                f"{cuda_version // 1000}.{(cuda_version % 1000) // 10}",
            )

        device_count = pynvml.nvmlDeviceGetCount()
        report.detail("Количество GPU", device_count)
        if device_count == 0:
            report.warning("NVML инициализирована, но совместимые GPU не обнаружены.")
            return True, 0

        for index in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            name = _text(pynvml.nvmlDeviceGetName(handle))
            report.info(f"GPU {index}: {name}")

            memory = _optional_nvml(pynvml.nvmlDeviceGetMemoryInfo, handle)
            if memory is not None:
                report.detail("Память всего", format_bytes(memory.total), indent=1)
                report.detail("Память занято", format_bytes(memory.used), indent=1)
                report.detail("Память свободно", format_bytes(memory.free), indent=1)

            capability = _optional_nvml(pynvml.nvmlDeviceGetCudaComputeCapability, handle)
            if capability is not None:
                report.detail("Вычислительная способность", f"{capability[0]}.{capability[1]}", indent=1)

            utilization = _optional_nvml(pynvml.nvmlDeviceGetUtilizationRates, handle)
            if utilization is not None:
                report.detail("Загрузка GPU", f"{utilization.gpu}%", indent=1)

            temperature = _optional_nvml(
                pynvml.nvmlDeviceGetTemperature,
                handle,
                pynvml.NVML_TEMPERATURE_GPU,
            )
            if temperature is not None:
                report.detail("Температура", f"{temperature} °C", indent=1)

        report.success("Драйвер NVIDIA и физические GPU доступны через NVML.")
        return True, device_count
    except Exception as error:
        report.exception("Не удалось получить сведения через NVML.", error)
        return False, 0
    finally:
        if initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


def report_pytorch(report: DiagnosticReport) -> tuple[bool, int]:
    """Показывает состояние CUDA с точки зрения текущей установки PyTorch."""
    report.section(3, "PyTorch, CUDA Runtime и cuDNN")
    try:
        import torch
    except ImportError:
        report.warning("PyTorch не установлен; библиотечная часть отчёта пропущена.")
        return False, 0
    except Exception as error:
        report.exception("PyTorch установлен, но его импорт завершился ошибкой.", error)
        return False, 0

    report.detail("Версия PyTorch", torch.__version__)
    report.detail("CUDA сборки PyTorch", torch.version.cuda or "сборка только для CPU")
    report.detail("Версия cuDNN", torch.backends.cudnn.version() or "недоступна")

    try:
        cuda_available = torch.cuda.is_available()
        device_count = torch.cuda.device_count()
    except Exception as error:
        report.exception("PyTorch не смог опросить подсистему CUDA.", error)
        return False, 0

    report.detail("CUDA доступна PyTorch", "да" if cuda_available else "нет")
    report.detail("Количество GPU в PyTorch", device_count)

    for index in range(device_count):
        try:
            properties = torch.cuda.get_device_properties(index)
            report.info(f"Устройство PyTorch {index}: {properties.name}")
            report.detail("Память всего", format_bytes(properties.total_memory), indent=1)
            report.detail(
                "Вычислительная способность",
                f"{properties.major}.{properties.minor}",
                indent=1,
            )
        except Exception as error:
            report.exception(f"Не удалось получить свойства GPU {index} через PyTorch.", error)

    if cuda_available and device_count > 0:
        report.success("PyTorch обнаружил CUDA и хотя бы одно устройство GPU.")
        return True, device_count

    report.warning("Текущая установка PyTorch не готова к вычислениям на CUDA.")
    return False, device_count


def main() -> int:
    """Запускает все независимые разделы диагностики."""
    build_parser().parse_args()
    report = DiagnosticReport("Сводная диагностика GPU")
    report.begin()

    report_environment(report)
    nvml_ready, nvml_devices = report_nvml(report)
    torch_ready, torch_devices = report_pytorch(report)

    report.section(4, "Сопоставление результатов")
    if nvml_devices and torch_devices and nvml_devices != torch_devices:
        report.warning(
            "NVML и PyTorch обнаружили разное количество GPU; проверьте ограничения видимости устройств."
        )
    elif nvml_devices and torch_devices:
        report.success("NVML и PyTorch обнаружили одинаковое количество GPU.")
    else:
        report.info("Сопоставление количества GPU невозможно из-за неполных данных.")

    if nvml_ready and torch_ready:
        conclusion = "Драйвер NVIDIA и PyTorch видят GPU; базовая конфигурация выглядит работоспособной."
    elif nvml_ready:
        conclusion = "Драйвер видит GPU, но PyTorch не подтвердил готовность CUDA."
    elif torch_ready:
        conclusion = "PyTorch использует GPU, но системные сведения NVML недоступны."
    else:
        conclusion = "Готовность GPU не подтверждена; изучите ошибки и предупреждения выше."
    report.finish(conclusion)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
