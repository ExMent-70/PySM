"""Практическая диагностика PyTorch и вычислений CUDA на каждом GPU."""

from __future__ import annotations

import sys
from pathlib import Path


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
    return build_help_parser(
        "Проверить обнаружение GPU и выполнить тестовую операцию PyTorch на CUDA."
    )


def main() -> int:
    """Запускает импорт, инвентаризацию и практический тест CUDA."""
    build_parser().parse_args()
    report = DiagnosticReport("Диагностика PyTorch и CUDA")
    report.begin()

    report.section(1, "Окружение Python")
    report.detail("Версия Python", compact_python_version(sys.version))
    report.detail("Интерпретатор", sys.executable)

    report.section(2, "Импорт и версия PyTorch")
    try:
        import torch
    except ImportError:
        report.error("PyTorch не установлен в текущем окружении Python.")
        report.finish("Диагностика остановлена: PyTorch недоступен.")
        return 0
    except Exception as error:
        report.exception("PyTorch установлен, но его импорт завершился ошибкой.", error)
        report.finish("Диагностика остановлена: PyTorch не удалось загрузить.")
        return 0

    report.detail("Версия PyTorch", torch.__version__)
    report.detail("CUDA сборки PyTorch", torch.version.cuda or "сборка только для CPU")
    report.detail("Версия cuDNN", torch.backends.cudnn.version() or "недоступна")
    report.success("PyTorch успешно импортирован.")

    report.section(3, "Доступность CUDA и устройства")
    try:
        cuda_available = torch.cuda.is_available()
        device_count = torch.cuda.device_count()
    except Exception as error:
        report.exception("PyTorch не смог опросить подсистему CUDA.", error)
        report.finish("Готовность PyTorch к работе с CUDA не подтверждена.")
        return 0

    report.detail("CUDA доступна", "да" if cuda_available else "нет")
    report.detail("Количество GPU", device_count)
    if not cuda_available or device_count == 0:
        if torch.version.cuda is None:
            report.error("Установлена сборка PyTorch без поддержки CUDA.")
        else:
            report.error("Сборка PyTorch поддерживает CUDA, но доступные GPU не обнаружены.")
        report.line("Проверьте драйвер NVIDIA и совместимость установленной сборки PyTorch.", indent=1)
        report.finish("PyTorch не готов к вычислениям на GPU.")
        return 0

    for index in range(device_count):
        try:
            properties = torch.cuda.get_device_properties(index)
            report.info(f"GPU {index}: {properties.name}")
            report.detail("Память всего", format_bytes(properties.total_memory), indent=1)
            report.detail(
                "Вычислительная способность",
                f"{properties.major}.{properties.minor}",
                indent=1,
            )
            report.detail("Мультипроцессоров", properties.multi_processor_count, indent=1)
        except Exception as error:
            report.exception(f"Не удалось получить свойства GPU {index}.", error)

    report.success("PyTorch обнаружил устройства CUDA.")

    report.section(4, "Практическая операция на GPU")
    passed_devices = 0
    for index in range(device_count):
        try:
            device = torch.device(f"cuda:{index}")
            source = torch.tensor([1.0, 2.0, 3.0], device=device)
            result = source.mul(2).cpu().tolist()
            torch.cuda.synchronize(device)
            if result != [2.0, 4.0, 6.0]:
                raise RuntimeError(f"получен неожиданный результат: {result}")
            report.success(f"GPU {index}: перенос данных и вычисление выполнены корректно.")
            passed_devices += 1
        except Exception as error:
            report.exception(f"GPU {index}: тестовая операция CUDA завершилась ошибкой.", error)

    if passed_devices == device_count:
        conclusion = "PyTorch успешно выполнил вычисление на всех обнаруженных GPU."
    elif passed_devices:
        conclusion = "PyTorch работает не на всех обнаруженных GPU; проверьте ошибки отдельных устройств."
    else:
        conclusion = "PyTorch видит GPU, но не смог выполнить тестовую операцию CUDA."
    report.finish(conclusion)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
