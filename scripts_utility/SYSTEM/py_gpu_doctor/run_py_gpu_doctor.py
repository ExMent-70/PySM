# pysm_lib/script_samples/py_gpu_report/run_py_gpu_report.py

import sys

# Попытка импорта необходимых библиотек
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False


def print_header(title):
    print(f"\n<h3>--- {title} ---</h3>")


print("<b>Полный отчет о конфигурации GPU и библиотек</b>")
print(f"Версия Python: {sys.version}")

# --- Проверка наличия PYNVML ---
if not PYNVML_AVAILABLE:
    print("\n❌ КРИТИЧЕСКАЯ ОШИБКА: Библиотека 'pynvml' не найдена.")
    print("  Для сбора детальной информации о драйвере и GPU необходима эта библиотека.")
    print("  Установите ее командой: pip install pynvml")
    sys.exit()

try:
    pynvml.nvmlInit()

    # --- Секция 1: Информация о драйвере и системной CUDA ---
    print_header("Информация о системе и драйвере NVIDIA")
    
    # --- НАЧАЛО ИЗМЕНЕНИЙ ВНУТРИ БЛОКА ---
    #
    # ИСПРАВЛЕНИЕ: Удаляем вызов .decode('utf-8'), так как pynvml.nvmlSystemGetDriverVersion()
    # уже возвращает строку (str), а не байты (bytes).
    #
    driver_version = pynvml.nvmlSystemGetDriverVersion()
    # --- КОНЕЦ ИЗМЕНЕНИЙ ВНУТРИ БЛОКА ---
    
    print(f"Версия драйвера NVIDIA: <b>{driver_version}</b>")

    try:
        # Эта функция может отсутствовать в очень старых драйверах
        cuda_driver_version_int = pynvml.nvmlSystemGetCudaDriverVersion()
        cuda_driver_version_str = f"{cuda_driver_version_int // 1000}.{(cuda_driver_version_int % 1000) // 10}"
        print(f"Версия CUDA (поддерживаемая драйвером): <b>{cuda_driver_version_str}</b>")
    except pynvml.NVMLError_FunctionNotFound:
        print("Версия CUDA (поддерживаемая драйвером): <b>Не удалось определить (старая версия драйвера)</b>")


    # --- Секция 2: Информация из PyTorch ---
    print_header("Информация из библиотеки PyTorch")
    if not TORCH_AVAILABLE:
        print("❌ PyTorch не установлен. Информация недоступна.")
    else:
        print(f"Версия PyTorch: <b>{torch.__version__}</b>")
        if torch.cuda.is_available():
            print("Состояние CUDA в PyTorch: ✅ <b>Доступно</b>")
            print(f"Версия PyTorch CUDA: <b>{torch.version.cuda}</b> (скомпилировано с этой версией)")
            # Проверка доступности и версии CUDNN
            if torch.backends.cudnn.is_available():
                print("Состояние CUDNN в PyTorch: ✅ <b>Доступно</b>")
                cudnn_version = torch.backends.cudnn.version()
                print(f"Версия CUDNN: <b>{cudnn_version}</b>")
            else:
                print("Состояние CUDNN в PyTorch: ❌ <b>Недоступно</b>")
        else:
            print("Состояние CUDA в PyTorch: ❌ <b>Недоступно</b>")


    # --- Секция 3: Детальная информация по каждому GPU ---
    print_header("Обнаруженные устройства GPU")
    device_count = pynvml.nvmlDeviceGetCount()
    if device_count == 0:
        print("Не найдено ни одного GPU.")
    else:
        print(f"Найдено устройств: <b>{device_count}</b>")
        for i in range(device_count):
            print(f"\n<i><u>--- Устройство ID: {i} ---</u></i>")
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            
            # --- НАЧАЛО ИЗМЕНЕНИЙ ВНУТРИ БЛОКА ---
            #
            # ИСПРАВЛЕНИЕ: Удаляем вызов .decode('utf-8'), так как pynvml.nvmlDeviceGetName()
            # также возвращает строку (str), а не байты (bytes).
            #
            name = pynvml.nvmlDeviceGetName(handle)
            # --- КОНЕЦ ИЗМЕНЕНИЙ ВНУТРИ БЛОКА ---
            
            print(f"  Название: <b>{name}</b>")

            # Память
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_mem_mb = mem_info.total / (1024**2)
            used_mem_mb = mem_info.used / (1024**2)
            free_mem_mb = mem_info.free / (1024**2)
            print(f"  Память: {used_mem_mb:.0f} МБ / {total_mem_mb:.0f} МБ (Использовано / Всего)")

            # Вычислительная способность
            try:
                major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                print(f"  Вычислительная способность (Compute Capability): <b>{major}.{minor}</b>")
            except pynvml.NVMLError_FunctionNotFound:
                 print("  Вычислительная способность (Compute Capability): <b>Не удалось определить</b>")

            # Температура
            try:
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                print(f"  Температура: <b>{temp}°C</b>")
            except pynvml.NVMLError:
                print("  Температура: Не удалось получить данные")



except pynvml.NVMLError as error:
    print(f"\n❌ ОШИБКА NVML: Не удалось инициализировать библиотеку. {error}")
    print("  Это может означать, что драйвер NVIDIA не установлен или работает некорректно.")
finally:
    if PYNVML_AVAILABLE:
        pynvml.nvmlShutdown()

print("\n<b>Отчет завершен</b>\n")