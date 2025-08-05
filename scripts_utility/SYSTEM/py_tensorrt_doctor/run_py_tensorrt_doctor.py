# pysm_lib/script_samples/py_tensorrt_doctor/run_py_tensorrt_doctor.py

import sys

# Оборачиваем главный импорт в try-except, так как это самая частая точка отказа.
try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False


print("<b>Диагностический скрипт TensorRT</b>")
print(f"Версия Python: {sys.version}")


# --- Тест 1: Можем ли мы импортировать библиотеку? ---
print("\n<b><i>[Тест 1] Проверка импорта библиотеки 'tensorrt'</i></b>")
if not TENSORRT_AVAILABLE:
    print("❌ КРИТИЧЕСКАЯ ОШИБКА: Не удалось импортировать библиотеку 'tensorrt'.")
    print("Это корневая причина проблемы. Последующие тесты будут пропущены.")
    print("Обычно это означает одно из следующего:")
    print("  1. Python-пакет 'tensorrt' не установлен (например, через 'pip install tensorrt').")
    print("  2. Вы используете установленную вручную версию TensorRT, но путь к ее библиотекам не прописан в системных переменных PATH или LD_LIBRARY_PATH.")
    print("  3. Существует конфликт с другой библиотекой.")
    sys.exit() # Выходим раньше, нет смысла продолжать
else:
    print("✅ УСПЕХ: Библиотека 'tensorrt' успешно импортирована.")


# --- Тест 2: Можем ли мы получить версию и создать логгер? ---
print("\n<b><i>[Тест 2] Запрос версии и создание логгера</i></b>")
try:
    trt_version = trt.__version__
    print(f"Результат: Версия TensorRT - {trt_version}")
    
    # Логгер является основой для всех операций в TensorRT
    logger = trt.Logger(trt.Logger.WARNING)
    print("  - Логгер TensorRT успешно создан.")
    print("✅ УСПЕХ: Версия запрошена и логгер создан.")
except Exception as e:
    print(f"❌ ОШИБКА: Произошло исключение во время базовой инициализации: {e}")
    sys.exit()


# --- Тест 3: ГЛАВНЫЙ ТЕСТ - Сборка простого движка ---
print("\n<b><i>[Тест 3] Попытка собрать простой inference-движок</i></b>")
# Этот тест проверяет, что TensorRT может взаимодействовать с GPU и зависимыми библиотеками (cuDNN)
EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
builder = trt.Builder(logger)
network = builder.create_network(EXPLICIT_BATCH)
config = builder.create_builder_config()

try:
    # Создаем минимальную сеть: вход -> identity-слой -> выход
    # Это самый простой возможный граф.
    input_tensor = network.add_input(name="input", dtype=trt.float32, shape=(1, 3, 224, 224))
    identity_layer = network.add_identity(input_tensor)
    output_tensor = identity_layer.get_output(0)
    network.mark_output(output_tensor)
    print("  - Создано минимальное определение сети.")

    # Пытаемся собрать движок
    print("  - Собираем движок... (Это ключевой шаг)")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        print("❌❌❌ КРИТИЧЕСКИЙ СБОЙ: Сборка движка не удалась, сборщик вернул None.")
        print("Это часто указывает на серьезную несовместимость между TensorRT, cuDNN, CUDA Toolkit и драйвером NVIDIA.")
    else:
        # --- НАЧАЛО ИЗМЕНЕНИЙ ВНУТРИ БЛОКА ---
        #
        # ИСПРАВЛЕНИЕ: Преобразуем объект IHostMemory в bytes и используем стандартную функцию len().
        #
        engine_bytes = bytes(serialized_engine)
        print(f"  - Движок успешно собран! Размер: {len(engine_bytes)} байт.")
        # --- КОНЕЦ ИЗМЕНЕНИЙ ВНУТРИ БЛОКА ---
        print("✅✅✅ ПОЛНЫЙ УСПЕХ: Все операции TensorRT, по-видимому, работают корректно!")

except Exception as e:
    print("❌❌❌ КРИТИЧЕСКИЙ СБОЙ: Произошла ошибка выполнения во время сборки движка.")
    print("Это конкретное сообщение об ошибке, которое нам нужно:")
    import traceback
    traceback.print_exc()


print("\n<b>Диагностика завершена</b>\n")