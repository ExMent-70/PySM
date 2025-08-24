# run_py_tensorrt_doctor.py

import sys
import traceback

# 1. Определяем флаги для отслеживания состояния системы.
direct_api_available = False
onnx_ep_available = False

print("<b>Диагностический скрипт TensorRT (v2.0)</b>")
print(f"Версия Python: {sys.version}")

# --- Тест 1: Проверка прямого API (пакет 'tensorrt') ---
print("\n<b><i>[Тест 1] Проверка прямого доступа через пакет 'tensorrt'</i></b>")
try:
    import tensorrt as trt
    direct_api_available = True
    print("✅ УСПЕХ: Python-пакет 'tensorrt' импортирован.")
except ImportError:
    print("ℹ️ ИНФО: Python-пакет 'tensorrt' не установлен.")
    print("   Это нормально для большинства сценариев, где TensorRT используется через ONNX Runtime.")
    print("   Продолжаем проверку интеграции с ONNX Runtime...")
except Exception as e:
    print(f"❌ ОШИБКА: Неожиданная проблема при импорте 'tensorrt': {e}")


# --- Тест 2: Проверка интеграции с ONNX Runtime (самый важный тест) ---
print("\n<b><i>[Тест 2] Проверка доступности TensorRT через ONNX Runtime</i></b>")
try:
    import onnxruntime as ort
    print("  - Пакет 'onnxruntime' успешно импортирован.")
    
    available_providers = ort.get_available_providers()
    print(f"  - Доступные провайдеры ONNX: {available_providers}")

    if 'TensorrtExecutionProvider' in available_providers:
        onnx_ep_available = True
        print("✅ УСПЕХ: 'TensorrtExecutionProvider' доступен!")
        print("   Это означает, что ONNX Runtime может использовать TensorRT для ускорения.")
    else:
        print("⚠️ ПРЕДУПРЕЖДЕНИЕ: 'TensorrtExecutionProvider' не найден в списке доступных.")
        print("   Это может означать, что установлена версия onnxruntime без поддержки GPU,")
        print("   либо есть несовместимость с версией CUDA или драйвера NVIDIA.")

except ImportError:
    print("❌ ОШИБКА: Не удалось импортировать 'onnxruntime'.")
    print("   Вероятно, пакет 'onnxruntime-gpu' не установлен или установлен некорректно.")
except Exception as e:
    print(f"❌ ОШИБКА: Произошло исключение при проверке ONNX Runtime: {e}")


# --- Тест 3: Глубокая проверка прямого API (если он доступен) ---
if direct_api_available:
    print("\n<b><i>[Тест 3] Глубокая проверка прямого API TensorRT...</i></b>")
    try:
        trt_version = trt.__version__
        logger = trt.Logger(trt.Logger.WARNING)
        print(f"  - Версия TensorRT: {trt_version}")
        print("  - Логгер TensorRT успешно создан.")

        EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        builder = trt.Builder(logger)
        network = builder.create_network(EXPLICIT_BATCH)
        config = builder.create_builder_config()
        
        input_tensor = network.add_input(name="input", dtype=trt.float32, shape=(1, 3, 224, 224))
        identity_layer = network.add_identity(input_tensor)
        network.mark_output(identity_layer.get_output(0))
        print("  - Минимальная сеть для сборки движка создана.")
        
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine:
            print(f"  - Движок успешно собран! Размер: {len(bytes(serialized_engine))} байт.")
            print("✅ УСПЕХ: Прямой API TensorRT полностью функционален.")
        else:
            print("❌ ОШИБКА: Сборка движка через прямой API не удалась (builder вернул None).")

    except Exception as e:
        print("❌ КРИТИЧЕСКАЯ ОШИБКА: Произошло исключение при глубокой проверке прямого API.")
        traceback.print_exc()

# --- Итоговое заключение ---
print("\n<b>--- Заключение по диагностике ---</b>")
if onnx_ep_available:
    print("✅ <b>Ваша система готова к ускорению моделей через ONNX Runtime с использованием TensorRT.</b>")
    print("   Это основной и самый распространенный сценарий использования.")
    if not direct_api_available:
        print("   (Прямой доступ к API через `import tensorrt` не настроен, но для большинства приложений он и не требуется).")
elif direct_api_available:
    print("⚠️ <b>Ваша система настроена для разработки с прямым API TensorRT, но ONNX Runtime не может его использовать.</b>")
    print("   Проверьте совместимость версий `onnxruntime-gpu` и `tensorrt`.")
else:
    print("❌ <b>TensorRT не доступен ни напрямую, ни через ONNX Runtime.</b>")
    print("   Убедитесь, что установлен пакет 'onnxruntime-gpu' и что версии драйвера NVIDIA и CUDA совместимы.")

print("\n<b>Диагностика завершена.</b>\n")