# История изменений

## 2.2026.07.16

- Удалена специальная установка внешнего wheel InsightFace 0.7.3 для `cp311`.
- InsightFace 1.0.1 устанавливается с `--no-deps`, чтобы зависимость CPU `onnxruntime` не подменила `onnxruntime-gpu`.
- Удалён принудительный pin `numpy==1.26.4`; версия NumPy теперь определяется корневым `requirements.txt`.
- Добавлена постпроверка пакета `onnxruntime-gpu`, отсутствия конфликта CPU/GPU-дистрибутивов и наличия CUDA/TensorRT provider.
