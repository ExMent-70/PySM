# installer_lib/config.py

"""
Центральный конфигурационный файл.
Хранит константы, URL-адреса и маппинги для установки зависимостей.
"""

# 1. URL-адреса для PyTorch в зависимости от версии CUDA.
#    Ключи - это версии CUDA, которые мы будем определять в системе.
TORCH_INDEX_URLS = {
    "12.8": "https://download.pytorch.org/whl/nightly/cu128",  # Для Blackwell и Ada
    "12.4": "https://download.pytorch.org/whl/cu124",
    "12.1": "https://download.pytorch.org/whl/cu121",  # Добавил для большей совместимости
    "11.8": "https://download.pytorch.org/whl/cu118",
    "cpu": "https://download.pytorch.org/whl/cpu",
}

# 2. Рекомендуемая версия CUDA для каждого поколения GPU NVIDIA (для Windows).
#    Это прямое отражение логики из Rust-кода.
GPU_GENERATION_TO_CUDA_VERSION = {
    "blackwell": "12.8",
    "ada lovelace": "12.8", # Ada также использует последнюю версию
    "ampere": "12.4",
    "turing": "12.4",
    "pascal": "11.8",
}

# 3. Паттерны для определения поколения GPU по имени видеокарты.
GPU_GENERATION_PATTERNS = {
    "blackwell": ["RTX 50"],
    "ada lovelace": ["RTX 40", "RTX ADA", "L40", "L4"],
    "ampere": ["RTX 30", "RTX A", "A40", "A100", "A6000", "A5000", "A4000"],
    "turing": ["RTX 20", "GTX 16", "TITAN RTX", "QUADRO"],
    "pascal": ["GTX 10", "TITAN X", "TESLA"],
}

# 4. Сопоставление поколения и Compute Capability
GPU_GENERATION_TO_COMPUTE_CAPABILITY = {
    "blackwell": "9.0",
    "ada lovelace": "8.9",
    "ampere": "8.6",
    "turing": "7.5",
    "pascal": "6.1",
    "unknown": "5.0", # Минимально поддерживаемая версия
}

# 5. Поколения, поддерживающие TensorRT
GPU_GENERATION_TENSORRT_SUPPORT = {"ampere", "ada lovelace", "blackwell"}

# 6. URL для pre-compiled wheel'а insightface под Windows.
INSIGHTFACE_WINDOWS_WHEEL_URL = "https://huggingface.co/hanamizuki-ai/pypi-wheels/resolve/main/insightface/insightface-0.7.3-cp311-cp311-win_amd64.whl"

# 7. Имена пакетов, относящихся к специальным категориям.
TORCH_FAMILY = {"torch", "torchvision", "torchaudio", "torchtext", "torchdata"}
ONNXRUNTIME_FAMILY = {"onnxruntime"}
INSIGHTFACE_FAMILY = {"insightface"}
TRITON_FAMILY = {"triton"}