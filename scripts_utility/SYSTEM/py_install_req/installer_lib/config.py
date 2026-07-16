# installer_lib/config.py

"""
Центральный конфигурационный файл.
Хранит константы, URL-адреса и маппинги для установки зависимостей.
"""

TORCH_INDEX_URLS = {
    "12.8": "https://download.pytorch.org/whl/cu128",
    "12.6": "https://download.pytorch.org/whl/cu126",
    "12.4": "https://download.pytorch.org/whl/cu124",
    "12.1": "https://download.pytorch.org/whl/cu121",
    "11.8": "https://download.pytorch.org/whl/cu118",
    "cpu": "https://download.pytorch.org/whl/cpu",
}

GPU_GENERATION_TO_CUDA_VERSION = {
    "blackwell": "12.8",
    "ada lovelace": "12.8",
    "ampere": "12.4",
    "turing": "12.4",
    "pascal": "11.8",
}

GPU_GENERATION_PATTERNS = {
    "blackwell": ["RTX 50"],
    "ada lovelace": ["RTX 40", "RTX ADA", "L40", "L4"],
    "ampere": ["RTX 30", "RTX A", "A40", "A100", "A6000", "A5000", "A4000"],
    "turing": ["RTX 20", "GTX 16", "TITAN RTX", "QUADRO"],
    "pascal": ["GTX 10", "TITAN X", "TESLA"],
}

GPU_GENERATION_TO_COMPUTE_CAPABILITY = {
    "blackwell": "9.0",
    "ada lovelace": "8.9",
    "ampere": "8.6",
    "turing": "7.5",
    "pascal": "6.1",
    "unknown": "5.0",
}

GPU_GENERATION_TENSORRT_SUPPORT = {"ampere", "ada lovelace", "blackwell"}

TORCH_FAMILY = {"torch", "torchvision", "torchaudio", "torchtext", "torchdata"}
ONNXRUNTIME_FAMILY = {"onnxruntime", "onnxruntime-gpu", "onnxruntime-directml"}
INSIGHTFACE_FAMILY = {"insightface"}
TRITON_FAMILY = {"triton"}
