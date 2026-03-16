# analize/analyze_faces/face_lib/result_writer.py

import logging
import gc
from typing import List, Dict, Tuple
import numpy as np

# Импортируем только для аннотации типов, сам класс передается в __init__
if False:
    from _common.face_storage import FaceStorageManager

logger = logging.getLogger(__name__)

class AnalysisResultWriter:
    """
    Класс-буфер для накопления результатов анализа.
    Накапливает данные в памяти до достижения batch_size, затем сбрасывает их
    на диск через FaceStorageManager.
    """

    def __init__(self, storage_manager: 'FaceStorageManager', batch_size: int = 50):
        """
        Args:
            storage_manager: Экземпляр менеджера хранения для записи на диск.
            batch_size: Количество изображений в буфере перед сбросом на диск.
        """
        self.storage_manager = storage_manager
        self.batch_size = batch_size
        
        # Буфер хранит кортежи: (filename, faces_meta, embeddings_list, original_shape)
        self._buffer: List[Tuple[str, List[Dict], List[np.ndarray], Tuple[int, int]]] = []

    def add_result(self, filename: str, meta: List[Dict], embeddings: List[np.ndarray], original_shape: Tuple[int, int]):
        """
        Добавляет результат анализа одного файла в буфер.
        Автоматически вызывает flush(), если буфер заполнен.
        """
        self._buffer.append((filename, meta, embeddings, original_shape))
        
        if len(self._buffer) >= self.batch_size:
            self.flush()

    def flush(self):
        """Принудительно записывает содержимое буфера на диск и очищает память."""
        if not self._buffer:
            return

        logger.debug(f"Сброс буфера результатов ({len(self._buffer)} файлов)...")
        
        # Передаем данные менеджеру хранения
        self.storage_manager.save_batch(self._buffer)
        
        # Очищаем буфер
        self._buffer.clear()
        
        # Принудительный сбор мусора полезен при обработке больших массивов изображений,
        # чтобы избежать фрагментации памяти и OOM на длинных дистанциях.
        gc.collect()

    def close(self):
        """
        Метод завершения работы. Сбрасывает остатки буфера.
        Должен вызываться перед финализацией storage_manager.
        """
        self.flush()