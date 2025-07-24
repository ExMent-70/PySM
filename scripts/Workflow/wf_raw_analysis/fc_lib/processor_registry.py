# fc_lib/processor_registry.py
# --- ИЗМЕНЕНИЙ НЕ ТРЕБУЕТСЯ ---

import logging
from typing import List
from .fc_config import ConfigManager
from .face_data_processor_interface import FaceDataProcessorInterface
from .fc_attribute_analyzer import AttributeAnalyzer
from .fc_onnx_manager import ONNXModelManager
# from .fc_face_describer import FaceDescriber # Пример другого обработчика


logger = logging.getLogger(__name__)





def initialize_face_data_processors(
    config: ConfigManager,
    run_image_analysis_flag: bool,
    onnx_manager: ONNXModelManager,
) -> List[FaceDataProcessorInterface]:
    """
    Инициализирует активные обработчики данных лица. Обработчики,
    зависящие от анализа изображений, создаются только если
    run_image_analysis_flag = True.
    """
    active_processors: List[FaceDataProcessorInterface] = []
    logger.debug("Инициализация обработчиков данных лица...")

    if run_image_analysis_flag:
        try:
            # AttributeAnalyzer не требует onnx_manager в конструкторе.
            processor = AttributeAnalyzer(config)
            
            if processor.is_enabled:
                active_processors.append(processor)
                logger.info(f"Обработчик {type(processor).__name__} активирован.")
            else:
                logger.info(f"Обработчик {type(processor).__name__} отключен.")
        except Exception as e:
            logger.error(f"Ошибка инициализации AttributeAnalyzer: {e}", exc_info=True)
    else:
        logger.info(
            "Инициализация AttributeAnalyzer пропущена (анализ изображений отключен)."
        )

    logger.debug(
        f"Всего активировано обработчиков данных лица: {len(active_processors)}"
    )
    return active_processors
