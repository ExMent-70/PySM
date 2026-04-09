# file: window_state_manager.py

"""
Модуль для сохранения и восстановления физического состояния GUI-окон.
Является универсальным компонентом и может переиспользоваться в любых
скриптах на базе PySide6/PyQt6.
"""

import logging
from typing import Dict, Any, Optional

from PySide6.QtCore import QByteArray
from PySide6.QtWidgets import QMainWindow, QSplitter

logger = logging.getLogger(__name__)

class WindowStateManager:
    """
    Сервисный класс для сериализации и десериализации геометрии окна
    и пропорций сплиттеров в JSON-совместимый формат (Base64).
    """

    @staticmethod
    def save_state(window: QMainWindow, splitters: Optional[Dict[str, QSplitter]] = None) -> Dict[str, Any]:
        """
        Собирает геометрию окна и сплиттеров.
        
        :param window: Главное окно приложения.
        :param splitters: Словарь вида {'имя_сплиттера': QSplitter}.
        :return: Словарь с Base64-строками состояния.
        """
        state_data: Dict[str, Any] = {}
        
        try:
            # Сохраняем геометрию окна (размер, положение на мониторах)
            state_data['geometry'] = window.saveGeometry().toBase64().data().decode('utf-8')
            # Сохраняем состояние окна (максимизировано/свернуто)
            state_data['window_state'] = window.saveState().toBase64().data().decode('utf-8')
            
            # Сохраняем состояния всех переданных сплиттеров
            if splitters:
                state_data['splitters'] = {}
                for name, splitter in splitters.items():
                    state_data['splitters'][name] = splitter.saveState().toBase64().data().decode('utf-8')
                    
            logger.debug("Состояние окна успешно сохранено.")
        except Exception as e:
            logger.error(f"Ошибка при сохранении состояния окна: {e}")
            
        return state_data

    @staticmethod
    def restore_state(window: QMainWindow, state_data: Dict[str, Any], splitters: Optional[Dict[str, QSplitter]] = None) -> None:
        """
        Восстанавливает геометрию окна и сплиттеров из словаря.
        """
        if not state_data:
            return

        try:
            # Восстанавливаем окно
            if 'geometry' in state_data and state_data['geometry']:
                window.restoreGeometry(QByteArray.fromBase64(state_data['geometry'].encode('utf-8')))
            if 'window_state' in state_data and state_data['window_state']:
                window.restoreState(QByteArray.fromBase64(state_data['window_state'].encode('utf-8')))
            
            # Восстанавливаем сплиттеры
            if splitters and 'splitters' in state_data:
                saved_splitters = state_data['splitters']
                for name, splitter in splitters.items():
                    if name in saved_splitters and saved_splitters[name]:
                        splitter.restoreState(QByteArray.fromBase64(saved_splitters[name].encode('utf-8')))
                        
            logger.debug("Состояние окна успешно восстановлено.")
        except Exception as e:
            logger.error(f"Ошибка при восстановлении состояния окна: {e}")