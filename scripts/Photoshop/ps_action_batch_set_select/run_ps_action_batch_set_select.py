# 1. БЛОК: Импорты
# ==============================================================================
import argparse
import json
import sys
import os
import time  # <--- ДОБАВЛЕН ИМПОРТ
from argparse import Namespace
from pathlib import Path
from typing import Dict, List, Tuple

# Попытка импорта библиотек из экосистемы PySM.
try:
    from pysm_lib.pysm_context import ConfigResolver, pysm_context
    from pysm_lib.pysm_progress_reporter import tqdm
    IS_MANAGED_RUN = True
except ImportError:
    IS_MANAGED_RUN = False
    ConfigResolver, pysm_context = None, None
    class TqdmMock:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        @staticmethod
        def write(m, *a, **kw): print(m)
        def set_description(self, *a, **kw): pass
        def update(self, n=1): pass
    tqdm = TqdmMock

# Импорт ключевых зависимостей и GUI
try:
    from photoshop import api
    from photoshop.api.enumerations import SaveOptions
    from comtypes import COMError
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import (
        QApplication, QDialog, QVBoxLayout, QListWidget, QLabel,
        QHBoxLayout, QDialogButtonBox, QListWidgetItem, QStyle
    )
    PYSIDE_AVAILABLE = True
except ImportError:
    PYSIDE_AVAILABLE = False


# 2. БЛОК: Получение конфигурации (ИЗМЕНЕН)
# ==============================================================================
def get_config() -> Namespace:
    """Определяет аргументы скрипта и получает их значения."""
    parser = argparse.ArgumentParser(
        description="Пакетная обработка файлов с помощью экшена, который можно выбрать интерактивно."
    )
    # Аргументы для выбора файлов
    parser.add_argument(
        "--ps_mode", type=str, required=True,
        choices=['active_document', 'active_document_folder', 'selected_file', 'selected_file_folder'],
        help="Режим работы скрипта, определяющий источник файлов."
    )
    parser.add_argument(
        "--ps_file_path", type=str,
        help="Путь к файлу (для режимов 'selected_file' и 'selected_file_folder')."
    )
    parser.add_argument(
        "--ps_recursive", action="store_true", default=False,
        help="Рекурсивный поиск файлов в папках."
    )
    # Аргументы для выбора экшена
    parser.add_argument(
        "--ps_action_set", type=str,
        help="Имя набора экшенов. Если не указано, появится диалог выбора."
    )
    parser.add_argument(
        "--ps_action_name", type=str,
        help="Имя экшена. Если не указано, появится диалог выбора."
    )
    # --- НАЧАЛО ИЗМЕНЕНИЙ ВНУТРИ БЛОКА ---
    # Аргументы для управления процессом
    parser.add_argument(
        "--ps_need_save", action="store_true",
        help="Нужно ли принудительно сохранять документ после выполнения экшена."
    )
    parser.add_argument(
        "--ps_wait_after_action", type=float, default=1.0,
        help="Задержка в секундах после выполнения экшена, чтобы Photoshop успел обработать операцию."
    )
    # --- КОНЕЦ ИЗМЕНЕНИЙ ВНУТРИ БЛОКА ---

    if IS_MANAGED_RUN and ConfigResolver:
        return ConfigResolver(parser).resolve_all()
    else:
        return parser.parse_args()


# 3. БЛОК: Вспомогательные функции и классы GUI
# ==============================================================================
# ... (код без изменений) ...
def get_all_actions_js(app: api.Application) -> Dict[str, List[str]]:
    """
    Получает полный список всех наборов и их экшенов, отфильтровывая
    служебные элементы (имена вида __name__).
    """
    javascript_code = """
    function getAllActions() {
        var sets = []; var i = 1;
        while (true) {
            var ref = new ActionReference(); ref.putIndex(stringIDToTypeID("actionSet"), i);
            var desc; try { desc = executeActionGet(ref); } catch (e) { break; }
            var setName = desc.getString(stringIDToTypeID("name"));
            var numChildren = desc.getInteger(stringIDToTypeID("numberOfChildren"));
            var actions = [];
            if (numChildren > 0) {
                for (var j = 1; j <= numChildren; j++) {
                    var ref2 = new ActionReference();
                    ref2.putIndex(stringIDToTypeID("action"), j);
                    ref2.putIndex(stringIDToTypeID("actionSet"), i);
                    var desc2 = executeActionGet(ref2);
                    actions.push(desc2.getString(stringIDToTypeID("name")));
                }
            }
            sets.push({ "name": setName, "actions": actions });
            i++;
        }
        return JSON.stringify(sets);
    }
    getAllActions();
    """
    try:
        json_result = app.eval_javascript(javascript_code)
        list_of_sets = json.loads(json_result)
        
        filtered_data = {}
        for item in list_of_sets:
            set_name = item['name']
            if set_name.startswith('__') and set_name.endswith('__'):
                continue

            actions = [
                action for action in item['actions'] 
                if not (action.startswith('__') and action.endswith('__'))
            ]
            
            if actions:
                filtered_data[set_name] = actions
                
        return filtered_data
    except Exception as e:
        tqdm.write(f"Ошибка при получении списка экшенов: {e}")
        return {}

class ActionSelectorDialog(QDialog):
    """Класс диалогового окна для выбора экшена."""
    def __init__(self, actions_data: Dict[str, List[str]], initial_set: str = None, initial_action: str = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Выбор экшена для выполнения")
        self.setMinimumSize(650, 400)
        self.actions_data = actions_data
        
        self.icon_set_closed = self.style().standardIcon(QStyle.StandardPixmap.SP_DirClosedIcon)
        self.icon_set_open = self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon)
        self.icon_action = self.style().standardIcon(QStyle.StandardPixmap.SP_FileIcon)
        
        self.set_label = QLabel("Набор экшенов:")
        self.set_list = QListWidget()
        self.action_label = QLabel("Экшен:")
        self.action_list = QListWidget()
        
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        
        for set_name in sorted(self.actions_data.keys()):
            self.set_list.addItem(QListWidgetItem(self.icon_set_closed, set_name))
        
        self.set_list.currentItemChanged.connect(self.update_actions_list)
        self.action_list.itemDoubleClicked.connect(self.accept)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        
        self.setup_layout()

        if initial_set:
            items = self.set_list.findItems(initial_set, Qt.MatchFlag.MatchExactly)
            if items:
                self.set_list.setCurrentItem(items[0])
                self.update_actions_list(items[0], None, initial_action)
        elif self.set_list.count() > 0:
            self.set_list.setCurrentRow(0)

    def setup_layout(self):
        ok_button = self.button_box.button(QDialogButtonBox.Ok); ok_button.setText("ОК")
        cancel_button = self.button_box.button(QDialogButtonBox.Cancel); cancel_button.setText("Отмена")
        lists_layout = QHBoxLayout()
        set_layout = QVBoxLayout(); set_layout.addWidget(self.set_label); set_layout.addWidget(self.set_list)
        lists_layout.addLayout(set_layout, 1)
        action_layout = QVBoxLayout(); action_layout.addWidget(self.action_label); action_layout.addWidget(self.action_list)
        lists_layout.addLayout(action_layout, 2)
        main_layout = QVBoxLayout(); main_layout.addLayout(lists_layout); main_layout.addWidget(self.button_box)
        self.setLayout(main_layout)

    def update_actions_list(self, current_item, previous_item, initial_action_to_select=None):
        if previous_item: previous_item.setIcon(self.icon_set_closed)
        if not current_item:
            self.action_list.clear()
            return
        current_item.setIcon(self.icon_set_open)
        set_name = current_item.text()
        self.action_list.clear()
        actions = self.actions_data.get(set_name, [])
        for action_name in actions:
            self.action_list.addItem(QListWidgetItem(self.icon_action, action_name))
        if initial_action_to_select:
            action_items = self.action_list.findItems(initial_action_to_select, Qt.MatchFlag.MatchExactly)
            if action_items: self.action_list.setCurrentItem(action_items[0])
        elif self.action_list.count() > 0:
            self.action_list.setCurrentRow(0)

    def get_selection(self) -> Tuple[str | None, str | None]:
        set_item = self.set_list.currentItem()
        action_item = self.action_list.currentItem()
        return (set_item.text() if set_item else None, action_item.text() if action_item else None)


# 4. БЛОК: Определение списка файлов
# ==============================================================================
# ... (код без изменений) ...
def get_files_to_process(config: Namespace, app: api.Application) -> List[str]:
    """На основе режима работы определяет и возвращает список путей к файлам."""
    mode = config.ps_mode
    tqdm.write(f"\nОпределение списка файлов для режима: <b>{mode}</b>")
    
    target_folder = None
    if mode == 'active_document':
        if len(app.documents) == 0:
            tqdm.write("ОШИБКА: Режим <i>active_document</i> требует наличия открытого документа.")
            return []
        try:
            return [str(app.activeDocument.fullName)]
        except COMError:
            tqdm.write("ОШИБКА: Активный документ должен быть сохранен, чтобы получить его путь.")
            return []

    elif mode == 'active_document_folder':
        if len(app.documents) == 0:
            tqdm.write("ОШИБКА: Режим <i>active_document_folder</i> требует открытого документа.")
            return []
        try:
            target_folder = Path(app.activeDocument.path)
            tqdm.write(f"Целевая папка (из активного документа): {target_folder}")
        except COMError:
            tqdm.write("ОШИБКА: Активный документ должен быть сохранен.")
            return []

    elif mode == 'selected_file':
        if not config.ps_file_path:
            tqdm.write("ОШИБКА: Для режима <i>selected_file</i> нужен --ps_file_path.")
            return []
        target_file = Path(config.ps_file_path)
        if not target_file.is_file():
            tqdm.write(f"ОШИБКА: Файл не найден: {target_file}")
            return []
        return [str(target_file)]

    elif mode == 'selected_file_folder':
        if not config.ps_file_path:
            tqdm.write("ОШИБКА: Для режима <i>selected_file_folder</i> нужен --ps_file_path.")
            return []
        target_file = Path(config.ps_file_path)
        if not target_file.is_file():
            # Если файл не найден, пытаемся использовать его родительскую папку
            target_folder = target_file.parent
            if not target_folder.is_dir():
                tqdm.write(f"ОШИБКА: Файл '{target_file}' не найден, и его родительская папка '{target_folder}' также не существует.")
                return []
            tqdm.write(f"ВНИМАНИЕ: Файл '{target_file.name}' не найден, работаем с его родительской папкой: {target_folder}")
        else:
            target_folder = target_file.parent
            tqdm.write(f"Целевая папка (из указанного файла): {target_folder}")

    if target_folder:
        tqdm.write(f"Поиск *.psd в: {target_folder} (Рекурсия: {'Вкл' if config.ps_recursive else 'Выкл'})")
        pattern = "**/*.psd" if config.ps_recursive else "*.psd"
        image_files_paths = list(target_folder.glob(pattern))
        return sorted([str(f) for f in image_files_paths if f.is_file()])

    return []

# 5. БЛОК: Главная функция-оркестратор (ИЗМЕНЕН)
# ==============================================================================
def main():
    """Главная функция, управляющая всем процессом."""
    config = get_config()
    print("<b>Пакетная обработка с выбором экшена</b>")

    try:
        app = api.Application()
    except COMError:
        tqdm.write("ОШИБКА: Не удалось подключиться к Photoshop."); sys.exit(1)

    # ЭТАП 1: Определить, какой экшен выполнять
    action_set, action_name = config.ps_action_set, config.ps_action_name
    
    if not (action_set and action_name):
        tqdm.write("\nИмя экшена не указано, переход в интерактивный режим...")
        if not PYSIDE_AVAILABLE:
            tqdm.write("ОШИБКА: PySide6 не найдена, интерактивный режим невозможен."); sys.exit(1)
        actions_data = get_all_actions_js(app)
        if not actions_data:
            tqdm.write("Не удалось получить список экшенов или он пуст."); sys.exit(1)
        q_app = QApplication.instance() or QApplication(sys.argv)
        dialog = ActionSelectorDialog(actions_data, action_set, action_name)
        if dialog.exec() == QDialog.Accepted:
            action_set, action_name = dialog.get_selection()
            if not action_set or not action_name:
                tqdm.write("Выбор не сделан. Операция отменена."); sys.exit(0)
        else:
            tqdm.write("Операция отменена пользователем."); sys.exit(0)
    
    tqdm.write(f"Будет выполнен экшен: <b>{action_name}</b> из набора <b>{action_set}</b>")

    # ЭТАП 2: Определить список файлов
    initial_active_doc_name = app.activeDocument.name if len(app.documents) > 0 else None
    image_files = get_files_to_process(config, app)

    if not image_files:
        tqdm.write("\nНе найдено файлов для обработки. Завершение работы."); sys.exit(0)
    
    work_folder = Path(image_files[0]).parent if image_files else None
    print(f"\nНайдено {len(image_files)} файлов для обработки.")

    # ЭТАП 3: Выполнить пакетную обработку
    # --- НАЧАЛО ИЗМЕНЕНИЙ ВНУТРИ БЛОКА ---
    with tqdm(total=len(image_files), desc="Обработка", unit="file", dynamic_ncols=True) as progress_bar:
        for full_path in image_files:
            doc = None
            file_name = Path(full_path).name
            try:
                progress_bar.set_description(f"Обработка: {file_name}")
                doc = app.open(full_path)
                app.doAction(action_name, action_set)

                # Добавляем задержку
                if config.ps_wait_after_action > 0:
                    time.sleep(config.ps_wait_after_action)

                # Сохраняем документ, если указан флаг
                if config.ps_need_save:
                    doc.save()
                    
            except Exception as e:
                tqdm.write(f"\nОШИБКА при обработке '{file_name}': {e}")
            finally:
                # Отказоустойчивое закрытие документа
                if doc:
                    try:
                        if doc.name != initial_active_doc_name:
                            doc.close(SaveOptions.DoNotSaveChanges)
                    except Exception as close_e:
                        tqdm.write(f"\nВНИМАНИЕ: Не удалось корректно закрыть '{file_name}'. Возможно, он был изменен экшеном. Ошибка: {close_e}")
                
                progress_bar.update(1)
    # --- КОНЕЦ ИЗМЕНЕНИЙ ВНУТРИ БЛОКА ---

    print("\n<b>Пакетная обработка завершена.</b>\n")
    if IS_MANAGED_RUN and work_folder:
        pysm_context.log_link(
            url_or_path=str(work_folder),
            text=f"Открыть рабочую папку: {work_folder}",
        )
    sys.exit(0)

# 6. БЛОК: Точка входа
# ==============================================================================
if __name__ == "__main__":
    main()