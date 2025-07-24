# 1. БЛОК: Импорты
# ==============================================================================
import argparse
import json
import sys
import os
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
        def __init__(self, i, *a, **kw): self.i = i
        def __iter__(self): return iter(self.i)
        @staticmethod
        def write(m, *a, **kw): print(m)
        def set_description(self, *a, **kw): pass
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


# 2. БЛОК: Получение конфигурации
# ==============================================================================
def get_config() -> Namespace:
    """Определяет аргументы скрипта и получает их значения."""
    parser = argparse.ArgumentParser(
        description="Пакетная обработка файлов с помощью экшена, который можно выбрать интерактивно."
    )
    # Аргументы из run_ps_action_batch.py
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
    # Аргументы из run_ps_action_set_select.py (сделаны необязательными)
    parser.add_argument(
        "--ps_action_set", type=str,
        help="Имя набора экшенов. Если не указано, появится диалог выбора."
    )
    parser.add_argument(
        "--ps_action_name", type=str,
        help="Имя экшена. Если не указано, появится диалог выбора."
    )

    if IS_MANAGED_RUN and ConfigResolver:
        return ConfigResolver(parser).resolve_all()
    else:
        return parser.parse_args()


# 3. БЛОК: Вспомогательные функции и классы GUI (из run_ps_action_set_select.py)
# ==============================================================================
def get_all_actions_js(app: api.Application) -> Dict[str, List[str]]:
    """
    Получает полный список всех наборов и их экшенов, отфильтровывая
    служебные элементы (имена вида __name__).
    """
    # --- НАЧАЛО ИЗМЕНЕНИЙ ВНУТРИ БЛОКА ---
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
            // КОММЕНТАРИЙ: Не фильтруем пустые наборы здесь, чтобы сохранить их в списке,
            // если они не служебные. Фильтрацию пустых произведем в Python.
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
        
        # КОММЕНТАРИЙ: Фильтрация служебных элементов и пустых наборов в Python.
        filtered_data = {}
        for item in list_of_sets:
            set_name = item['name']
            # Пропускаем служебные наборы
            if set_name.startswith('__') and set_name.endswith('__'):
                continue

            # Фильтруем служебные экшены внутри набора
            actions = [
                action for action in item['actions'] 
                if not (action.startswith('__') and action.endswith('__'))
            ]
            
            # Добавляем набор в итоговый словарь, только если он содержит экшены
            if actions:
                filtered_data[set_name] = actions
                
        return filtered_data
    except Exception as e:
        tqdm.write(f"Ошибка при получении списка экшенов: {e}")
        return {}
    # --- КОНЕЦ ИЗМЕНЕНИЙ ВНУТРИ БЛОКА ---

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


# 4. БЛОК: Определение списка файлов (из run_ps_action_batch.py)
# ==============================================================================
def get_files_to_process(config: Namespace, app: api.Application) -> List[str]:
    """На основе режима работы определяет и возвращает список путей к файлам."""
    mode = config.ps_mode
    print(f"\nРежим работы: <b>{mode}</b>. Подготовка списка файлов")
    
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
            #tqdm.write(f"Целевая папка (из активного документа): {target_folder}")
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
            tqdm.write(f"\nФайл {target_file} не найден. Определение рабочей папки...")
        target_folder = target_file.parent
        if not os.path.exists(target_folder):
            tqdm.write(f"Рабочая папка {target_folder} не найдена. Завершение работы...")
            return []
  
    if target_folder:
        print(f"\n{'Рекурсивный поиск' if config.ps_recursive else 'Поиск'} файлов PSD в папке {target_folder}.")
        pattern = "**/*.psd" if config.ps_recursive else "*.psd"
        image_files_paths = list(target_folder.glob(pattern))
        return sorted([str(f) for f in image_files_paths if f.is_file()])

    return []


# 5. БЛОК: Главная функция-оркестратор
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
    # -------------------------------------------
    action_set, action_name = config.ps_action_set, config.ps_action_name
    
    # --- НАЧАЛО ИЗМЕНЕНИЙ ВНУТРИ БЛОКА ---
    if action_set and action_name:
        # Автоматический режим: Проверяем, не пытается ли пользователь запустить служебный экшен.
        is_service_set = action_set.startswith('__') and action_set.endswith('__')
        is_service_action = action_name.startswith('__') and action_name.endswith('__')
        if is_service_set or is_service_action:
            tqdm.write(f"ОШИБКА: Запуск служебных экшенов или наборов (вида '__имя__') в автоматическом режиме запрещен.")
            sys.exit(1)
        
        # Проверка существования (простой вариант, без вызова JS)
        print("Автоматический режим: проверка параметров...")
        # Полную проверку существования можно опустить, т.к. doAction сам вернет ошибку.
        
    else:
        # Интерактивный режим
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
    # --- КОНЕЦ ИЗМЕНЕНИЙ ВНУТРИ БЛОКА ---
    
    print(f"\nЗапуск экшена <b>{action_name}</b> из набора <b>{action_set}</b>")

    # ЭТАП 2: Определить список файлов для обработки
    # -----------------------------------------------
    initial_active_doc_name = app.activeDocument.name if len(app.documents) > 0 else None
    image_files = get_files_to_process(config, app)

    if not image_files:
        tqdm.write("\nНе найдено файлов для обработки. Завершение работы."); sys.exit(0)
        
    print(f"\nНайдено <b>{len(image_files)}</b> файлов. Начало пакетной обработки.")

    # ЭТАП 3: Выполнить пакетную обработку
    # --------------------------------------
    with tqdm(total=len(image_files), desc="Обработка", unit="file", dynamic_ncols=True) as progress_bar:
        for full_path in image_files:
            doc = None
            file_name = Path(full_path).name
            try:
                progress_bar.set_description(f"Обработка: {file_name}")
                doc = app.open(full_path)
                app.doAction(action_name, action_set)
                doc.save()
            except Exception as e:
                tqdm.write(f"\nОШИБКА при обработке '{file_name}': {e}")
            finally:
                if doc and doc.name != initial_active_doc_name:
                    doc.close(SaveOptions.DoNotSaveChanges)
                progress_bar.update(1)

    print("\n<b>Пакетная обработка завершена.</b>\n")
    pysm_context.log_link(
        url_or_path=str(Path(config.ps_file_path).parent), # Передаем строку, а не объект Path
        text=f"Открыть папку с обработанными файлами PSD",
        )  
    print(" ", file=sys.stderr)


    sys.exit(0)


# 6. БЛОК: Точка входа
# ==============================================================================
if __name__ == "__main__":
    main()