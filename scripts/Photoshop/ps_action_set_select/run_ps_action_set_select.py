# 1. БЛОК: Импорты
# ==============================================================================
import argparse
import json
import sys
from argparse import Namespace
from typing import Dict, List, Tuple

# Попытка импорта библиотек из экосистемы PySM.
try:
    from pysm_lib.pysm_context import ConfigResolver, pysm_context
    from pysm_lib.pysm_progress_reporter import tqdm
    IS_MANAGED_RUN = True
except ImportError:
    IS_MANAGED_RUN = False
    ConfigResolver, pysm_context = None, None
    class TqdmWriteMock:
        @staticmethod
        def write(msg, *args, **kwargs): print(msg)
    tqdm = TqdmWriteMock()

# Импорт ключевых зависимостей и GUI
try:
    from photoshop import api
    from comtypes import COMError
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import (
        QApplication, QDialog, QVBoxLayout, QListWidget, QLabel, 
        QHBoxLayout, QDialogButtonBox, QListWidgetItem, QStyle
    )
    PYSIDE_AVAILABLE = True
except ImportError:
    PYSIDE_AVAILABLE = False


# 2. БЛОК: Вспомогательные функции и классы
# ==============================================================================
def get_all_actions_js(app: api.Application) -> Dict[str, List[str]]:
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
            if (actions.length > 0) { sets.push({ "name": setName, "actions": actions }); }
            i++;
        }
        return JSON.stringify(sets);
    }
    getAllActions();
    """
    try:
        json_result = app.eval_javascript(javascript_code)
        list_of_sets = json.loads(json_result)
        return {item['name']: item['actions'] for item in list_of_sets}
    except Exception as e:
        tqdm.write(f"Ошибка при получении списка экшенов: {e}")
        return {}


class ActionSelectorDialog(QDialog):
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
        
        for set_name in self.actions_data.keys():
            self.set_list.addItem(QListWidgetItem(self.icon_set_closed, set_name))
        
        self.set_list.currentItemChanged.connect(self.update_actions_list_selection)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        
        self.setup_layout()

        # Логика предвыбора
        if initial_set:
            items = self.set_list.findItems(initial_set, Qt.MatchFlag.MatchExactly)
            if items:
                self.set_list.setCurrentItem(items[0])
                # Обновляем список экшенов и пытаемся выбрать начальный
                self.update_actions_list(items[0], None, initial_action_to_select=initial_action)
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
        if previous_item:
            previous_item.setIcon(self.icon_set_closed)
        
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
            if action_items:
                self.action_list.setCurrentItem(action_items[0])
        elif self.action_list.count() > 0:
            self.action_list.setCurrentRow(0)

    def update_actions_list_selection(self, current, prev):
        self.update_actions_list(current, prev)
        
    def get_selection(self) -> Tuple[str | None, str | None]:
        set_item = self.set_list.currentItem()
        action_item = self.action_list.currentItem()
        return (set_item.text() if set_item else None, action_item.text() if action_item else None)


# 3. БЛОК: Получение и обработка конфигурации
# ==============================================================================
def get_raw_config() -> Namespace:
    parser = argparse.ArgumentParser(description="Выбор и запуск экшена Photoshop.")
    parser.add_argument("--ps_action_set", type=str, help="Имя набора экшенов для авто-режима или предвыбора.")
    parser.add_argument("--ps_action_name", type=str, help="Имя экшена для авто-режима или предвыбора.")
    parser.add_argument("--__show_dialog_always", action="store_true", help="Служебный: всегда показывать диалог.")
    parser.add_argument("--__save_set_to_context", action="store_true", help="Служебный: сохранить имя набора в контекст.")
    parser.add_argument("--__save_action_to_context", action="store_true", help="Служебный: сохранить имя экшена в контекст.")
    parser.add_argument("--__execute_action", action="store_true", help="Служебный: выполнить экшен.")
    
    if IS_MANAGED_RUN:
        return ConfigResolver(parser).resolve_all()
    return parser.parse_args()

def process_config(raw_config: Namespace) -> Namespace:
    return raw_config


# 4. БЛОК: Главная функция-оркестратор
# ==============================================================================
def main():
    raw_config = get_raw_config()
    config = process_config(raw_config)

    tqdm.write("--- Скрипт выбора и запуска экшена: Начало ---")
    
    try:
        app = api.Application()
    except COMError:
        tqdm.write("ОШИБКА: Не удалось подключиться к Photoshop."); sys.exit(1)

    show_dialog = config.__show_dialog_always or not (config.ps_action_set and config.ps_action_name)

    if not show_dialog:
        # Автоматический режим
        tqdm.write("Автоматический режим: проверка и запуск по переданным параметрам...")
        all_actions = get_all_actions_js(app)
        set_exists = config.ps_action_set in all_actions
        action_exists = set_exists and config.ps_action_name in all_actions[config.ps_action_set]

        if not action_exists:
            tqdm.write(f"ОШИБКА: Экшен '{config.ps_action_name}' или набор '{config.ps_action_set}' не найдены."); sys.exit(1)
        
        tqdm.write("Проверка существования экшена пройдена успешно.")
        if config.__execute_action:
            if len(app.documents) == 0:
                tqdm.write("ОШИБКА: Для выполнения экшена должен быть открыт документ."); sys.exit(1)
            try:
                tqdm.write(f"Запуск экшена '{config.ps_action_name}' из набора '{config.ps_action_set}'...")
                app.doAction(config.ps_action_name, config.ps_action_set)
                tqdm.write("Экшен успешно выполнен.")
            except Exception as e:
                tqdm.write(f"Ошибка при выполнении экшена: {e}"); sys.exit(1)
        else:
            tqdm.write("Выполнение экшена пропущено согласно флагу '__execute_action'.")
    else:
        # Интерактивный режим
        tqdm.write("Интерактивный режим: вызов диалога выбора...")
        if not PYSIDE_AVAILABLE:
            tqdm.write("ОШИБКА: PySide6 не найдена, интерактивный режим невозможен."); sys.exit(1)
            
        actions_data = get_all_actions_js(app)
        if not actions_data:
            tqdm.write("Не удалось получить список экшенов или он пуст."); sys.exit(1)
            
        q_app = QApplication.instance() or QApplication(sys.argv)
        dialog = ActionSelectorDialog(actions_data, config.ps_action_set, config.ps_action_name)
        
        if dialog.exec() == QDialog.Accepted:
            final_set, final_action = dialog.get_selection()
            if not final_set or not final_action:
                tqdm.write("Выбор не сделан. Операция отменена."); sys.exit(0)
            
            tqdm.write(f"Выбран набор: '{final_set}', экшен: '{final_action}'")

            if config.__save_set_to_context and IS_MANAGED_RUN:
                pysm_context.set("ps_action_set", final_set)
                tqdm.write(f"Набор '{final_set}' сохранен в контекст как 'ps_action_set'.")
            
            if config.__save_action_to_context and IS_MANAGED_RUN:
                pysm_context.set("ps_action_name", final_action)
                tqdm.write(f"Экшен '{final_action}' сохранен в контекст как 'ps_action_name'.")

            if config.__execute_action:
                if len(app.documents) == 0:
                    tqdm.write("ОШИБКА: Для выполнения экшена должен быть открыт документ."); sys.exit(1)
                try:
                    tqdm.write(f"Запуск экшена '{final_action}' из набора '{final_set}'...")
                    app.doAction(final_action, final_set)
                    tqdm.write("Экшен успешно выполнен.")
                except Exception as e:
                    tqdm.write(f"Ошибка при выполнении экшена: {e}"); sys.exit(1)
            else:
                tqdm.write("Выполнение экшена пропущено согласно настройкам.")
        else:
            tqdm.write("Операция отменена пользователем."); sys.exit(0)

    tqdm.write("--- Скрипт выбора и запуска экшена: Успешное завершение ---")
    sys.exit(0)

# 5. БЛОК: Точка входа
# ==============================================================================
if __name__ == "__main__":
    main()