# 1. БЛОК: Импорты
# ==============================================================================
import argparse
import json
import sys
from argparse import Namespace
from typing import Dict, List

# Попытка импорта библиотек из экосистемы PySM.
try:
    from pysm_lib import pysm_context # pysm_context не используется, но оставляем для унификации
    from pysm_lib.pysm_context import ConfigResolver
    from pysm_lib.pysm_progress_reporter import tqdm
    IS_MANAGED_RUN = True
except ImportError:
    IS_MANAGED_RUN = False
    pysm_context = None
    ConfigResolver = None
    # Заглушка для tqdm
    class TqdmWriteMock:
        @staticmethod
        def write(msg, *args, **kwargs):
            print(msg)
    tqdm = TqdmWriteMock()

# Импорт ключевых зависимостей
try:
    from photoshop import api
    from comtypes import COMError
except ImportError:
    tqdm.write("ОШИБКА: Библиотеки 'photoshop-python-api' и 'comtypes' не найдены.")
    sys.exit(1)


# 2. БЛОК: Получение конфигурации (ОБНОВЛЕННЫЙ ПАТТЕРН)
# ==============================================================================
def get_config() -> Namespace:
    """
    Определяет аргументы скрипта и получает их значения с помощью ConfigResolver.
    """
    parser = argparse.ArgumentParser(description="Удаляет набор экшенов из палитры.")
    parser.add_argument(
        "--ps_action_set_name", type=str, required=True,
        help="Точное имя набора экшенов для удаления. Поддерживает шаблоны."
    )
    if IS_MANAGED_RUN and ConfigResolver:
        resolver = ConfigResolver(parser)
        return resolver.resolve_all()
    else:
        return parser.parse_args()


# 3. БЛОК: Вспомогательные функции
# ==============================================================================
def get_all_actions_js(app: api.Application) -> Dict[str, List[str]]:
    """
    Выполняет JavaScript в Photoshop для получения полного списка
    всех наборов и вложенных в них экшенов.
    """
    javascript_code = """
    function getAllActions() {
        var sets = [];
        var i = 1;
        while (true) {
            var ref = new ActionReference();
            ref.putIndex(stringIDToTypeID("actionSet"), i);
            var desc;
            try {
                desc = executeActionGet(ref);
            } catch (e) {
                break;
            }
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
            if (actions.length > 0) {
                 sets.push({ "name": setName, "actions": actions });
            }
            i++;
        }
        return JSON.stringify(sets);
    }
    getAllActions();
    """
    try:
        json_result = app.eval_javascript(javascript_code)
        list_of_sets = json.loads(json_result)
        actions_data = {item['name']: item['actions'] for item in list_of_sets}
        return actions_data
    except Exception as e:
        tqdm.write(f"Ошибка при получении списка экшенов: {e}")
        return {}

def delete_action_set_js(app: api.Application, set_name: str) -> bool:
    """Надежно удаляет набор экшенов через JavaScript."""
    print(f"Попытка удалить набор экшенов <i>{set_name}</i>...")
    js_code = f"""
    function deleteActionSet(setName) {{
        try {{
            var ref = new ActionReference();
            ref.putName(stringIDToTypeID("actionSet"), setName);
            var desc = new ActionDescriptor();
            desc.putReference(stringIDToTypeID("null"), ref);
            executeAction(stringIDToTypeID("delete"), desc, DialogModes.NO);
            return true;
        }} catch (e) {{ return false; }}
    }};
    deleteActionSet({json.dumps(set_name)});
    """
    try:
        result = app.eval_javascript(js_code)
        return str(result).lower() == 'true'
    except Exception as e:
        tqdm.write(f"\nКритическая ошибка при удалении набора экшенов:")
        tqdm.write(f"{e}")
        return False

# 4. БЛОК: Главная функция-оркестратор
# ==============================================================================
def main():
    # Получаем готовую конфигурацию одним вызовом
    config = get_config()
    action_set_name = config.ps_action_set_name

    print("<b>Удаление набора экшенов из палитры Photoshop</b>\n")

    try:
        print("Подключение к Adobe Photoshop...")
        app = api.Application()
        print("Подключение выполнено успешно.")
    except COMError:
        tqdm.write("ОШИБКА: Не удалось подключиться к Photoshop.")
        sys.exit(1)

    print(f"\nПроверка существования набора <i>{action_set_name}</i>...")
    all_sets = get_all_actions_js(app)

    if action_set_name not in all_sets:
        print(f"Удаление не требуется, набор <i>{action_set_name}</i> отсутствует.\n")
        sys.exit(0)

    print(f"Набор <i>{action_set_name}</i> найден. Удаление...")
    if delete_action_set_js(app, action_set_name):
        tqdm.write(f"\n<b>Набор <i>{action_set_name}</i> успешно удален.</b>\n")
        sys.exit(0)
    else:
        tqdm.write(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось удалить набор, несмотря на его наличие.")
        sys.exit(1)

# 5. БЛОК: Точка входа
# ==============================================================================
if __name__ == "__main__":
    main()