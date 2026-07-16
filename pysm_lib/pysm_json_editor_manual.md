# API визуального редактора JSON-переменных PySM

Модуль `pysm_lib.pysm_json_editor` предоставляет общий визуальный редактор
контекстных переменных типа `json`. Он поддерживает обычные имена переменных,
dotted-пути к вложенным объектам и массивам, схемы `<var_name>__schema`, темы и
иконки PySM.

`pysm_lib.pysm_json_editor` является стабильным публичным фасадом. Внутренняя
Qt-реализация разделена по назначению в пакете `pysm_lib.gui.json_editor`:

- `model.py` - дерево JSON и `JsonModel`;
- `delegates.py` - редакторы значений и диалог создания элемента;
- `window.py` - окно, работа с контекстом и lifecycle API.

## Блокирующий вызов

```python
from pysm_lib.pysm_json_editor import edit_json_variable

result = edit_json_variable(
    "project.settings",
    title="Настройки проекта",
    message="Проверьте параметры перед продолжением.",
)

if result.saved:
    settings = result.value
```

`edit_json_variable()` показывает окно и ожидает его закрытия, но не вызывает
`sys.exit()` и не завершает общий `QApplication`. Поэтому функцию можно вызывать
как из отдельного PySM-скрипта, так и из уже работающего Qt-интерфейса. Вызов
должен выполняться в GUI-потоке Qt.

Аргументы:

- `var_name` - имя JSON-переменной или dotted-путь к объекту/массиву;
- `title` - заголовок окна;
- `message` - HTML-текст над деревом;
- `context` - необязательный совместимый объект контекста; по умолчанию
  используется `pysm_lib.pysm_context`;
- `parent` - необязательное родительское Qt-окно;
- `apply_theme` - применять ли тему PySM к `QApplication`.

## Результат

Функция возвращает `JsonEditorResult` со следующими полями:

- `status` - `SAVED`, `UNCHANGED` или `CANCELLED`;
- `var_name` - отредактированное имя или dotted-путь;
- `value` - итоговое значение в окне;
- `changed` - были ли данные изменены и записаны;
- `saved` - вычисляемый признак подтверждения данных пользователем.

При ошибке загрузки API выбрасывает `ValueError` или исключение контекстного
слоя. Вызывающий скрипт самостоятельно решает, как показать или записать ошибку.

## Неблокирующее окно

Для интеграции в существующий GUI можно получить настроенное окно без запуска
локального event loop:

```python
from pysm_lib.pysm_json_editor import create_json_editor

window = create_json_editor("project")
window.finished.connect(handle_result)
window.show()
```

При использовании `create_json_editor()` вызывающий код отвечает за наличие
`QApplication` и хранение ссылки на окно до его закрытия.
