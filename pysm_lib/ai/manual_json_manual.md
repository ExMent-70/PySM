# Ручной AI JSON-обмен

`pysm_lib.ai` и `pysm_lib.gui.ai` нужны для сценария, в котором скрипт готовит
prompt, оператор вручную передаёт его в любой AI-чат, а затем вставляет или
открывает JSON-ответ. Пакеты не делают сетевых запросов, не выбирают
AI-провайдера и не хранят API-ключи.

Главное правило: общий API **готовит и проверяет транспортный JSON**, но
**никогда сам не меняет предметные данные скрипта**. Например, только
`photo_selection` решает, какие номера фотографий допустимы, а только
`list_create` решает, какие поля `info` разрешено обновлять.

## Что использовать в новом скрипте

Для обычного сценария достаточно двух сущностей:

```python
from pysm_lib.ai import AiJsonRequest
from pysm_lib.gui.ai import edit_ai_json_response
```

- `AiJsonRequest` описывает один AI-сценарий: шаблон prompt, справочные данные
  и функцию проверки ответа. Он находится в независимом от Qt пакете
  `pysm_lib.ai`.
- `edit_ai_json_response(request, parent)` открывает готовый модальный диалог
  из двух вкладок и возвращает `AiJsonDialogResult`. Он находится в
  `pysm_lib.gui.ai`.

Последовательность работы выглядит так:

1. Скрипт собирает только те справочные данные, которые AI вправе видеть.
2. Скрипт создаёт `AiJsonRequest` и передаёт свою функцию
   `response_validator`.
3. Общий диалог подставляет данные в шаблон, показывает prompt и копирует его
   в буфер обмена.
4. Оператор передаёт prompt в выбранный AI-сервис и вставляет ответ обратно.
5. Общий API извлекает JSON даже из ответа с поясняющим текстом или Markdown
   JSON-блоком, затем вызывает `response_validator`.
6. Только если проверка прошла, результат получает статус `VALIDATED`.
7. Вызывающий скрипт применяет уже проверенное значение `result.value`.

При закрытии окна или ошибке проверки статус будет `CANCELLED`; данные менять
нельзя.

## Что предоставляет PySM и что должен сделать скрипт

| Задача | `pysm_lib.ai` / `pysm_lib.gui.ai` | Код конкретного скрипта |
| --- | --- | --- |
| Шаблон prompt | Подставляет `{{TOKEN}}`, сериализует списки и словари в JSON | Определяет текст шаблона и значения токенов |
| Исходный текст оператора | Даёт поле ввода, preview и копирование в буфер | Может настроить подпись и placeholder поля |
| Ответ AI | Открывает UTF-8 JSON-файл или принимает вставленный текст, извлекает JSON из Markdown | Описывает требуемую структуру в prompt |
| Безопасность | Не пропускает отсутствующие токены и неверный JSON | Проверяет ID, типы, диапазоны, дубликаты и бизнес-правила |
| Изменение данных | Никогда не делает | Применяет `result.value` после `result.accepted` |

## Полный пример

Ниже независимый пример импорта коротких текстов для известных пользователей.
Он показывает весь код, который должен быть в предметном скрипте.

```python
from dataclasses import dataclass
from typing import Any

from pysm_lib.ai import (
    AiJsonRequest,
    get_json_array,
    require_json_object,
    require_json_object_item,
)
from pysm_lib.gui.ai import edit_ai_json_response


@dataclass
class User:
    user_id: str
    name: str
    note: str = ""


PROMPT_TEMPLATE = """Ты сопоставляешь свободный текст с известными пользователями.

Справочник:
{{USERS_JSON}}

Исходный текст оператора:
{{RAW_TEXT}}

Верни только JSON:
{
  "matched": [
    {"user_id": "U001", "source_name": "Имя из текста", "note": "Текст"}
  ],
  "unresolved": [
    {"source_name": "Неоднозначное имя", "reason": "Причина"}
  ]
}
"""


def build_user_reference(users: list[User]) -> list[dict[str, str]]:
    """Передать AI только ID и имя, без изменяемых данных приложения."""

    return [{"user_id": user.user_id, "name": user.name} for user in users]


def validate_ai_response(
    payload: Any,
    users: list[User],
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """Проверить ответ AI и вернуть безопасные изменения, но не применять их."""

    response = require_json_object(payload)
    matched = get_json_array(response, "matched")
    unresolved = get_json_array(response, "unresolved")

    known_ids = {user.user_id for user in users}
    updates: dict[str, str] = {}
    for index, raw_item in enumerate(matched):
        item = require_json_object_item(
            raw_item,
            field_name="matched",
            index=index,
        )
        user_id = str(item.get("user_id", "")).strip()
        if user_id not in known_ids:
            raise ValueError(f"matched[{index}] содержит неизвестный user_id: {user_id}.")
        if user_id in updates:
            raise ValueError(f"user_id {user_id} повторяется в matched.")

        note = item.get("note")
        if not isinstance(note, str):
            raise ValueError(f"matched[{index}].note должен быть строкой.")
        updates[user_id] = note.strip()

    # При необходимости можно так же строго проверить поля unresolved.
    return updates, unresolved


def import_notes_from_ai(parent_window, users: list[User]) -> None:
    request = AiJsonRequest(
        title="AI-импорт заметок",
        prompt_template=PROMPT_TEMPLATE,
        prompt_values={"USERS_JSON": build_user_reference(users)},
        raw_text_label="Неструктурированный текст с заметками:",
        raw_text_placeholder="Например: Анна — любит море; Борис — играет в шахматы.",
        response_validator=lambda payload: validate_ai_response(payload, users),
        show_success_message=False,
    )

    result = edit_ai_json_response(request, parent_window)
    if not result.accepted:
        return  # Ничего не менять: JSON не прошёл проверку или диалог закрыт.

    updates, unresolved = result.value
    users_by_id = {user.user_id: user for user in users}
    for user_id, note in updates.items():
        users_by_id[user_id].note = note

    # Здесь скрипт может показать unresolved или предложить ручную обработку.
    print(f"Обновлено: {len(updates)}; требует проверки: {len(unresolved)}")
```

В примере AI получает только `user_id` и имя. Даже если он вернёт придуманный,
неизвестный или повторный идентификатор, `validate_ai_response()` остановит
импорт до изменения `User.note`.

## Шаблон prompt

Шаблон — обычная UTF-8 строка или текстовый файл. Токен имеет вид `{{TOKEN}}`.
Значения из `prompt_values` подставляются так:

- `str` вставляется как есть — это подходит для `{{RAW_TEXT}}`;
- список, словарь, число, `bool` или `None` сериализуются в красивый JSON;
- отсутствующий токен вызывает ошибку до копирования prompt.

`{{RAW_TEXT}}` используется по умолчанию: диалог берёт значение из поля
оператора. Если нужен другой токен, передайте `raw_text_token="SOURCE"` в
`AiJsonRequest` и используйте `{{SOURCE}}` в шаблоне. Если исходный текст не
нужен вовсе, передайте `raw_text_token=None`.

Чтобы хранить шаблон рядом со скриптом, можно использовать:

```python
from pathlib import Path
from pysm_lib.ai import load_prompt_template

template = load_prompt_template(Path(__file__).parent / "ai_prompt_template.txt")
```

`load_prompt_template(..., default_text=..., create_default=True)` создаст
редактируемый файл с шаблоном при первом запуске.

## Правила для `response_validator`

Эта функция получает уже разобранный Python-объект JSON и должна:

1. Проверить корневой тип и обязательные поля.
2. Проверить, что все идентификаторы принадлежат текущим данным скрипта.
3. Проверить типы и диапазоны: строки, номера, даты, разрешённые значения.
4. Отклонить дубликаты, противоречия и небезопасные пути.
5. Вернуть компактное, типизированное значение для применения.
6. Не менять модель, файлы или контекст PySM внутри валидатора.

Для простых проверок формы ответа доступны:

```python
response = require_json_object(payload)
matched = get_json_array(response, "matched", required=True)
item = require_json_object_item(matched[0], field_name="matched", index=0)
```

Эти функции проверяют только JSON-форму. Они не знают, какие ID допустимы,
какие номера являются корректными и как применять данные.

## Если нужен собственный интерфейс

Готовый диалог не обязателен. Можно использовать только ядро:

```python
request = AiJsonRequest(...)
prompt = request.build_prompt(raw_text_from_your_widget)
# Показать или скопировать prompt своим способом.

validated_value = request.validate_response(response_text_from_your_widget)
# Применить validated_value своим способом.
```

`pysm_lib.gui.ai.create_ai_json_dialog(request, parent)` создаёт готовый
диалог без запуска вложенного event loop. Это удобно, если скрипт сам
управляет показом окон. Обычно достаточно более простого
`edit_ai_json_response()`.
