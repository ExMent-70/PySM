# Руководство по модулю `pysm_context.py`

## 1. Назначение

`pysm_context.py` предоставляет основной API для работы пользовательских скриптов PySM с общим контекстом выполнения.

В штатном режиме live-контекст хранится в сегменте общей памяти
`multiprocessing.shared_memory`. Связанный `.context.json` используется как
checkpoint/persistence-файл и не является live-transport между PySM и
скриптами.

Скрипты не должны напрямую читать shared memory или `.context.json`. Рабочий
контракт — объект `pysm_context`:

```python
from pysm_lib import pysm_context

session_name = pysm_context.get("wf_session_name")
pysm_context.set("wf_session_name", "Test_JPG_2")
```

Независимо от runtime backend переменные имеют единую модель:

```json
{
  "variable_name": {
    "type": "string",
    "value": "value",
    "description": "Описание переменной",
    "read_only": false,
    "choices": null
  }
}
```

`pysm_context` является экземпляром класса `PySMContext`.

## 2. Основные принципы

### 2.1. Runtime backend

Основной backend задаётся в корневом `config.toml`:

```toml
[runtime_context]
backend = "shared_memory"
shared_memory_min_size_mb = 16
shared_memory_max_size_mb = 128
shared_memory_overflow_policy = "fail_current_write"
checkpoint_policy = "after_each_script"
```

Доступны два режима:

- `shared_memory` — основной production-режим;
- `file` — диагностический fallback через `.context.json`.

В shared-memory сегменте хранится UTF-8 JSON payload с заголовком, версией
формата, generation counter, состоянием записи и CRC32 checksum. На Windows
чтение и запись между процессами синхронизируются named mutex.

`PySMContext` отслеживает generation counter. Если другой процесс обновил
контекст, следующий вызов чтения перезагружает локальный кэш из общей памяти.

### 2.2. Запись и `commit`

Семантика записи зависит от backend.

В режиме `shared_memory` вызовы `set`, `set_structured`, `update` и `remove`:

1. Обновляют локальный кэш.
2. Немедленно публикуют целый snapshot в shared memory.
3. Увеличивают generation counter.

Параметр `commit=True` не делает shared-memory запись «более немедленной» — она
и так выполняется при каждой мутации.

В режиме `file` сохраняется lazy commit:

1. Изменения накапливаются в локальном кэше.
2. `commit=True` принудительно записывает `.context.json` атомарно.
3. При нормальном завершении скрипта выполняется зарегистрированный через
   `atexit` вызов `commit()`.

```python
pysm_context.set("var", value, commit=True)
```

Важно: script-side `commit()` сохраняет текущий runtime store. Политикой
checkpoint из shared memory в `.context.json` управляет основной процесс PySM:

- `after_each_script` — checkpoint после каждого завершённого скрипта;
- `on_save_exit` — checkpoint при сохранении коллекции или штатном закрытии.

При `on_save_exit` файл `.context.json` может временно отставать от live-состояния.

### 2.3. IPC-синхронизация

Shared memory хранит данные, а управляющие сообщения в `stderr` уведомляют
основной процесс PySM об изменениях интерфейса и маршрутизации:

```text
PYSM_CONTEXT_UPDATE:{...}
```

Для маршрутизации выполнения:

```text
PYSM_ROUTING_CMD:{...}
```

Эти сообщения не заменяют runtime store и не должны разбираться пользовательскими
скриптами вручную.

### 2.4. Dot-notation

Модуль поддерживает доступ к вложенным JSON-структурам через точечную нотацию:

```python
pysm_context.get_structured("wf_school_info.city")
pysm_context.set_structured("wf_school_info.city", "Иркутск")
pysm_context.remove("wf_school_info.city")
```

В обновлённой версии также поддерживаются индексы списков:

```python
pysm_context.get_structured("template.override_labels.0")
pysm_context.set_structured("template.override_labels.0", "portrait_0")
pysm_context.remove("template.override_labels.0")
```

## 3. Инициализация контекста

При импорте модуля создаётся singleton:

```python
pysm_context = PySMContext()
```

Во время инициализации модуль:
1. Ищет служебные аргументы `--pysm-context-shm-name`,
   `--pysm-context-mode` и `--pysm-context-file` либо соответствующие переменные
   окружения `PYSM_CONTEXT_SHM_NAME`, `PYSM_CONTEXT_MODE` и `PYSM_CONTEXT_FILE`.
2. В режиме `shared_memory` подключается к уже созданному PySM сегменту по имени.
3. При file backend открывает `FileContextStore` для `.context.json`.
4. Удаляет служебные аргументы из `sys.argv`, чтобы они не попадали в CLI
   пользовательского скрипта.
5. Загружает начальный snapshot и запоминает generation backend.
6. Регистрирует `commit()` через `atexit` для штатного завершения.

Имя и размер shared-memory сегмента определяет основной процесс PySM. Скрипт
только подключается к существующему сегменту и не должен вызывать `unlink()`.

## 4. Формат переменной контекста

Стандартная переменная имеет вид:

```json
{
  "type": "json",
  "value": {
    "city": "Улан-Удэ"
  },
  "description": "Описание",
  "read_only": false,
  "choices": null
}
```

Поля:

- `type` — тип переменной.
- `value` — фактическое значение.
- `description` — описание.
- `read_only` — запрет изменения.
- `choices` — список допустимых значений для переменных типа `choice`.

## 5. Методы чтения

### 5.1. `get`

```python
get(key: str, default: Any = None) -> Any
```

Получает значение верхнеуровневой переменной.

Пример:

```python
value = pysm_context.get("wf_session_name")
```

Если переменная отсутствует, возвращается `default`.

Ограничение: `get()` не поддерживает dot-notation.

### 5.2. `get_structured`

```python
get_structured(key: str, default: Any = None) -> Any
```

Получает верхнеуровневое или вложенное значение.

Примеры:

```python
city = pysm_context.get_structured("wf_school_info.city")
tag = pysm_context.get_structured("template.override_labels.0")
```

Поддерживает:
- вложенные словари;
- индексы списков;
- смешанные пути вида `items.0.name`.

Если путь отсутствует, возвращает `default`.

### 5.3. `exists`

```python
exists(key: str) -> bool
```

Проверяет существование переменной или вложенного пути.

Главное отличие от `get()` и `get_structured()`:

```python
pysm_context.exists("some.path")
```

отличает отсутствующий ключ от существующего ключа со значением `None`.

Примеры:

```python
pysm_context.exists("wf_school_info.city")
pysm_context.exists("template.override_labels.0")
```

### 5.4. `get_variable`

```python
get_variable(key: str) -> Optional[Dict[str, Any]]
```

Возвращает полную модель верхнеуровневой переменной:

```python
{
  "type": "...",
  "value": ...,
  "description": "...",
  "read_only": false,
  "choices": null
}
```

Метод не применяет dot-notation.

### 5.5. `get_all`

```python
get_all() -> Dict[str, Any]
```

Возвращает все переменные в виде словаря:

```python
{
  "var_name": value
}
```

Метаданные переменных не возвращаются.

### 5.6. `get_schema`

```python
get_schema(key: str, default: Any = None) -> Any
```

Возвращает схему переменной по соглашению:

```text
<key>_schema
```

Пример:

```python
schema = pysm_context.get_schema("wf_school_info")
```

Фактически читает:

```python
pysm_context.get_structured("wf_school_info_schema")
```

Если схема отсутствует, возвращает `default`.

## 6. Методы записи

### 6.1. `set`

```python
set(key: str, value: Any, var_type: Optional[str] = None, commit: bool = False) -> None
```

Устанавливает значение верхнеуровневой переменной.

Если переменная существует:
- проверяется `read_only`;
- обновляется `value`;
- при наличии `var_type` обновляется `type`.

Если переменная отсутствует:
- создаётся новая переменная;
- тип определяется автоматически через `_infer_type_from_value`, если `var_type` не задан.

Пример:

```python
pysm_context.set("wf_session_name", "Школа")
```

### 6.2. `set_structured`

```python
set_structured(key_path: str, value: Any, commit: bool = False) -> None
```

Устанавливает значение по dot-notation.

Примеры:

```python
pysm_context.set_structured("wf_school_info.city", "Иркутск")
pysm_context.set_structured("template.override_labels.0", "portrait_0")
```

Поведение:
- если путь не содержит точку, работает как `set`;
- если базовая переменная отсутствует, создаёт JSON-переменную;
- если промежуточные dict-ключи отсутствуют, создаёт их;
- если путь проходит через list, индекс должен уже существовать;
- списки автоматически не расширяются.

Списки не расширяются намеренно, чтобы не создавать скрытых структур.

### 6.3. `update`

```python
update(update_dict: Dict[str, Any], commit: bool = False) -> None
```

Обновляет несколько верхнеуровневых переменных.

Пример:

```python
pysm_context.update({
    "var_a": 1,
    "var_b": "text"
})
```

Метод не применяет dot-notation к ключам словаря.

### 6.4. `remove`

```python
remove(keys_to_remove: Optional[Union[str, List[str]]] = None, commit: bool = False) -> None
```

Удаляет переменные или вложенные значения.

Примеры:

```python
pysm_context.remove("wf_session_name")
pysm_context.remove("wf_school_info.city")
pysm_context.remove("template.override_labels.0")
pysm_context.remove(["var_a", "var_b"])
```

Если `keys_to_remove is None`, удаляются все пользовательские переменные, кроме зарезервированных:

```python
pysm_info
pysm_set_instance_ids
pysm_next_script
```

Поддерживает:
- удаление верхнеуровневых переменных;
- удаление вложенных ключей словаря;
- удаление элементов списка по индексу.

После удаления вложенного значения отправляется IPC-обновление всего базового JSON-объекта.

## 7. Методы шаблонов и путей

### 7.1. `resolve_template`

```python
resolve_template(template_string: Optional[str]) -> str
```

Рекурсивно заменяет плейсхолдеры вида:

```text
{variable_name}
```

или:

```text
{json_var.key}
```

на значения из контекста.

Пример:

```python
path = pysm_context.resolve_template("{wf_session_path}/{wf_session_name}")
```

Защита от бесконечной рекурсии ограничена глубиной `max_depth = 10`.

### 7.2. `resolve_path`

```python
resolve_path(path_str: str) -> pathlib.Path
```

Преобразует путь в абсолютный.

Если путь относительный и в контексте есть:

```python
pysm_info["collection_dir"]
```

то путь разрешается относительно директории коллекции.

## 8. Маршрутизация выполнения

### 8.1. `set_next_script`

```python
set_next_script(instance_id: str, commit: bool = False) -> None
```

Отправляет команду перехода к другому экземпляру скрипта.

Пример:

```python
pysm_context.set_next_script("instance_id_123")
```

Фактическая команда отправляется в `stderr`:

```text
PYSM_ROUTING_CMD:{...}
```

Параметр `commit` оставлен для обратной совместимости и не используется.

### 8.2. `list_instances`

```python
list_instances() -> List[Dict[str, str]]
```

Возвращает список экземпляров текущего набора из переменной:

```python
pysm_set_instance_ids
```

## 9. Логирование HTML, изображений и ссылок

### 9.1. `log_image`

```python
log_image(
    image_path: Union[str, pathlib.Path],
    width: int = 300,
    align: str = "left",
    margin: int = 5,
    img_desc: Optional[str] = None,
)
```

Выводит изображение в лог PySM как base64 HTML-блок.

### 9.2. `log_link`

```python
log_link(
    url_or_path: str,
    text: Optional[str] = None,
    align: str = "left",
    margin: int = 5,
)
```

Выводит ссылку в лог PySM.

Если передан локальный путь, он преобразуется в `file://` URI.

### 9.3. `log_html`

```python
log_html(
    html_content: str,
    align: str = "left",
    margin: int = 5,
    padding: int = 10,
)
```

Выводит произвольный HTML в лог PySM.

Переносы строк заменяются на `<br>`, чтобы сообщение `PYSM_HTML_BLOCK` оставалось одной строкой.

## 10. Метаданные Photoshop

### 10.1. `get_available_metadata_fields`

```python
get_available_metadata_fields() -> List[str]
```

Возвращает список поддерживаемых XMP-полей.

### 10.2. `get_document_metadata`

```python
get_document_metadata(
    doc_path: Optional[str] = None,
    fields: Union[str, List[str]] = "__all__",
    clear_before_write: bool = False,
    prefix: str = "psd_meta_",
) -> Dict[str, Any]
```

Извлекает метаданные из документа Photoshop и сохраняет их в контекст.

Поддерживает:
- активный документ Photoshop;
- документ по пути;
- XMP-поля;
- системные параметры документа;
- очистку старых переменных перед записью.

## 11. ConfigResolver

`ConfigResolver` используется скриптами для получения параметров с учётом приоритетов:

1. Явно переданный CLI-аргумент.
2. Значение из контекста PySM.
3. Значение по умолчанию из `argparse`.

Пример:

```python
parser = argparse.ArgumentParser()
parser.add_argument("--source_dir", type=str, default="")
config = ConfigResolver(parser).resolve_all()
```

### 11.1. Обработка путей

Аргументы, содержащие в имени:

```text
path
dir
file
folder
```

автоматически разрешаются как пути.

### 11.2. Обработка шаблонов

Строковые значения проходят через:

```python
resolve_template()
```

Это позволяет использовать:

```text
{wf_session_path}/{wf_session_name}
```

### 11.3. Обработка списков

Если `argparse` ожидает `nargs="+"` или `nargs="*"`, строковое значение из контекста преобразуется в список строк по строкам.

## 12. Рекомендации по использованию

### 12.1. Для простых переменных

```python
pysm_context.get("var")
pysm_context.set("var", value)
```

### 12.2. Для вложенных JSON-структур

```python
pysm_context.get_structured("json_var.key")
pysm_context.set_structured("json_var.key", value)
```

### 12.3. Для проверки существования

```python
if pysm_context.exists("json_var.key"):
    ...
```

### 12.4. Для удаления

```python
pysm_context.remove("json_var.key")
```

### 12.5. Для схем

```python
schema = pysm_context.get_schema("json_var")
```

## 13. Ограничения

1. `set_structured()` не расширяет списки автоматически.
2. `update()` работает только с верхнеуровневыми переменными.
3. `get_variable()` возвращает только верхнеуровневую модель переменной.
4. `remove()` не удаляет зарезервированные переменные при массовой очистке.
5. `resolve_template()` приводит найденные значения к строке.
6. Размер shared-memory payload ограничен capacity текущего сегмента. Если
   script-side запись не помещается, текущая операция завершается ошибкой и не
   переключается молча на file backend.
7. Основной процесс может увеличить сегмент только между скриптами в пределах
   `shared_memory_max_size_mb`. Запущенный скрипт продолжает работать с тем
   сегментом, имя которого получил при старте.
8. При `checkpoint_policy = "on_save_exit"` `.context.json` может не содержать
   последние live-изменения до сохранения коллекции или штатного закрытия PySM.

При ошибках shared memory сначала проверять runtime-диагностику PySM: backend,
имя и размер сегмента, checkpoint file, payload size/usage ratio и текст
ошибки. Для временной диагностики можно выбрать `backend = "file"`, но это не
целевой production-режим.

## 14. Изменения в обновлённой версии

Добавлено:
- основной runtime backend `shared_memory`;
- абстракция `ContextStore` и диагностический `FileContextStore`;
- заголовок shared-memory payload с generation counter, состоянием записи и
  CRC32 checksum;
- Windows named mutex для межпроцессной синхронизации;
- автоматическое обновление локального кэша при смене generation;
- checkpoint policies `after_each_script` и `on_save_exit`;
- контролируемое увеличение сегмента между скриптами;
- telemetry размера payload и заполнения сегмента;
- `exists()`;
- `get_schema()`;
- поддержка индексов списков в `get_structured()`;
- поддержка индексов списков в `set_structured()`;
- поддержка индексов списков в `remove()`;
- более строгая обработка пустых путей;
- атомарная запись checkpoint JSON через временный файл и `os.replace`.

Сохранено:
- существующий публичный API;
- lazy commit для file backend;
- IPC-синхронизация;
- формат переменных контекста;
- поведение `get()` и `set()`.

Изменено:
- `.context.json` больше не является live-transport в shared-memory режиме;
- shared-memory мутации публикуются сразу независимо от `commit=False`;
- прямое чтение `.context.json` пользовательскими скриптами не является
  поддерживаемым runtime-контрактом.
