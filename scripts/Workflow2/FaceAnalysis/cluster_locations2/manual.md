# Руководство по скрипту "Кластеризация/классификация фотографий по локациям"

Версия описания: v2.2026.05.08

## 1. Назначение

Скрипт `cluster_locations2` анализирует фотографии и присваивает им метку локации или сюжета на основе визуального содержимого. Для получения признаков используется CLIP или SigLIP 2 ONNX.

Поддерживаются два режима:

1. `clustering` - автоматическая группировка похожих изображений без заранее заданных названий локаций.
2. `classification` - zero-shot классификация изображений по заранее заданным текстовым описаниям локаций.

Результаты записываются в `info_faces.json` и дополнительно сохраняются в папку `_Embeddings`.

## 2. Требования к данным

В папке анализа, переданной через `a_cl_data_dir`, должны быть:

* `info_faces.json` - основной JSON с данными по фотографиям.
* `JPG/` - папка с оригинальными изображениями.
* `JPG/Masks/` - папка с масками, если анализ выполняется по маскам.

Поддерживаемые расширения входных изображений: `.jpg`, `.jpeg`, `.png`.

Если используется режим масок, имя файла маски должно заканчиваться суффиксом `a_cl_mask_suffix` / `model_params.mask_suffix`. По этому суффиксу скрипт находит соответствующий оригинал в папке `JPG`.

## 3. Режимы работы

### 3.1 Clustering

Режим `clustering` строит image embeddings для всех выбранных изображений и группирует их алгоритмом DBSCAN.

В `info_faces.json` для найденных фотографий обновляются поля:

* `location_cluster` - числовая метка кластера.
* `location_name` - строковое представление метки.

Метка `-1` означает, что изображение не попало ни в один устойчивый кластер.

### 3.2 Classification

Режим `classification` сравнивает image embeddings с text embeddings, построенными по списку `a_cl_location_prompts`.

Для каждой фотографии выбирается наиболее близкий prompt. Если score ниже `match_threshold`, фотография получает метку `-1`.

В `info_faces.json` обновляются поля:

* `location_cluster` - индекс выбранного prompt или `-1`.
* `location_name` - строковое представление метки.
* `location_score` - similarity score лучшего совпадения.
* `location_prompt` - текст выбранного prompt или `null`, если метка `-1`.

Дополнительно в `_Embeddings/location_prompts.json` сохраняется соответствие индексов prompt-ам.

## 4. Основные параметры запуска

* `mode` - режим работы: `clustering` или `classification`.
* `model_backend` - backend модели: `siglip2_onnx` или `clip`.
* `spatial_strategy` - способ сжатия spatial embeddings для `siglip2_onnx` в режиме `clustering`.
* `a_cl_data_dir` - папка анализа с `info_faces.json` и `JPG`.
* `a_cl_config_file` - путь к `config.toml`.
* `use_originals` - использовать оригиналы из `JPG` вместо масок из `JPG/Masks`.
* `a_cl_location_prompts` - список текстовых описаний локаций для `classification`.
* `match_threshold` - минимальный score для принятия результата классификации; значение по умолчанию `0.11`.
* `a_cl_cluster_eps` - параметр `eps` для DBSCAN.
* `a_cl_mask_suffix` - суффикс имени файлов масок.
* `all_threads` - число потоков для параллельной загрузки изображений; значение по умолчанию `12`, `0` означает автоматическое значение.
* `batch_size` - размер batch для обработки изображений моделью.
* `cache_mode` - режим работы кеша: `use`, `refresh`, `off`.

`batch_size`, `all_threads` и `input_size` валидируются в начале pipeline. Некорректные значения приводят к `PipelineError`.

## 5. Конфигурация модели

Основные параметры находятся в `config.toml`.

Секция `[model]`:

* `backend = "clip"` - CLIP через ONNX Runtime.
* `backend = "siglip2_onnx"` - SigLIP 2 SO400M через ONNX Runtime.
* `name` - имя модели для fingerprint и диагностики.

Секция `[paths]`:

* `model_root` - корневая папка моделей.

Секция `[clip]`:

* `model_onnx` - путь к ONNX-файлу CLIP относительно `model_root`.
* `tokenizer_path` - путь к tokenizer CLIP относительно `model_root`.
* `input_size` - размер входа CLIP, обычно `[224, 224]`.

Для CLIP в обычном runtime используется lightweight tokenizer по локальным файлам `vocab.json` и `merges.txt`. Он не импортирует `transformers` и `torch`. `transformers.CLIPTokenizer` используется только как fallback для первичного скачивания tokenizer-файлов, если локального tokenizer ещё нет.

Секция `[siglip2_onnx]`:

* `model_dir` - папка ONNX-модели SigLIP 2 относительно `model_root`.
* `tokenizer_path` - путь к tokenizer SigLIP 2 относительно `model_root`.
* `input_size` - размер входа SigLIP 2, `[384, 384]`.
* `spatial_strategy` - стратегия spatial embeddings для `clustering`.

Секция `[model_params]`:

* `mask_suffix` - суффикс файлов масок.

`input_size` выбирается автоматически из секции активного backend. Если переключить backend через `model_backend`, вручную менять `input_size` не нужно.

Важно: препроцессинг изображений выполняется внутри адаптера выбранной модели и включён в fingerprint кеша, поэтому старый image cache автоматически инвалидируется при изменении этой логики.

### 5.1 SigLIP 2 ONNX

Backend `siglip2_onnx` использует модель `siglip2-so400m-patch14-384` через ONNX Runtime:

* используется `vision_model.onnx` от `siglip2-so400m-patch14-384`;
* для `clustering` берётся `last_hidden_state`, после чего patch-признаки сжимаются через `spatial_strategy`;
* для `classification` используются compact pooled image/text embeddings размером `1152`.
* если для vision-модели выбран `TensorrtExecutionProvider`, text-модель всё равно запускается через `CUDAExecutionProvider`, а при его отсутствии через `CPUExecutionProvider`, потому что TensorRT нестабилен для динамической формы `input_ids`.

Для `grid_9x9` рабочий диапазон `a_cl_cluster_eps` на тестовой сессии был примерно `0.18-0.19`. Для других сессий и стратегий eps нужно подбирать отдельно.

Поддерживаемые `spatial_strategy`:

* `flatten_axis1_norm` - старый полный spatial descriptor, около `839808` чисел на фото.
* `grid_9x9` - сжатая spatial-сетка, около `93312` чисел на фото.
* `grid_6x6` - сжатая spatial-сетка, около `41472` чисел на фото.
* `grid_3x3` - сжатая spatial-сетка, около `10368` чисел на фото.
* `mean_std` - очень компактный descriptor, около `2304` чисел на фото.

Файлы модели должны лежать в локальной папке проекта:

```text
_BIN/models/SigLIP2/siglip2-so400m-patch14-384-ONNX/
├── vision_model.onnx
├── text_model.onnx
└── text_model.onnx_data
```

Если нужных файлов нет, `siglip2_onnx` скачивает их автоматически в эти локальные папки проекта. Загрузка выполняется с progress bar PySM по байтам:

* для `clustering` сначала требуется только `vision_model.onnx`;
* для `classification` дополнительно скачиваются `text_model.onnx`, `text_model.onnx_data` и tokenizer;
* tokenizer сохраняется в `_BIN/models/tokenizer/siglip2-so400m-patch14-384/`;
* `transformers` и `torch` для загрузки и токенизации не используются.

Минимальный конфиг:

```toml
[model]
backend = "siglip2_onnx"
name = "SigLIP2 SO400M ONNX spatial"

[model_params]
input_size = [384, 384]

[siglip2_onnx]
model_dir = "models/SigLIP2/siglip2-so400m-patch14-384-ONNX"
vision_model = "vision_model.onnx"
image_output = "last_hidden_state"
text_model = "text_model.onnx"
tokenizer_path = "models/tokenizer/siglip2-so400m-patch14-384"
input_size = [384, 384]
spatial_strategy = "grid_9x9"
```

Backend и spatial strategy можно переопределить из командной строки/PySM без редактирования `config.toml`:

```powershell
--model_backend siglip2_onnx --spatial_strategy grid_9x9
```

Для возврата к CLIP:

```powershell
--model_backend clip
```

При смене `model_backend` или `spatial_strategy` используйте `cache_mode=refresh` для первого запуска, чтобы пересчитать embeddings новым способом.

## 6. Cache system

Кеш хранится в папке `_Cache` внутри `a_cl_data_dir`.

### 6.1 Image cache

Пути:

* `_Cache/image_clustering/` - image embeddings для режима `clustering`.
* `_Cache/image_classification/` - image embeddings для режима `classification`.

Содержит:

* `embeddings.npy` - матрица image embeddings.
* `index.json` - соответствие имени файла строке в матрице.
* `manifest.json` - паспорт кеша текущей сессии.

`manifest.json` нужен, чтобы безопасно понять, можно ли использовать существующий `embeddings.npy` для текущего запуска.

В manifest входят:

* версия схемы кеша;
* fingerprint модели;
* версия image preprocessing;
* `input_size`;
* режим входных данных: оригиналы или маски;
* `mask_suffix`;
* список файлов и их сигнатуры;
* hash manifest.

Image cache пересчитывается, если изменились модель, tokenizer, input size, preprocessing, режим originals/masks, mask suffix, состав/порядок/состояние файлов.

Кеши кластеризации и классификации разделены намеренно. Для `siglip2_onnx` в режиме `clustering` используются spatial embeddings, а в режиме `classification` - compact pooled embeddings. Эти матрицы имеют разный смысл и размер, поэтому не должны перезаписывать друг друга при переключении режима работы. Старая папка `_Cache/image/`, если она уже была создана прежними версиями скрипта, считается legacy cache и новой версией не используется.

### 6.2 Text cache

Путь: `_Cache/text/`

Содержит:

* `embeddings.npy` - матрица text embeddings.
* `meta.json` - fingerprint модели и ordered список prompts.

Порядок prompts является частью cache identity, потому что индексы prompts используются как значения `location_cluster`. Если поменять порядок одинаковых prompt-строк, text cache будет пересчитан.

### 6.3 cache_mode

`cache_mode` управляет чтением и записью кеша:

* `use` - использовать кеш, если manifest/meta совпадает; иначе пересчитать и сохранить.
* `refresh` - принудительно пересчитать embeddings и перезаписать кеш.
* `off` - пересчитать embeddings без чтения и записи кеша.

## 7. Выходные файлы

Скрипт обновляет:

* `info_faces.json`

Скрипт создаёт или перезаписывает:

* `_Embeddings/location_embeddings_clustering.npy` - image embeddings последнего запуска `clustering`.
* `_Embeddings/location_index_clustering.json` - индекс embeddings последнего запуска `clustering`.
* `_Embeddings/location_embeddings_classification.npy` - image embeddings последнего запуска `classification`.
* `_Embeddings/location_index_classification.json` - индекс embeddings последнего запуска `classification`.
* `_Embeddings/location_prompts.json` - только для `classification`

`info_faces.json` сохраняется через временный файл с последующей заменой, чтобы снизить риск повреждения JSON при аварийном завершении записи.

## 8. Рекомендации по prompts

Для `siglip2_onnx` можно использовать русские prompts. Обычно лучше работают короткие, конкретные описания локации или сюжета.

Примеры:

* `a photograph of a classroom with school desks`
* `a photograph of a classroom with a blackboard`
* `a photograph of a school gymnasium`
* `a photograph of a school hallway`
* `a photograph of a schoolyard with trees`

В интерфейсе PySM каждый prompt вводится отдельной строкой. Кавычки вокруг prompt не нужны.

## 9. Важные изменения в текущей версии

В этой версии pipeline был стабилизирован:

* image cache переведён на session manifest cache.
* Image cache разделён по режимам: `_Cache/image_clustering/` и `_Cache/image_classification/`.
* Итоговые файлы embeddings в `_Embeddings` разделены по режимам, чтобы spatial и pooled embeddings не перезаписывали друг друга.
* Добавлен `cache_mode`: `use`, `refresh`, `off`.
* Text cache учитывает порядок prompts.
* В fingerprint добавлены параметры модели и версия image preprocessing.
* Добавлен backend `siglip2_onnx` для `siglip2-so400m-patch14-384`.
* Для `clustering` SigLIP 2 ONNX использует spatial embeddings, для `classification` - compact pooled embeddings.
* Classification записывает `location_score`, `location_prompt` и `location_prompts.json`.
* CLIP image preprocessing приведён к стандартным mean/std.
* `workers` используется для параллельной загрузки изображений внутри batch.
* `info_faces.json` сохраняется через временный файл.
* Ошибки pipeline переводятся в `PipelineError`.
* Удалён старый дублирующий путь `EmbeddingService`.
* CLIP-specific tensor mapping вынесен в отдельный `ClipModelLoader`.

## 10. Ограничения

* Поддерживаются backend-ы `clip` и `siglip2_onnx`.
* `siglip2_onnx` требует локальные файлы ONNX-модели и tokenizer в `_BIN`.
* При смене `mode`, `spatial_strategy`, модели или набора файлов image cache пересчитывается.
* Полный пересчёт image cache выполняется при любом изменении manifest; частичный incremental update не используется.
