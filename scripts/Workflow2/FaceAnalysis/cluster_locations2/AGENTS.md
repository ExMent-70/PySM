# AGENTS.md для cluster_locations2

Локальные инструкции для скрипта:

```text
scripts/Workflow2/FaceAnalysis/cluster_locations2
```

## Назначение

`cluster_locations2` анализирует изображения фотосессии и присваивает им метки локаций/сюжетов.

Режимы:

* `clustering` - группировка похожих локаций через DBSCAN.
* `classification` - zero-shot классификация по текстовым prompts.

## Текущий Backend

* Сейчас поддерживаются `clip` и `siglip2_onnx`.
* Основной SigLIP 2 backend использует ONNX Runtime и модель `siglip2-so400m-patch14-384`.
* В режиме `clustering` `siglip2_onnx` использует spatial image embeddings.
* В режиме `classification` `siglip2_onnx` использует compact pooled image/text embeddings.
* Для `siglip2_onnx` text-модель не должна запускаться через TensorRT; использовать CUDA, а если CUDA недоступен, CPU.
* CLIP-specific ONNX tensor mapping находится в `infrastructure/models/clip/clip_model_loader.py`.
* CLIP runtime tokenizer находится в `infrastructure/models/clip/clip_light_tokenizer.py` и не должен импортировать `transformers`/`torch`.
* `transformers.CLIPTokenizer` допустим только как fallback для первичного скачивания локальных `vocab.json` и `merges.txt`.
* Общий ONNX session loader находится в `infrastructure/model_loader.py`.
* SigLIP 2 ONNX adapter находится в `infrastructure/models/siglip2_onnx/siglip2_onnx_model.py`.

## Cache

* Image cache режима `clustering` хранится в `_Cache/image_clustering/`.
* Image cache режима `classification` хранится в `_Cache/image_classification/`.
* Image cache использует session-level `manifest.json`.
* Text cache хранится в `_Cache/text/`.
* Порядок prompts является частью identity text cache.
* `cache_mode`: `use`, `refresh`, `off`.
* `_Embeddings` stores mode-specific image embeddings: `location_embeddings_clustering.npy` and `location_embeddings_classification.npy`.
* Не возвращаться к старому partial-cache подходу без отдельного обсуждения.
* SigLIP 2 ONNX модель и tokenizer должны лежать в локальных папках `_BIN/models/SigLIP2/...` и `_BIN/models/tokenizer/...`.
* Если SigLIP 2 ONNX файлы отсутствуют, downloader скачивает их в локальные папки проекта с progress bar PySM.
* Downloader не должен использовать `transformers` или `torch`; для скачивания используется `requests`, для токенизации пакет `tokenizers`.

## Результаты Classification

В режиме `classification` скрипт пишет:

* `location_cluster`
* `location_name`
* `location_score`
* `location_prompt`

Также создаётся `_Embeddings/location_prompts.json`.

## Важные Правила Правок

* Ошибки pipeline должны проходить через `PipelineError`.
* `info_faces.json` сохранять через временный файл и replace.
* CLIP image preprocessing использует стандартные CLIP mean/std.
* SigLIP 2 ONNX image preprocessing использует RGB и нормализацию `(x - 0.5) / 0.5`.
* Версия image preprocessing включена в model fingerprint.
* Не запускать полный pipeline на реальных данных без подтверждения пользователя.
* Если меняется поведение, обновлять `manual.md`.
* Если меняются параметры запуска, обновлять `script_passport.json`, `config.toml` и `manual.md`.

## Проверка

Минимальная проверка после изменений:

```powershell
python -m compileall scripts/Workflow2/FaceAnalysis/cluster_locations2
```

Перед коммитом:

```powershell
git diff --check -- scripts/Workflow2/FaceAnalysis/cluster_locations2
git status --short
```
