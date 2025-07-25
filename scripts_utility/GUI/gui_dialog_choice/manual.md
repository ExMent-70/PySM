# Руководство по "Диалогу выбора из списка" (v1.0)

## 1. Назначение

Скрипт **"Диалог выбора из списка"** позволяет создать точку ветвления в вашем рабочем процессе, запрашивая у пользователя выбор одной из нескольких предопределенных опций. Это идеальный инструмент для сценариев, где нужно выбрать режим обработки, тип отчета или любое другое строковое значение из заданного набора.

## 2. Логика работы

1.  **Получает параметры:** Скрипт принимает на вход имя переменной для результата, заголовок, сообщение и, самое главное, список вариантов для выбора.
2.  **Показывает диалог:** Открывается стандартное системное окно с выпадающим списком, содержащим все предоставленные варианты.
3.  **Обрабатывает выбор:**
    -   Если пользователь выбирает один из вариантов и нажимает "ОК", скрипт сохраняет выбранную строку в Контекст Коллекции под именем, указанным в `dlg_choice_var`, и завершается с **кодом `0` (Успех)**.
    -   Если пользователь нажимает "Отмена" или закрывает окно, **никакие данные не сохраняются**, и скрипт завершается с **кодом `1` (Отмена)**, немедленно останавливая всю цепочку.

## 3. Параметры

-   **Имя переменной результата** (`dlg_choice_var`):
    -   **Назначение:** Имя переменной, в которую будет сохранен выбор пользователя.
    -   **Пример:** `processing_mode`.

-   **Заголовок окна** (`dlg_choice_title`):
    -   **Назначение:** Текст в заголовке диалогового окна.
    -   **Пример:** `Выбор режима обработки`.

-   **Текст-приглашение** (`dlg_choice_message`):
    -   **Назначение:** Сообщение, которое видит пользователь над выпадающим списком.
    -   **Пример:** `Выберите, как обработать фотографии:`.

-   **Список вариантов** (`dlg_choice_list`):
    -   **Назначение:** Это основной параметр, определяющий содержимое выпадающего списка. Каждый вариант должен быть на новой строке.
    -   **Пример:**
        ```
        Быстрая обработка
        Качественная обработка
        Только экспорт в JPG
        ```

-   **Значение по умолчанию** (`dlg_choice_dvalue`):
    -   **Назначение:** Позволяет задать, какой из вариантов будет выбран при открытии окна.
    -   **Приоритет:** Это значение будет использовано, **только если** в Контексте Коллекции еще нет значения для переменной, указанной в `dlg_choice_var`. Если в контексте уже есть подходящее значение, оно будет выбрано в первую очередь.
    -   **Пример:** `Качественная обработка`.

## 4. Пример использования

**Задача:** Перед началом обработки спросить у пользователя, какой тип водяного знака накладывать на фотографии.

1.  Добавляем в цепочку скрипт **"Диалог выбора из списка"**.
2.  Настраиваем параметры:
    -   `Имя переменной результата`: `watermark_type`
    -   `Заголовок`: `Выбор водяного знака`
    -   `Список вариантов`:
        ```
        Без водяного знака
        Маленький в углу
        Большой по центру
        ```
    -   `Значение по умолчанию`: `Маленький в углу`
3.  Следующий скрипт в цепочке может теперь использовать переменную `{watermark_type}`, чтобы применить соответствующую логику.