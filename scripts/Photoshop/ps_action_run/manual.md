# Руководство по использованию скрипта "Запуск Экшена" (v1.05.03)

## 1. Назначение

Этот скрипт является мощным инструментом автоматизации, который позволяет программно запускать предварительно записанные экшены (Actions) в Adobe Photoshop.

Он выполняет команду `doAction`, которая находит и выполняет указанный экшен для текущего активного документа.

## 2. Параметры

**Внимание:** Оба параметра являются обязательными и чувствительными к регистру и пробелам. Имена должны точно совпадать с теми, что отображаются в палитре "Actions" в Photoshop.

-   **Набор экшенов** (`ps_action_set`):
    -   **Тип:** Строка
    -   **Обязательный:** Да
    -   **Описание:** Имя "папки" (набора), в которой находится ваш экшен.

-   **Имя экшена** (`ps_action_name`):
    -   **Тип:** Строка
    -   **Обязательный:** Да
    -   **Описание:** Имя конкретного экшена внутри набора, который необходимо запустить.

### Пример

Если в палитре Actions у вас такая структура:
-   Набор: `My Filters`
    -   Экшен: `Apply Sharpen`

То параметры должны быть следующими:
-   `ps_action_set`: `My Filters`
-   `ps_action_name`: `Apply Sharpen`

## 3. Логика работы

1.  Скрипт считывает обязательные параметры: имя набора и имя экшена.
2.  Подключается к запущенному приложению Adobe Photoshop.
3.  Проверяет, открыт ли в Photoshop хотя бы один документ. Если нет — работа завершается с ошибкой.
4.  Отправляет команду на выполнение экшена (`app.doAction`).
5.  Если Photoshop не может найти экшен с указанными именами или если во время выполнения экшена происходит внутренняя ошибка (например, не выполнены условия для его работы), скрипт завершается с ошибкой.

---
## 4. Ответственность пользователя (ВАЖНО)

Скрипт является только **исполнителем**. Он не анализирует содержимое экшена и не проверяет его применимость к текущему документу.

**Вся ответственность за:**
-   Правильное написание имен набора и экшена.
-   Последствия выполнения экшена (изменение, сохранение или удаление данных).
-   Подготовку документа к выполнению экшена (например, выбор нужного слоя, снятие блокировок).

**...полностью лежит на пользователе.** Перед использованием скрипта в рабочих процессах убедитесь, что вы точно знаете, что делает указанный экшен.
---

## 5. Коды выхода

-   **Код выхода `0` (Успех):**
    -   Экшен был успешно найден и выполнен.
-   **Код выхода `1` (Ошибка):** Произошла ошибка. Это может быть:
    -   Не удалось подключиться к Adobe Photoshop.
    -   В Photoshop нет открытых документов.
    -   Экшен или набор с указанным именем не найден.
    -   Произошла ошибка во время выполнения самого экшена.

    При получении кода `1` выполнение всей цепочки скриптов будет **остановлено**.