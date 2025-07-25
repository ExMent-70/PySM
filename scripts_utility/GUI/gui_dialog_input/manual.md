# Руководство по использованию скрипта "Пользовательский ввод" (v1.05.03)

## 1. Назначение

Скрипт **"Пользовательский ввод"** позволяет запросить у пользователя ввод текстовой информации прямо в процессе выполнения цепочки. Ключевой особенностью является **гибкая система валидации**, которая гарантирует получение данных в нужном формате.

## 2. Параметры

Скрипт имеет следующие настраиваемые параметры:

-   **Имя переменной результата** (`dlg_input_var`):
    -   Имя переменной, в которую будет сохранено значение.
-   **Текст-приглашение** (`dlg_input_msg`):
    -   Текст, который отображается над полем ввода.
-   **Заголовок окна** (`dlg_input_title`):
    -   Текст в заголовке окна.
-   **Значение по умолчанию** (`dlg_input_dvalue`):
    -   Текст, который будет вписан в поле ввода при его открытии.

### Система валидации

Валидация настраивается с помощью одного основного и двух вспомогательных параметров.

-   **Тип валидации** (`dlg_input_valid_type`):
    -   **Описание:** Это главный параметр, управляющий проверкой. Он представляет собой выпадающий список.
    -   **Варианты:**
        -   `none`: **(По умолчанию)** Валидация отключена. Пользователь может ввести любое значение, включая пустое.
        -   `not_empty`: Любой непустой текст.
        -   `integer`: Целое число (положительное, отрицательное или ноль).
        -   `positive_integer`: Положительное целое число или ноль.
        -   `float`: Число с плавающей точкой.
        -   `email`: Адрес электронной почты.
        -   `custom`: **Пользовательский режим**. При выборе этого варианта становятся активными два следующих поля.

-   **Пользовательский шаблон** (`dlg_input_custom_regexp`):
    -   **Описание:** Поле для ввода вашего собственного регулярного выражения. **Активно, только если "Тип валидации" установлен в `custom`**.

-   **Описание пользовательского шаблона** (`dlg_input_custom_regexp_desc`):
    -   **Описание:** Поле для ввода вашего текста сообщения об ошибке. **Активно, только если "Тип валидации" установлен в `custom`**.

## 3. Логика работы

### 3.1. Валидация
Если выбран любой тип валидации, кроме `none`, пользователь не сможет закрыть окно с кнопкой "ОК", пока не введет данные, соответствующие формату.

### 3.2. Управление потоком выполнения (коды выхода)

Поведение скрипта теперь полностью соответствует другим диалоговым окнам в PySM:

-   **Код выхода `0` (Успех):**
    -   **Условие:** Пользователь ввел корректные данные (если валидация включена) и нажал "ОК".
    -   **Действие:** Введенное значение сохраняется в Контекст Коллекции. Следующий скрипт в цепочке будет запущен.

-   **Код выхода `1` (Отмена):**
    -   **Условие:** Пользователь нажал кнопку "Отмена" или закрыл окно системной кнопкой (крестик).
    -   **Действие:** **Никакие данные в контекст не сохраняются.** Выполнение всей цепочки скриптов будет **немедленно остановлено**.

## 4. Пример использования

**Задача:** Запросить у пользователя уникальный идентификатор проекта, который должен состоять из 3 букв и 4 цифр (например, `PRJ-1234`).

1.  Добавляем в цепочку скрипт **"Пользовательский ввод"**.
2.  Настраиваем параметры:
    -   `Имя переменной результата`: `project_id`
    -   `Заголовок`: `Ввод ID проекта`
    -   `Тип валидации`: `custom`
    -   `Пользовательский шаблон`: `^[A-Z]{3}-\d{4}$`
    -   `Описание пользовательского шаблона`: `Введите ID проекта в формате XXX-0000 (например, PRJ-1234).`
3.  Следующий скрипт в цепочке может теперь безопасно использовать переменную `{project_id}`, будучи уверенным, что он будет запущен только в случае корректного ввода.