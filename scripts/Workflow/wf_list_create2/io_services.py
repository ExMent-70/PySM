"""
io_services.py
==============
Модуль отвечает за ввод-вывод данных (I/O).
"""

import csv
import json
import sys
import pathlib
import re  # Добавлен для поиска версии в файле
from typing import List, Dict, Any, Optional, Tuple

from domain import Student, ExtraService, CHILDREN_LIST_FILENAME

try:
    import jinja2
except ImportError:
    jinja2 = None

# -------------------------------------------------------------------------
# Константы шаблона
# -------------------------------------------------------------------------

# Версия шаблона. При изменении структуры шаблона нужно увеличивать это число.
TEMPLATE_VERSION = "2.1"

# Имя файла шаблона для AI
AI_PROMPT_TEMPLATE_FILENAME = "ai_prompt_template.txt"

# ================= ШАБЛОН ПРОМПТА =================
DEFAULT_AI_PROMPT = """
**Роль:** Ты — профессиональный редактор, корректор и Data Scientist.

**Задача:**
Тебе предоставлен JSON-список учеников и неструктурированный текст. 
Твоя цель — заполнить ВСЕ поля в объекте `info` для каждого ученика, используя данные из текста.

**Входные данные:**
1. JSON-структура (список учеников и поля info):
{{STUDENT_LIST_JSON}}

2. Неструктурированный текст (список имен и данных, например, цитат, хобби и т.д.):
{{RAW_TEXT}}

**Инструкции (Строго к исполнению):**

1. **Анализ структуры (Dynamic Field Mapping):**
   - Посмотри, какие ключи есть внутри объекта `info` в исходном JSON (например: "Цитата", "Мечта", "Хобби").
   - Найди соответствующую информацию в тексте для каждого поля.
   - Если в тексте есть данные только для одного поля (например, только цитаты), заполни только его, остальные оставь без изменений.

2. **Сопоставление (Smart Matching):**
   - Ищи людей нестрого: "Саша" = "Александр", "Лера" = "Валерия", "Вячеслав" = "Слава".
   - Если фамилия и имя в тексте перепутаны местами — это тот же человек.
   - Если человека из текста нет в JSON — игнорируй его.

3. **Правила Редактуры (Типографика и Грамматика):**
   - **Регистр:** Текст должен начинаться с Заглавной буквы.
   - **Тире:** Все дефисы (`-`), разделяющие части предложения, заменяй на длинное тире (`—`) с пробелами.
   - **Кавычки:** Убирай внешние кавычки, если текст просто вставлен. Если внутри есть прямая речь, используй «ёлочки».
   - **Авторство:** Если указан автор (например, Og Buda), выноси его в скобки в конце. Исправляй имена (OG Buda).
   - **Эмодзи:** Сохраняй только смысловые эмодзи. Мусор удаляй.
   - **Ошибки:** Исправляй орфографию и пунктуацию (гдз -> ГДЗ, ни что -> ничто).
   - **Форматирование:** Если текст — стих/диалог, используй `\\n` для переноса строк.

4. **Формат вывода:**
   - Верни ТОЛЬКО валидный JSON.
   - Не сокращай список, верни полную структуру со всеми студентами.
"""

def get_ai_prompt_template(directory: pathlib.Path) -> str:
    """
    Возвращает текст шаблона для AI.
    Если файла нет, создает его с дефолтным содержимым.
    """
    path = directory / AI_PROMPT_TEMPLATE_FILENAME
    if not path.exists():
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(DEFAULT_AI_PROMPT)
        except IOError as e:
            print(f"Warning: Could not create AI template: {e}", file=sys.stderr)
            return DEFAULT_AI_PROMPT
            
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except IOError:
        return DEFAULT_AI_PROMPT

# Шаблон HTML с версией в заголовке
DEFAULT_HTML_TEMPLATE = f"""<!-- VERSION: {TEMPLATE_VERSION} -->
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>Список класса {{{{ class_name }}}}</title>
    <style>
        body {{ font-family: 'Times New Roman', Times, serif; font-size: 14pt; width: 210mm; margin: auto; }}
        .header, .footer {{ margin-bottom: 5px; text-align: center; }}
        .class-name {{ font-size: 16pt; font-weight: bold; text-align: center; margin: 10px 0; }}
        table {{ width: 100%; border-collapse: collapse; border: 1px solid black; }}
        th, td {{ border: 1px solid black; padding: 5px; text-align: left; vertical-align: middle; }}
        th {{ background-color: #f2f2f2; font-weight: bold; text-align: center; }}
        td:nth-child(1), td:nth-child(2), td:nth-child(5) {{ text-align: center; }}
        .total-row td {{ font-weight: bold; }}
        .signatures {{ margin-top: 20px; display: flex; justify-content: space-between; }}
        .signer {{ display: inline-block; text-align: center; }}
        .signer-name {{ margin-right: 150px; }}
        .signature-line {{ border-bottom: 1px solid black; width: 200px; margin-top: 30px; }}
        .signature-caption {{ font-size: 10pt; }}
        
        .main-service-price {{ color: #444; font-size: 0.95em; }}
        
        /* Доп. услуги */
        .extras-container {{ margin-top: 4px; font-size: 0.85em; border-top: 1px dotted #999; padding-top: 2px; }}
        .extras-item {{ margin-left: 5px; }}
        .extras-calc {{ color: #444; }}
        .extras-comment {{ font-size: 0.9em; color: #666; font-style: italic; }}

        /* --- Блок доп. информации (Extended Mode) --- */
        .info-container {{ 
            margin-top: 8px; 
            padding: 5px; 
            background-color: #f9f9f9; 
            border: 1px solid #eee;
            font-size: 0.9em;
        }}
        .info-row {{ display: flex; margin-bottom: 2px; }}
        .info-key {{ font-weight: bold; margin-right: 5px; color: #555; min-width: 100px; }}
        .info-val {{ font-style: italic; }}

        @media print {{
            @page {{ size: A4; margin: 15mm; }}
            body {{ margin: 0; font-size: 12pt; -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
            tr {{ page-break-inside: avoid; }}
            .info-container {{ border: 1px solid #ccc; }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <p>Приложение №1 к договору №______-25/26 от «_____» __________ 202__ г.</p>
        <p>Количество и вид заказываемой фотопродукции</p>
    </div>
    <div class="class-name">{{{{ class_name }}}}</div>
    <table>
        <thead>
            <tr>
                <th style="width: 10%;">№ съемки</th>
                <th style="width: 5%;">№</th>
                <th style="width: 35%;">Фамилия, имя</th>
                <th style="width: 40%;">Вид фотопродукции</th>
                <th style="width: 10%;">Итого (руб)</th>
            </tr>
        </thead>
        <tbody>
            {{% for student in students %}}
            <tr>
                <td>{{{{ student.shoot_order if student.shoot_order is not none else '' }}}}</td>
                <td>{{{{ student.alpha_order }}}}</td>
                <td><b>{{{{ student.surname }}}}</b> {{{{ student.name }}}}</td>
                <td>
                    <!-- Услуги -->
                    <div>
                        {{{{ student.service_type }}}} 
                        <span class="main-service-price">({{{{ student.service_cost }}}} руб.)</span>
                    </div>
                    {{% if student.extra_services %}}
                    <div class="extras-container">
                        {{% for ex in student.extra_services %}}
                        <div class="extras-item">
                            + {{{{ ex.name }}}} 
                            <span class="extras-calc">
                                ({{{{ ex.qty }}}} шт. x {{{{ ex.cost }}}} = {{{{ ex.qty * ex.cost }}}} руб.)
                            </span>
                            {{% if ex.comment %}}
                            <span class="extras-comment">({{{{ ex.comment }}}})</span>
                            {{% endif %}}
                        </div>
                        {{% endfor %}}
                    </div>
                    {{% endif %}}

                    <!-- Доп. информация (Extended Mode) -->
                    {{% if extended_mode and student.info %}}
                    <div class="info-container">
                        {{% for key, val in student.info.items() %}}
                        <div class="info-row">
                            <span class="info-key">{{{{ key }}}}:</span>
                            <span class="info-val">{{{{ val }}}}</span>
                        </div>
                        {{% endfor %}}
                    </div>
                    {{% endif %}}
                </td>
                <td>{{{{ student.total_cost }}}}</td>
            </tr>
            {{% endfor %}}
            <tr class="total-row">
                <td colspan="4" style="text-align: right; border: none;">Итого к оплате:</td>
                <td style="text-align: center;">{{{{ total_cost }}}}</td>
            </tr>
        </tbody>
    </table>
    <div class="signatures" style="margin-top:50px;">
        <div class="signer">
            ЗАКАЗЧИК
            <div class="signature-line"></div>
            <div class="signature-caption">(подпись)</div>
        </div>
        <div class="signer signer-name">
             ИСПОЛНИТЕЛЬ
            <div class="signature-line"></div>
            <div class="signature-caption">(подпись)</div>
        </div>
    </div>
</body>
</html>"""


def ensure_template_exists(directory: pathlib.Path) -> pathlib.Path:
    """
    Проверяет версию шаблона HTML.
    Если версии не совпадают, старый файл бэкапится, а новый записывается.
    """
    template_path = directory / "_list_template.html"
    need_update = False

    if template_path.exists():
        try:
            # Считываем первые 100 байт, чтобы найти версию в комментарии
            with open(template_path, 'r', encoding='utf-8') as f:
                head = f.read(100)
            
            # Ищем <!-- VERSION: X.X -->
            match = re.search(r"<!-- VERSION: ([\d\.]+) -->", head)
            if match:
                file_version = match.group(1)
                if file_version != TEMPLATE_VERSION:
                    need_update = True
            else:
                # Если тега версии нет, значит файл старый
                need_update = True
                
        except Exception as e:
            print(f"WARNING: Ошибка проверки версии шаблона: {e}", file=sys.stderr)
            need_update = True
    else:
        need_update = True

    if need_update:
        # Если файл существует, делаем бэкап
        if template_path.exists():
            backup_path = directory / "_list_template.old.html"
            try:
                if backup_path.exists():
                    backup_path.unlink()
                template_path.rename(backup_path)
                print(f"INFO: Шаблон обновлен до версии {TEMPLATE_VERSION}. Старый сохранен в {backup_path.name}", file=sys.stderr)
            except OSError as e:
                print(f"WARNING: Не удалось создать бэкап шаблона: {e}", file=sys.stderr)

        # Записываем новый шаблон
        try:
            with open(template_path, 'w', encoding='utf-8') as f:
                f.write(DEFAULT_HTML_TEMPLATE)
        except IOError as e:
            print(f"ERROR: Не удалось создать файл шаблона: {e}", file=sys.stderr)
            
    return template_path


def load_session(path: pathlib.Path) -> Tuple[Dict[str, Any], List[Student]]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    metadata = {
        "class_name": data.get("class_name", path.stem),
        "service_type": data.get("service_type", ""),
        "info_columns": data.get("info_columns", [])
    }
    
    students = [Student.from_dict(s) for s in data.get("students", [])]
    return metadata, students


def save_session(path: pathlib.Path, class_name: str, service_type: str, 
                 students: List[Student], info_columns: List[str]) -> None:
    # Сериализация с гарантией наличия всех ключей info
    students_data_list = []
    for s in students:
        s_data = s.to_dict()
        info_dict = s_data.get('info', {})
        for col in info_columns:
            if col not in info_dict:
                info_dict[col] = ""
        s_data['info'] = info_dict
        students_data_list.append(s_data)

    data = {
        "class_name": class_name,
        "service_type": service_type,
        "info_columns": info_columns,
        "students": students_data_list
    }
    
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def export_to_csv(path: pathlib.Path, students: List[Student], info_columns: List[str] = None) -> None:
    info_columns = info_columns or []
    fieldnames = [
        "shoot_order", "alpha_order", "surname", "name", 
        "service_type", "service_cost", "extra_services", "total_cost"
    ] + info_columns
    
    rows_to_write = []
    for s in students:
        extras_items = []
        for ex in s.extra_services:
            comment = f" [{ex.comment}]" if ex.comment else ""
            extras_items.append(f"{ex.name} ({ex.qty}x{ex.cost}){comment}")
        extras_str = "; ".join(extras_items)

        row = {
            "shoot_order": s.shoot_order if s.shoot_order is not None else "",
            "alpha_order": s.alpha_order,
            "surname": s.surname,
            "name": s.name,
            "service_type": s.service_type,
            "service_cost": s.service_cost,
            "extra_services": extras_str,
            "total_cost": s.total_cost
        }
        for col in info_columns:
            row[col] = s.info.get(col, "")
        rows_to_write.append(row)

    with open(path, 'w', newline='', encoding='utf-16') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')
        writer.writeheader()
        writer.writerows(rows_to_write)


def export_to_txt(directory: pathlib.Path, students: List[Student], filename: str = None) -> None:
    output_path = directory / (filename or CHILDREN_LIST_FILENAME)
    sorted_students = sorted(
        [s for s in students if s.shoot_order is not None], 
        key=lambda s: s.shoot_order
    )
    lines = [f"{s.surname} {s.name}".strip() for s in sorted_students]
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))


def export_to_html(path: pathlib.Path, class_name: str, students: List[Student], 
                   template_dir: pathlib.Path, extended_mode: bool = False) -> bool:
    if not jinja2: raise ImportError("Jinja2 not installed")

    ensure_template_exists(template_dir)
    global_total = sum(s.total_cost for s in students)

    context = {
        "class_name": class_name,
        "students": students, 
        "total_cost": global_total,
        "extended_mode": extended_mode
    }

    try:
        env = jinja2.Environment(loader=jinja2.FileSystemLoader(template_dir))
        template = env.get_template("_list_template.html")
        with open(path, 'w', encoding='utf-8') as f:
            f.write(template.render(context))
        return True
    except Exception as e:
        raise IOError(f"HTML Generation failed: {e}")