"""
parser.py
=========
Модуль для интеллектуального и простого разбора списков имен.
Использует Natasha для NLP-анализа и извлечения сущностей,
а также ipymarkup/BeautifulSoup для генерации визуального представления
и извлечения цветовых кодов.
"""

import json
import pathlib
import sys
# Используем regex из стандартной библиотеки или сторонней, если нужно (в оригинале было regex as re)
import re 
from typing import List, Dict, Any, Tuple, Optional

# Импорт моделей данных
from .domain import Student


RESOURCE_DIR = pathlib.Path(__file__).parent / "resources"




def smart_capitalize(text: str) -> str:
    """
    Умно капитализирует строку, обрабатывая слова с дефисами или пробелами.
    Пример: "анна-мария" -> "Анна-Мария"
    """
    def capitalize_word(word: str) -> str:
        if not word: return ""
        return word[0].upper() + word[1:]
    
    if not text:
        return ""
    
    # Разбиваем по пробелам, затем каждую часть по дефисам
    parts = text.split()
    capitalized_parts = []
    for part in parts:
        by_hyphen = '-'.join(capitalize_word(subpart) for subpart in part.split('-'))
        capitalized_parts.append(by_hyphen)
        
    return ' '.join(capitalized_parts)


def simple_parse_text(text: str) -> List[Student]:
    """
    Извлекает имена и фамилии с помощью простого regex.
    Возвращает список объектов Student.
    """
    students: List[Student] = []
    lines = text.splitlines()
    # ИЗМЕНЕНО: Паттерн теперь использует стандартные классы символов для букв
    pattern = re.compile(r"^[ \d.\-)]*([A-Za-zА-Яа-яЁё\-]+)[ ,]+([A-Za-zА-Яа-яЁё\-]+)")
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        match = pattern.search(line)
        if match:
            surname, name = match.groups()
            students.append(Student(
                surname=smart_capitalize(surname),
                name=smart_capitalize(name)
            ))
    return students


class SmartParser:
    """
    Парсер, использующий Natasha для извлечения имен и ipymarkup для визуализации.
    """
# --- ИЗМЕНЕННЫЙ БЛОК 2: parser.py (Метод __init__ в SmartParser) ---
class SmartParser:
    """
    Парсер, использующий Natasha для извлечения имен и ipymarkup для визуализации.
    (Использует отложенную загрузку для ускорения запуска приложения).
    """
    def __init__(
        self,
        surname_style: Dict[str, str],
        name_style: Dict[str, str],
        fio_style: Dict[str, str],
    ):
        self.morph = None
        self.extractor = None
        self.palette = None
        self.normalization_dict = {}
        
        # Сохраняем стили в экземпляре класса, чтобы инициализировать палитру позже
        self._surname_style = surname_style
        self._name_style = name_style
        self._fio_style = fio_style
        
        self.SURNAME_COLOR_HEX = surname_style.get("background-color", "#e3f2fd")
        self.NAME_COLOR_HEX = name_style.get("background-color", "#e8f5e9")

        self._load_normalization_dict()
        
        # ВАЖНО: Инициализация Natasha и ipymarkup убрана отсюда в метод _lazy_init

    def _load_normalization_dict(self):
        """Загружает словарь нормализации имен (Саша -> Александр)."""
        dict_path = RESOURCE_DIR / "_names_normalization.json"
        default_dict = {"Саша": "Александр", "Аня": "Анна", "Настя": "Анастасия"}
        
        if dict_path.exists():
            try:
                with open(dict_path, 'r', encoding='utf-8') as f:
                    self.normalization_dict = json.load(f)
            except Exception:
                self.normalization_dict = default_dict
        else:
            try:
                dict_path.parent.mkdir(parents=True, exist_ok=True)
                with open(dict_path, 'w', encoding='utf-8') as f:
                    json.dump(default_dict, f, ensure_ascii=False, indent=4)
                self.normalization_dict = default_dict
            except Exception:
                self.normalization_dict = default_dict

    def _normalize_name(self, name: str) -> str:
        """Нормализует имя по словарю."""
        return self.normalization_dict.get(name.capitalize(), name.capitalize())


# --- ИЗМЕНЕННЫЙ БЛОК 3: parser.py (Новый _lazy_init и начало parse_text) ---
    def _lazy_init(self) -> bool:
        """Отложенная загрузка библиотек. Вызывается только при первом парсинге."""
        if self.extractor is not None:
            return True # Уже инициализировано
            
        try:
            import pymorphy3
            from natasha import NamesExtractor
            from ipymarkup.palette import Palette, Color, Rgb
            
            # Загрузка ML-моделей
            self.morph = pymorphy3.MorphAnalyzer()
            self.extractor = NamesExtractor(self.morph)
            
            # Инициализация палитры ipymarkup
            final_surname_color = Color(
                "Фамилия", background=Rgb(self.SURNAME_COLOR_HEX),
                border=Rgb(self._surname_style.get("border-color", "#bbdefb")),
                text=Rgb(self._surname_style.get("color", "#0d47a1"))
            )
            final_name_color = Color(
                "Имя", background=Rgb(self.NAME_COLOR_HEX),
                border=Rgb(self._name_style.get("border-color", "#c8e6c9")),
                text=Rgb(self._name_style.get("color", "#1b5e20"))
            )
            final_fio_color = Color(
                "ФИО", background=Rgb(self._fio_style.get("background-color", "#fff3e0")),
                border=Rgb(self._fio_style.get("border-color", "#ffe0b2")),
                text=Rgb(self._fio_style.get("color", "#e65100"))
            )
            self.palette = Palette([final_surname_color, final_name_color, final_fio_color])
            return True
            
        except ImportError as e:
            print(f"Lazy Load Error: {e}", file=sys.stderr)
            return False

    def parse_text(self, text: str) -> Tuple[List[Student], str]:
        """
        Извлекает имена и фамилии с помощью Natasha, генерирует HTML-разметку.
        """
        # 1. Проверяем и подгружаем тяжелые библиотеки
        if not self._lazy_init():
            return[], "<p style='color: red;'>Парсер не инициализирован (отсутствуют библиотеки natasha, pymorphy3, ipymarkup).</p>"

        # 2. Локальные импорты вспомогательных библиотек (быстро, кэшируется питоном)
        try:
            from bs4 import BeautifulSoup
            from pysm_lib.pysm_theme_api import format_ipymarkup_box
        except ImportError:
            BeautifulSoup = None
            format_ipymarkup_box = None

        # Временный список словарей для сбора данных перед конвертацией в Student
        parsed_dicts: List[Dict[str, Any]] =[]
        spans = []
        matches = self.extractor(text)
        
        # ШАГ 1: Основной парсинг для извлечения имен и фамилий
        for match in matches:
            fact = match.fact
            if not (fact.first and fact.last): 
                continue
            
            # Фильтрация мусорных слов
            if "доп" in fact.first.lower() or "фото" in fact.first.lower(): 
                continue

            p_first = self.morph.parse(fact.first)[0]
            p_last = self.morph.parse(fact.last)[0]
            
            # Эвристика определения где имя, а где фамилия на основе тегов pymorphy
            if ('Name' in p_last.tag and 'Surn' in p_first.tag) or \
               ('Surn' in p_first.tag and 'Surn' not in p_last.tag):
                surname, name = fact.first, fact.last
                span1_label, span2_label = " (Фамилия)", " (Имя)"
            else:
                surname, name = fact.last, fact.first
                span1_label, span2_label = " (Имя)", " (Фамилия)"

            # Нормализация
            normalized_surname = self._normalize_name(surname)
            normalized_name = self._normalize_name(name)

            parsed_dicts.append({
                "surname": smart_capitalize(normalized_surname), 
                "name": smart_capitalize(normalized_name)
            })
            
            # Подготовка spans для ipymarkup
            try:
                match_text = text[match.start:match.stop]
                s_match = re.search(re.escape(surname), match_text, re.IGNORECASE)
                n_match = re.search(re.escape(name), match_text, re.IGNORECASE)
                
                if s_match and n_match:
                    # Определяем порядок следования в исходном тексте
                    if span1_label == " (Фамилия)":
                        spans_to_add = ((s_match, span1_label), (n_match, span2_label))
                    else:
                        spans_to_add = ((n_match, span1_label), (s_match, span2_label))
                        
                    spans.append((match.start + spans_to_add[0][0].start(), match.start + spans_to_add[0][0].end(), spans_to_add[0][1]))
                    spans.append((match.start + spans_to_add[1][0].start(), match.start + spans_to_add[1][0].end(), spans_to_add[1][1]))
                else:
                    spans.append((match.start, match.stop, "ФИО"))
            except Exception:
                spans.append((match.start, match.stop, "ФИО"))


        # ШАГ 2: Генерация HTML-разметки через ipymarkup
        markup_html = ""
        if format_ipymarkup_box and self.palette:
            try:
                markup_html = format_ipymarkup_box(text, spans, palette=self.palette)
            except Exception as e:
                print(f"IPYMARKUP ERROR: {e}", file=sys.stderr)

        
        # ШАГ 3: Парсинг HTML для извлечения цветов (Backward Compatibility Logic)
        if BeautifulSoup and markup_html:
            background_regex = re.compile(r"background:\s*([^;]+)")
            color_regex = re.compile(r"color:\s*([^;]+)")
            
            try:
                soup = BeautifulSoup(markup_html, 'html.parser')
                # Ищем все span, у которых есть стиль background
                all_color_spans = soup.find_all('span', style=background_regex)
                
                # Ожидаем, что на каждого человека будет ровно 2 span-а (Фамилия, Имя)
                if len(all_color_spans) == len(parsed_dicts) * 2:
                    for i, person_data in enumerate(parsed_dicts):
                        span1 = all_color_spans[i * 2]
                        span2 = all_color_spans[i * 2 + 1]

                        style1 = span1.get('style', '')
                        style2 = span2.get('style', '')

                        # Извлекаем цвет фона
                        match1_bg = background_regex.search(style1)
                        color1_bg = match1_bg.group(1).strip() if match1_bg else None
                        match2_bg = background_regex.search(style2)
                        color2_bg = match2_bg.group(1).strip() if match2_bg else None
                        
                        # Извлекаем цвет текста
                        match1_fg = color_regex.search(style1)
                        color1_fg = match1_fg.group(1).strip() if match1_fg else None
                        match2_fg = color_regex.search(style2)
                        color2_fg = match2_fg.group(1).strip() if match2_fg else None

                        # Присваиваем цвета в словарь
                        person_data['color1'] = color1_bg
                        person_data['color1_fg'] = color1_fg
                        person_data['color2'] = color2_bg
                        person_data['color2_fg'] = color2_fg
                else:
                    print(f"ПРЕДУПРЕЖДЕНИЕ: Рассинхронизация парсера (Spans: {len(all_color_spans)}, People: {len(parsed_dicts)}). Раскраска отменена.")
            except Exception as e:
                 print(f"BS4 ERROR: {e}", file=sys.stderr)

        # ШАГ 4: Конвертация словарей в объекты Student
        result_students = []
        for d in parsed_dicts:
            student = Student(
                surname=d.get("surname", ""),
                name=d.get("name", ""),
                color1=d.get("color1"),
                color1_fg=d.get("color1_fg"),
                color2=d.get("color2"),
                color2_fg=d.get("color2_fg")
            )
            result_students.append(student)

        return result_students, markup_html
