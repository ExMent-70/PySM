# 1. Блок: Импорт
from pysm_lib import pysm_context

# 2. Блок: Чтение данных с указанием значения по умолчанию
print(f"Попробуем получить значения нескольких переменных, сохраненных в Контексте")
print(f"Например, переменных <b>user_name</b> и <b>processed_files</b>")
print(f"")

user_name = pysm_context.get("test_user_name", default="Незнакомец")
processed_files = pysm_context.get("test_processed_files_count", default=0)

print(f"Привет, {user_name}!")
print(f"Обработано файлов: {processed_files}")
print(f"")
print(f"Отлично! Все получилось!")