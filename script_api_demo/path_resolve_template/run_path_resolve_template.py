# 1. Блок: Импорт
from pysm_lib import pysm_context

# 2. Блок: Предположим, в контексте есть переменные:
# "output_dir": "D:/results"
# "image_name": "IMG_1234"
# "timestamp": "2024-05-20"

# 3. Блок: Создаем шаблон пути
path_template = "{test_output_dir}/{test_timestamp}_{test_image_name}.jpg"

# 4. Блок: Превращаем шаблон в реальный путь
resolved_path = pysm_context.resolve_template(path_template)
# resolved_path будет "D:/results/2024-05-20_IMG_1234.jpg"

print(f"Используется шаблон: {path_template}")
print(f"Итоговый путь к файлу: {resolved_path}")