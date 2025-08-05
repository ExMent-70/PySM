from pysm_lib import pysm_context
import os

# 2. Блок: Получаем относительный путь из контекста
# Допустим, в контексте есть переменная `source_folder` со значением "source_images"
relative_path = pysm_context.get("test_source_folder")

# 3. Блок: Преобразуем его в полный, безопасный путь
if relative_path:
    absolute_path = pysm_context.resolve_path(relative_path)
    # absolute_path будет равен 'D:\\MyProjects\\PhotoProcessing\\source_images'
    
    print(f"Относительный путь к папке: {relative_path}")
    print(f"Абсолютный путь к папке: {absolute_path}")
    print(f"Существует ли папка? {os.path.exists(absolute_path)}")