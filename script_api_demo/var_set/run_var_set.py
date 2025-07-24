# 1. Блок: Импорт
from pysm_lib import pysm_context

# Сохраним к контекст несколько переменных, которые будут использоваться в следующих примерах
# 2. Блок: Простые переменный различных типов данных
pysm_context.set("test_user_name", "Андрей")
pysm_context.set("test_processed_files_count", 150)
pysm_context.set("test_is_successful", True)
pysm_context.set("test_source_folder", "source_images")
pysm_context.set("test_output_dir", "D:/results")
pysm_context.set("test_image_name", "IMG_1234")
pysm_context.set("test_timestamp", "2024-05-20")

# 3. Блок: Более сложные переменные
user_data = {
    "name": "Андрей",
    "login": "andrey_p",
    "address": {
        "city": "Москва",
        "zip": "123456"
    },
    "roles": ["admin", "photographer"]
}
pysm_context.set("test_user_profile", user_data)
pysm_context.set("test_file_list", ["file1.jpg", "file2.jpg"])


print("9 переменных было добавлено в контекст коллекции.")
print(f"")
print(f"Еще раз нажмите на кпоку с буквой <b>I</b> на панели инструментов.")
print(f"Убедидесь, что новые переменные добавлены в Контекст коллекции.")