# 1. Блок: Импорт
from pysm_lib import pysm_context
import os

# 2. Блок: Получаем список всех экземпляров в наборе
all_instances = pysm_context.list_instances()

# 3. Блок: Находим ID нужного нам экземпляра по его имени
target_instance_id = None
for instance in all_instances:
    script_name = "hello_pysm"
    if instance.get("name") == script_name:
        target_instance_id = instance.get("id")
        break

# 4. Блок: Делаем условный переход
file_to_check = "data.csv"

if os.path.exists(file_to_check):
    if target_instance_id:
        print(f"Файл найден. Следующим будет выполнен скрипт '{script_name} ({target_instance_id})'.")
        pysm_context.set_next_script(target_instance_id)
else:
    print("Файл не найден. Выполнение продолжится по стандартной схеме.")