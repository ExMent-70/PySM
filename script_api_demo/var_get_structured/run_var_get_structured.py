# reader_script.py
from pysm_lib import pysm_context

print(f"Попробуем получить из Контекста более сложные данные:")
print(f"user_data = {{")
print(f"    \"name\": \"Андрей\",")
print(f"    \"login\": \"andrey_p\",")
print(f"    \"address\": {{")
print(f"        \"city\": ""Москва"",")
print(f"        \"zip\": \"123456\"")
print(f"    }},")
print(f"    \"roles\": [\"admin\", \"photographer\"]")
print(f"}},")
print(f"")

print(f"Получаем название города:")
# Читаем вложенные данные, используя точечную нотацию
city = pysm_context.get_structured("test_user_profile.address.city", default="Неизвестно")
# Этот вызов равносилен user_profile['address']['city']
print(f"Город пользователя: {city}")
print(f"")

print(f"Получаем название улицы:")
# Если ключ не найден, вернется значение по умолчанию
street = pysm_context.get_structured("test_user_profile.address.street", default="Улица не указана")
print(f"Улица: {street}")