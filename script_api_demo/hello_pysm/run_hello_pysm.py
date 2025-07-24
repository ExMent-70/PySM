# run_hello_pysm.py

# 1. Блок: Импортируем API
from pysm_lib import pysm_context

# 2. Блок: Основная логика
def main():
   
    # Получим информацию о запущенном экземпляре скрипта из контекста
    pysm_info = pysm_context.get("var_Hello")
    
    print(f"{pysm_info}")
    print(f"")
    print(f"Нажмите на кпоку с буквой <b>I</b> на панели инструментов.")
    print(f"Убедидесь, что поле Контекст коллекции содержит всего одну переменную <b>var_Hello</b>.")
    

if __name__ == "__main__":
    main()