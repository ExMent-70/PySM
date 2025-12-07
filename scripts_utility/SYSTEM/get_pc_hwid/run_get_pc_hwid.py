import subprocess
import hashlib
import sys

def main():
    print("--- Hardware ID Generator ---")
    try:
        raw_uuid = ""
        if sys.platform == 'win32':
            # Получаем UUID материнской платы
            cmd = 'wmic csproduct get uuid'
            # startupinfo скрывает окно консоли
            si = subprocess.STARTUPINFO()
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            raw_uuid = subprocess.check_output(cmd, startupinfo=si).decode().split('\n')[1].strip()
        else:
            # Linux/Mac fallback
            import uuid
            raw_uuid = str(uuid.getnode())

        if not raw_uuid:
            raise ValueError("Empty UUID")

        # Первый проход хеширования (Базовый HWID)
        hwid_hash = hashlib.sha256(raw_uuid.encode()).hexdigest()
        
        print(f"\nВаш базовый HWID:\n{hwid_hash}\n")
        print("Отправьте этот код разработчику.")
        
    except Exception as e:
        print(f"Ошибка получения HWID: {e}")
    
if __name__ == "__main__":
    main()