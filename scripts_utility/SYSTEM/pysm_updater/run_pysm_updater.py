import sys
import os
import shutil
import argparse
import zipfile
import urllib.request
import ssl
import time
import json
from pathlib import Path
from io import BytesIO
from datetime import datetime

# 1. БЛОК: Интеграция с PySM и константами
# ==============================================================================
try:
    from pysm_lib.pysm_context import pysm_context, ConfigResolver
    # Импортируем константу корня приложения
    from pysm_lib.app_constants import APPLICATION_ROOT_DIR
    IS_MANAGED_RUN = True
except ImportError:
    # Fallback для автономного запуска (если pysm_lib не в путях)
    IS_MANAGED_RUN = False
    ConfigResolver = None
    pysm_context = None
    # Если запущен не из PySM, считаем корнем текущую директорию
    APPLICATION_ROOT_DIR = Path(os.getcwd()).resolve()



# ==============================================================================
# КОНФИГУРАЦИЯ ПО УМОЛЧАНИЮ
# ==============================================================================
DEFAULT_REPO_URL = "https://github.com/ExMent-70/PySM/archive/refs/heads/main.zip"


def create_backup(target_dir: Path) -> Path:
    """Создает ZIP-архив текущего состояния системы перед обновлением."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = target_dir / "_backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    backup_file = backup_dir / f"pysm_backup_{timestamp}.zip"
    
    print(f"⏳ Создание резервной копии: {backup_file.name}...", flush=True)
    
    with zipfile.ZipFile(backup_file, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(target_dir):
            # Исключаем:
            # 1. _backups (чтобы не бэкапить бэкапы)
            # 2. .git (служебная папка репозитория)
            # 3. __pycache__ (кэш байткода)
            # 4. _BIN (тяжелые бинарники/окружение, указано в предыдущем запросе)
            if "_backups" in root or ".git" in root or "__pycache__" in root or "_BIN" in root:
                continue
                
            for file in files:
                abs_path = Path(root) / file
                try:
                    rel_path = abs_path.relative_to(target_dir)
                    zf.write(abs_path, rel_path)
                except ValueError:
                    continue # Файл вне дерева (редко, но бывает)
                
    print("✅ Резервная копия создана.", flush=True)
    return backup_file

def download_repo(url: str, token: str = None) -> bytes:
    """Скачивает репозиторий в память."""
    print(f"⏳ Скачивание обновлений с GitHub...", flush=True)
    print(f"   URL: {url}", flush=True)
    
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"token {token}")
    
    # Игнорируем SSL ошибки
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=60) as response:
            return response.read()
    except Exception as e:
        print(f"❌ Ошибка скачивания: {e}", flush=True)
        sys.exit(1)

def install_update(zip_bytes: bytes, target_dir: Path):
    """Распаковывает архив и обновляет файлы."""
    print("⏳ Установка обновлений...", flush=True)
    
    try:
        with zipfile.ZipFile(BytesIO(zip_bytes)) as zf:
            # Определение корневой папки в архиве (обычно RepoName-main)
            root_folder = None
            for name in zf.namelist():
                if '/' in name:
                    possible_root = name.split('/')[0]
                    if root_folder is None:
                        root_folder = possible_root
                    elif root_folder != possible_root:
                        root_folder = ""
                        break
            
            updated_count = 0
            
            for member in zf.infolist():
                if member.is_dir():
                    continue
                
                archive_path = member.filename
                
                # Обрезаем корневую папку архива, чтобы файлы легли правильно
                if root_folder and archive_path.startswith(root_folder + "/"):
                    rel_path = archive_path[len(root_folder) + 1:]
                else:
                    rel_path = archive_path
                
                if not rel_path: continue
                
                dest_path = target_dir / rel_path
                
                # Защита пользовательских конфигов от перезаписи
                if dest_path.name in ["local_settings.json", "user_config.py"]:
                    print(f"   Skip config: {rel_path}", flush=True)
                    continue

                dest_path.parent.mkdir(parents=True, exist_ok=True)
                
                with zf.open(member) as source, open(dest_path, "wb") as target:
                    shutil.copyfileobj(source, target)
                
                updated_count += 1
                
            print(f"✅ Обновлено файлов: {updated_count}", flush=True)
            
    except Exception as e:
        print(f"❌ Ошибка при распаковке: {e}", flush=True)
        sys.exit(1)

def check_requirements(target_dir: Path):
    """Проверяет наличие requirements.txt"""
    req_file = target_dir / "requirements.txt"
    if req_file.exists():
        print("ℹ️ Найден файл requirements.txt. Если в обновлении были новые библиотеки, выполните:")
        print(f"   pip install -r {req_file}")

# 2. БЛОК: Конфигурация
def get_config():
    parser = argparse.ArgumentParser(description="Автоматическое обновление PySM с GitHub.")
    
    parser.add_argument("--repo_url", type=str, default=DEFAULT_REPO_URL, help="Ссылка на zip-архив")
    parser.add_argument("--target_dir", type=str, help="Папка для обновления (по умолчанию - корень PySM)")
    parser.add_argument("--token", type=str, help="GitHub Token")
    parser.add_argument("--no_backup", action="store_true", help="Не создавать бэкап")

    if IS_MANAGED_RUN and ConfigResolver:
        return ConfigResolver(parser).resolve_all()
    return parser.parse_args()

# 3. БЛОК: Main
def main():
    print("=== ЗАПУСК ОБНОВЛЕНИЯ СИСТЕМЫ ===", flush=True)
    config = get_config()
    
    # ОПРЕДЕЛЕНИЕ ЦЕЛЕВОЙ ПАПКИ
    if config.target_dir:
        # Если пользователь явно указал путь через аргументы
        target_path = Path(config.target_dir).resolve()
    else:
        # Иначе берем из константы приложения (Single Source of Truth)
        target_path = APPLICATION_ROOT_DIR
    
    print(f"Целевая папка: {target_path}", flush=True)
    
    # Проверка на адекватность пути
    if not target_path.exists():
        print(f"❌ Ошибка: Целевая папка не найдена: {target_path}", flush=True)
        sys.exit(1)

    # 1. Бэкап
    if not config.no_backup:
        create_backup(target_path)
    
    # 2. Скачивание
    zip_data = download_repo(config.repo_url, config.token)
    
    # 3. Установка
    install_update(zip_data, target_path)
    
    # 4. Проверка зависимостей
    check_requirements(target_path)
    
    print("\n=== ОБНОВЛЕНИЕ ЗАВЕРШЕНО УСПЕШНО ===", flush=True)
    print("Пожалуйста, перезапустите PyScriptManager для применения изменений.", flush=True)

if __name__ == "__main__":
    main()