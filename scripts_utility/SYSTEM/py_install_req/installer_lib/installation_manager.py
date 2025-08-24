# installer_lib/installation_manager.py

import logging
import re
import json
from pathlib import Path
from typing import List, Dict

from .models import InstallationPlan, SystemInfo, PackageInfo
from .utils import run_command
from .config import INSIGHTFACE_WINDOWS_WHEEL_URL

# 1. Добавляем packaging. Если не установлена, будет предупреждение.
try:
    from packaging.requirements import Requirement
    from packaging.version import Version
except ImportError:
    Requirement = Version = None

class InstallationManager:
    # ... __init__ и execute_plan теперь используют проверку ...
    def __init__(
        self,
        plan: InstallationPlan,
        system_info: SystemInfo,
        python_executable: Path,
        force_upgrade: bool = False
    ):
        self.plan = plan
        self.system_info = system_info
        self.python_executable = python_executable
        self.force_upgrade = force_upgrade
        self.use_uv = self._check_and_install_uv()
        # 2. НОВЫЙ БЛОК: Получаем список установленных пакетов при инициализации.
        self.installed_packages = self._get_installed_packages()

    def execute_plan(self):
        # ... вызовы _install_* теперь будут работать с отфильтрованными списками ...
        if self.plan.is_empty():
            logging.info("План установки пуст. Установка не требуется.")
            return

        logging.info("\nСравнение с установленными пакетами и выполнение плана")
        self._install_regular_packages()
        self._install_torch_packages()
        self._install_onnx_packages()
        self._install_triton_packages()
        self._install_insightface_packages()
        logging.info("\nУстановка пакетов успешно завершена\n")

    def _get_installed_packages(self) -> Dict[str, str]:
        """Получает словарь {имя_пакета: версия} из окружения Python."""
        # 3. НОВЫЙ МЕТОД: Логика из вашего простого скрипта.
        logging.info("\nПолучение списка установленных пакетов...")
        cmd = [str(self.python_executable), "-m", "pip", "list", "--format=json"]
        success, stdout, stderr = run_command(cmd)
        if not success:
            logging.warning(f"Не удалось получить список пакетов: {stderr}")
            return {}
        try:
            packages_list = json.loads(stdout)
            installed = {p["name"].lower().replace("_", "-"): p["version"] for p in packages_list}
            logging.info(f"- найдено <b>{len(installed)}</b> пакетов установленных в ENV")
            return installed
        except json.JSONDecodeError:
            logging.warning("Не удалось расшифровать JSON от 'pip list'.")
            return {}

    def _filter_packages_to_install(self, packages: List[PackageInfo]) -> List[PackageInfo]:
        """Фильтрует список пакетов, оставляя только те, что требуют установки/обновления."""
        # 4. НОВЫЙ МЕТОД: Логика сравнения из вашего простого скрипта.
        to_install = []
        if not packages:
            return []
        
        if not Requirement or not Version:
            logging.warning("Библиотека 'packaging' не найдена. Сравнение версий будет неточным.")
            # Возвращаем все пакеты, если не можем провести точное сравнение
            return packages

        for pkg in packages:
            normalized_name = pkg.name.lower().replace("_", "-")
            
            if normalized_name not in self.installed_packages:
                logging.info(f"<b>  [+] Будет установлен: {pkg.to_spec()}</b>")
                to_install.append(pkg)
                continue

            current_version_str = self.installed_packages[normalized_name]
            if self.force_upgrade:
                logging.info(f"<i>  [U] Будет обновлен (--upgrade): {pkg.to_spec()}</i>")
                to_install.append(pkg)
                continue

            try:
                # Пытаемся создать объект Requirement для анализа спецификатора
                req = Requirement(pkg.to_spec())
                if req.specifier and not req.specifier.contains(Version(current_version_str)):
                    logging.error(f"  [!] Конфликт версий для <i>{pkg.name}</i>: требуется <i>{req.specifier}</i>, установлен <i>{current_version_str}</i>. Будет обновлен.")
                    to_install.append(pkg)
                else:
                    logging.info(f"  [=] Установлен/соответствует: <i>{pkg.to_spec()}</i>")
            except Exception:
                # Если спецификатор сложный или отсутствует, считаем, что все в порядке
                logging.info(f"  [=] Установлен: <i>{pkg.to_spec()}</i>")
        
        return to_install

    # ... _check_and_install_uv и _build_base_command без изменений ...
    def _check_and_install_uv(self) -> bool:
        logging.info("\nПроверка наличия менеджера пакетов UV для ускорения установки...")
        version_cmd = [str(self.python_executable), "-m", "uv", "--version"]
        success, _, _ = run_command(version_cmd)
        if success:
            logging.info("<i>UV найден (будет использован для установки)</i>")
            return True
        logging.info("<i>UV не найден. Попытка установки через pip...</i>")
        install_cmd = [str(self.python_executable), "-m", "pip", "install", "uv"]
        success, _, _ = run_command(install_cmd)
        if not success:
            logging.warning("<i>Не удалось установить UV (будет использован pip)</i>")
            return False
        success, _, _ = run_command(version_cmd)
        if success:
            logging.info("<i>UV успешно установлен</i>")
            return True
        else:
            logging.warning("<i>UV установлен, но не запускается (будет использован pip)</i>")
            return False

    def _build_base_command(self) -> List[str]:
        if self.use_uv:
            return [str(self.python_executable), "-m", "uv", "pip", "install"]
        else:
            return [str(self.python_executable), "-m", "pip", "install"]

    def _run_install_command(self, cmd: List[str], category_name: str):
        logging.info(f"Команда:<i> {' '.join(cmd)}</i>")
        success, stdout, stderr = run_command(cmd)
        if not success:
            logging.error(f"ОШИБКА при установке категории <b><i>{category_name}</i></b>.\n")
            logging.error(f"Stderr: {stderr}")
        else:
            logging.info(f"Категория <b><i>{category_name}</i></b> успешно установлена.\n")
            
    # 5. Методы _install_* теперь используют фильтрацию.
    def _install_regular_packages(self):
        packages_to_install = self._filter_packages_to_install(self.plan.regular_packages)
        if not packages_to_install: return
        logging.info("\n<b><i>Установка обычных пакетов...</i></b>")
        cmd = self._build_base_command()
        if self.force_upgrade: cmd.append("--upgrade")
        cmd.extend([pkg.to_spec() for pkg in packages_to_install])
        self._run_install_command(cmd, "Обычные пакеты")
        
    def _install_torch_packages(self):
        packages_to_install = self._filter_packages_to_install(self.plan.torch_packages)
        if not packages_to_install: return
        logging.info("\n<b><i>Установка пакетов Torch...</i></b>")
        cmd = self._build_base_command()
        cmd.append("--upgrade") # Torch лучше всегда обновлять до нужной версии
        if self.plan.torch_index_url:
            cmd.extend(["--index-url", self.plan.torch_index_url])
        cmd.extend([pkg.to_spec() for pkg in packages_to_install])
        self._run_install_command(cmd, "Torch")

    # ... и так далее для всех категорий ...
    def _install_onnx_packages(self):
        packages_to_install = self._filter_packages_to_install(self.plan.onnx_packages)
        if not packages_to_install: return
        logging.info("\n<b><i>Установка пакетов ONNX...</i></b>")
        cmd = self._build_base_command()
        cmd.append("--upgrade")
        if self.system_info.gpu and self.system_info.gpu.generation == "blackwell":
            cmd.append("--pre")
        cmd.append(self.plan.onnx_package_name or "onnxruntime")
        self._run_install_command(cmd, "ONNX")

    def _install_triton_packages(self):
        packages_to_install = self._filter_packages_to_install(self.plan.triton_packages)
        if not packages_to_install: return
        logging.info("\n<b><i>Установка пакетов Triton...</i></b>")
        cmd = self._build_base_command()
        cmd.append("--upgrade")
        cmd.append("triton-windows")
        self._run_install_command(cmd, "Triton")

    def _install_insightface_packages(self):
        packages_to_install = self._filter_packages_to_install(self.plan.insightface_packages)
        if not packages_to_install: return
        logging.info("\n<b><i>Установка пакетов Insightface...</i></b>")
        cmd = self._build_base_command()
        cmd.append("--upgrade")
        cmd.extend([INSIGHTFACE_WINDOWS_WHEEL_URL, "numpy==1.26.4"])
        self._run_install_command(cmd, "Insightface")