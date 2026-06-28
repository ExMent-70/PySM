# installer_lib/installation_manager.py

import logging
import json
import re
from pathlib import Path
from typing import List, Dict

from .models import InstallationPlan, SystemInfo, PackageInfo
from .utils import run_command, run_command_streaming
from .config import INSIGHTFACE_WINDOWS_WHEEL_URLS

try:
    from pysm_lib.pysm_icons import icons as pysm_icons
except Exception:
    pysm_icons = None

try:
    from packaging.requirements import Requirement
    from packaging.version import Version
except ImportError:
    Requirement = Version = None

class InstallationManager:
    """Выполняет план установки по категориям с проверками GPU-зависимостей.

    Менеджер не ставит весь requirements одним вызовом pip: специальные
    семейства (`torch`, `onnxruntime`, `insightface`, `triton`) требуют разных
    индексов, wheel-решений и пост-проверок. Поэтому план разбивается на
    категории, каждая категория получает отдельную команду и отдельный итог.
    """

    CATEGORY_ORDER = ["Обычные пакеты", "Torch", "ONNX", "Triton", "Insightface"]

    def __init__(
        self,
        plan: InstallationPlan,
        system_info: SystemInfo,
        python_executable: Path,
        force_upgrade: bool = False,
        plan_only: bool = False,
    ):
        self.plan = plan
        self.system_info = system_info
        self.python_executable = python_executable
        self.force_upgrade = force_upgrade
        self.plan_only = plan_only
        self.failures: List[str] = []
        self.category_results: Dict[str, Dict[str, str]] = {}
        self.use_log_icons = True
        self.use_uv = self._check_uv_available() if plan_only else self._check_and_install_uv()
        self.installed_packages = self._get_installed_packages()
        self.installed_torch_info = self._get_installed_torch_info()

    def execute_plan(self):
        """Сравнивает план с окружением, выполняет категории и подводит итог."""
        if self.plan.is_empty():
            logging.info("План установки пуст. Установка не требуется.")
            return

        logging.info("\nСравнение с установленными пакетами и выполнение плана")
        self._install_regular_packages()
        self._install_torch_packages()
        self._install_onnx_packages()
        self._install_triton_packages()
        self._install_insightface_packages()
        self._log_plan_notes()
        self._log_category_summary()

        if self.failures:
            if self.plan_only:
                details = "\n".join(f"  - {item}" for item in self.failures)
                logging.error(f"\nРежим плана: обнаружены блокирующие проблемы:\n{details}\n")
                return
            details = "\n".join(f"  - {item}" for item in self.failures)
            raise RuntimeError(f"Не удалось установить все категории пакетов:\n{details}")

        if self.plan_only:
            logging.info("\nРежим плана: установка пакетов не выполнялась.\n")
            return

        logging.info("\nУстановка пакетов успешно завершена\n")

    def _set_category_result(self, category_name: str, status: str, details: str = "") -> None:
        self.category_results[category_name] = {"status": status, "details": details}

    def _log_category_summary(self) -> None:
        title = "ИТОГ ПЛАНА УСТАНОВКИ" if self.plan_only else "ИТОГ УСТАНОВКИ"
        logging.info(f"\n<b>{title}</b>")

        for category_name in self.CATEGORY_ORDER:
            result = self.category_results.get(category_name)
            if not result:
                result = {"status": "SKIP", "details": "нет действий"}
            status = result["status"]
            details = result.get("details", "")
            line = f"{self._status_prefix(status, include_label=True)} {category_name}"
            if details:
                line += f": {details}"

            if status == "FAIL":
                logging.error(line)
            elif status == "PLAN":
                logging.warning(line)
            else:
                logging.info(line)

    def _log_plan_notes(self) -> None:
        if self.plan.requirement_rewrites:
            logging.warning("\n<b>Явные замены requirements для выбранного бэкенда:</b>")
            for rewrite in self.plan.requirement_rewrites:
                logging.warning(f"  - {rewrite}")

        if self.plan.diagnostics:
            logging.warning("\n<b>Диагностика плана установки:</b>")
            for diagnostic in self.plan.diagnostics:
                logging.warning(f"  - {diagnostic}")

    def _get_installed_packages(self) -> Dict[str, str]:
        """Получает словарь {имя_пакета: версия} из окружения Python."""
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
        """Возвращает только пакеты, которым действительно нужна установка.

        Решение строится из трех источников: `pip list`, спецификаторов
        requirements/constraints и специальных CUDA-проверок Torch-family.
        Прямые ссылки всегда пропускаются в установку, потому что для них
        нельзя надежно сравнить локальную версию с удаленным wheel/URL.
        """
        to_install = []
        if not packages:
            return []
        
        if not Requirement or not Version:
            logging.warning("Библиотека 'packaging' не найдена. Сравнение версий будет неточным.")
            # Без packaging безопаснее переустановить, чем ошибочно считать требование выполненным.
            return packages

        for pkg in packages:
            if pkg.direct_reference:
                logging.info(self._format_package_plan_action("DIRECT", "Прямая ссылка/локальный путь", pkg))
                to_install.append(pkg)
                continue

            if self._needs_torch_cuda_reinstall(pkg):
                logging.warning(
                    self._format_package_plan_action(
                        "WARN",
                        f"Переустановить Torch: установлена CPU/несовместимая сборка, выбрана CUDA {self.system_info.cuda.selected_version}",
                        pkg,
                    )
                )
                to_install.append(pkg)
                continue

            normalized_name = pkg.name.lower().replace("_", "-")
            
            if normalized_name not in self.installed_packages:
                logging.info(self._format_package_plan_action("INSTALL", "Установить", pkg))
                to_install.append(pkg)
                continue

            current_version_str = self.installed_packages[normalized_name]
            if self.force_upgrade:
                logging.info(self._format_package_plan_action("UPDATE", "Обновить", pkg))
                to_install.append(pkg)
                continue

            try:
                req = Requirement(pkg.to_spec())
                if req.specifier and not req.specifier.contains(Version(current_version_str)):
                    logging.error(
                        self._format_package_plan_action(
                            "WARN",
                            f"Конфликт версий: требуется {req.specifier}; обновить",
                            pkg,
                        )
                    )
                    to_install.append(pkg)
                elif not self._is_constraint_satisfied(pkg, current_version_str):
                    constraint = self._get_constraint_for_package(pkg)
                    logging.error(
                        self._format_package_plan_action(
                            "WARN",
                            f"Конфликт constraints: требуется {constraint.version or constraint.to_spec()}; обновить",
                            pkg,
                        )
                    )
                    to_install.append(pkg)
                else:
                    logging.info(self._format_package_plan_action("OK", "Соответствует", pkg))
            except Exception:
                # Нераспознанный, но уже установленный spec не должен ломать весь план.
                logging.info(self._format_package_plan_action("OK", "Установлен", pkg))
        
        return to_install

    def _format_package_plan_action(self, status: str, action: str, pkg: PackageInfo) -> str:
        details = self._format_package_version_details(status, pkg)
        return f"  {self._status_prefix(status)} {action}: <i>{pkg.to_spec()}</i>{details}"

    def _format_package_version_details(self, status: str, pkg: PackageInfo) -> str:
        current = self._get_current_version_label(pkg)
        target = self._get_target_version_label(pkg)

        if status == "OK":
            return f" (<b>{current}</b>)" if current != "не установлен" else ""

        if target:
            if current != "не установлен":
                if current == target:
                    return f" (<b>{current}</b>)"
                return f" (<b>{current}</b> -> <b>{target}</b>)"
            return f" (<b>{target}</b>)"

        if current != "не установлен":
            return f" (<b>{current}</b>)"
        return ""

    def _status_prefix(self, status: str, include_label: bool = False) -> str:
        status_icons = {
            "OK": ("OK", "[OK]"),
            "INSTALL": ("ADD", "[+]"),
            "UPDATE": ("REFRESH", "[U]"),
            "DIRECT": ("INFO", "[i]"),
            "WARN": ("WARNING", "[!]"),
            "FAIL": ("ERROR", "[FAIL]"),
            "PLAN": ("LIST", "[PLAN]"),
            "SKIP": ("OK", "[SKIP]"),
        }
        icon_name, fallback = status_icons.get(status, ("INFO", f"[{status}]"))
        icon = self._log_icon(icon_name, fallback)
        if include_label and icon != fallback:
            return f"{icon} <b>{status}</b>"
        return icon

    def _log_icon(self, icon_name: str, fallback: str) -> str:
        if not self.use_log_icons or pysm_icons is None:
            return fallback
        try:
            html = getattr(pysm_icons, icon_name)(14)
        except Exception:
            return fallback
        return html or fallback

    def _get_current_version_label(self, pkg: PackageInfo) -> str:
        normalized_name = pkg.name.lower().replace("_", "-")
        installed_version = self.installed_packages.get(normalized_name)
        if installed_version:
            return installed_version
        return "не установлен"

    def _get_target_version_label(self, pkg: PackageInfo) -> str:
        if pkg.direct_reference:
            return ""

        effective_spec = self._get_effective_requirement_spec(pkg)
        if effective_spec != pkg.to_spec():
            try:
                effective_req = Requirement(effective_spec)
                exact_versions = [
                    spec.version
                    for spec in effective_req.specifier
                    if spec.operator == "==" and "*" not in spec.version
                ]
                if len(exact_versions) == 1:
                    return exact_versions[0]
            except Exception:
                pass

        exact_version = self._get_exact_pinned_version(pkg)
        if exact_version:
            return exact_version

        constraint = self._get_constraint_for_package(pkg)
        if constraint:
            exact_constraint = self._get_exact_pinned_version(constraint)
            if exact_constraint:
                return exact_constraint

        return ""

    def _get_exact_pinned_version(self, pkg: PackageInfo) -> str:
        if not Requirement:
            return ""
        try:
            req = Requirement(pkg.to_spec())
        except Exception:
            return ""

        exact_versions = [
            spec.version
            for spec in req.specifier
            if spec.operator == "==" and "*" not in spec.version
        ]
        if len(exact_versions) == 1:
            return exact_versions[0]
        return ""

    def _get_constraint_for_package(self, pkg: PackageInfo) -> PackageInfo | None:
        normalized_name = pkg.name.lower().replace("_", "-")
        return self.plan.package_constraints.get(normalized_name)

    def _is_constraint_satisfied(self, pkg: PackageInfo, current_version_str: str) -> bool:
        constraint = self._get_constraint_for_package(pkg)
        if not constraint or not Requirement or not Version:
            return True
        if constraint.direct_reference:
            return True
        try:
            req = Requirement(constraint.to_spec())
            if not req.specifier:
                return True
            return req.specifier.contains(Version(current_version_str))
        except Exception:
            return True

    def _get_effective_requirement_spec(self, pkg: PackageInfo) -> str:
        if pkg.name.lower().replace("_", "-") not in {"torch", "torchvision", "torchaudio"}:
            return pkg.to_spec()
        if not (self.system_info.cuda and self.system_info.cuda.selected_version):
            return pkg.to_spec()
        expected_tag = self._cuda_local_tag(self.system_info.cuda.selected_version)
        if not expected_tag:
            return pkg.to_spec()
        return self._replace_torch_local_tag(pkg, expected_tag)

    def _check_uv_available(self) -> bool:
        logging.info("\nПроверка наличия менеджера пакетов UV...")
        version_cmd = [str(self.python_executable), "-m", "uv", "--version"]
        success, _, _ = run_command(version_cmd)
        if success:
            logging.info("<i>UV найден (будет использован для построения команд)</i>")
            return True
        logging.info("<i>UV не найден (команды будут построены для pip)</i>")
        return False

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
            cmd = [str(self.python_executable), "-m", "uv", "pip", "install"]
        else:
            cmd = [str(self.python_executable), "-m", "pip", "install"]

        cmd.extend(self.plan.pip_options)
        cmd = self._with_accelerator_resolver_options(cmd)
        return cmd

    def _run_install_command(self, cmd: List[str], category_name: str) -> bool:
        logging.info(f"Команда:<i> {' '.join(cmd)}</i>")
        if self.plan_only:
            logging.info(f"Режим плана: категория <b><i>{category_name}</i></b> не устанавливалась.\n")
            self._set_category_result(category_name, "PLAN", "запланировано")
            return True

        success, stdout, stderr = run_command_streaming(
            cmd,
            progress_title=f"Установка: {category_name}",
        )
        if not success:
            logging.error(f"ОШИБКА при установке категории <b><i>{category_name}</i></b>.\n")
            logging.error(f"Stderr: {stderr}")
            self.failures.append(category_name)
            self._set_category_result(category_name, "FAIL", "ошибка установки")
            return False
        else:
            logging.info(f"Категория <b><i>{category_name}</i></b> успешно установлена.\n")
            self._set_category_result(category_name, "OK", "установлено/обновлено")
            return True
            
    def _install_regular_packages(self):
        if not self.plan.regular_packages:
            self._set_category_result("Обычные пакеты", "SKIP", "нет требований")
            return
        packages_to_install = self._filter_packages_to_install(self.plan.regular_packages)
        if not packages_to_install:
            self._set_category_result("Обычные пакеты", "SKIP", "уже соответствует")
            return
        logging.info("\n<b><i>Установка обычных пакетов...</i></b>")
        cmd = self._build_base_command()
        if self.force_upgrade: cmd.append("--upgrade")
        for pkg in packages_to_install:
            cmd.extend(self._package_install_args(pkg))
        self._run_install_command(cmd, "Обычные пакеты")

    def _package_install_args(self, pkg: PackageInfo) -> List[str]:
        spec = pkg.to_spec()
        if spec.startswith("-e "):
            return ["-e", spec[3:].strip()]
        return [spec]
        
    def _install_torch_packages(self):
        if not self.plan.torch_packages:
            self._set_category_result("Torch", "SKIP", "нет требований")
            return
        packages_to_install = self._filter_packages_to_install(self.plan.torch_packages)
        if not packages_to_install:
            self._set_category_result("Torch", "SKIP", "уже соответствует")
            return
        logging.info("\n<b><i>Установка пакетов Torch...</i></b>")
        if not self._validate_torch_direct_references(packages_to_install):
            self._set_category_result("Torch", "FAIL", "блокирующий конфликт wheel/URL")
            return
        cmd = self._build_base_command()
        cmd.append("--upgrade") # Torch лучше всегда обновлять до нужной версии
        if self._torch_family_needs_cuda_reinstall(packages_to_install):
            cmd = self._add_reinstall_options(cmd, packages_to_install)
        if self.plan.torch_index_url:
            cmd = self._without_option(cmd, "--index-url", "-i")
            cmd.extend(["--index-url", self.plan.torch_index_url])
        cmd.extend(self._get_torch_specs(packages_to_install))
        if self._run_install_command(cmd, "Torch") and not self.plan_only:
            if self._verify_torch_cuda_installation(packages_to_install):
                self._set_category_result("Torch", "OK", "установлено/обновлено, CUDA проверена")
            else:
                self._set_category_result("Torch", "FAIL", "ошибка проверки CUDA")

    def _install_onnx_packages(self):
        if not self.plan.onnx_packages:
            self._set_category_result("ONNX", "SKIP", "нет требований")
            return
        packages_to_install = self._filter_packages_to_install(self._normalize_onnx_packages(self.plan.onnx_packages))
        if not packages_to_install:
            self._set_category_result("ONNX", "SKIP", "уже соответствует")
            return
        logging.info("\n<b><i>Установка пакетов ONNX...</i></b>")
        cmd = self._build_base_command()
        cmd.append("--upgrade")
        if self.system_info.gpu and self.system_info.gpu.generation == "blackwell":
            cmd.append("--pre")
        cmd.extend(self._get_onnx_specs(packages_to_install))
        if self._run_install_command(cmd, "ONNX") and not self.plan_only:
            if self._verify_onnx_runtime_installation():
                self._set_category_result("ONNX", "OK", "установлено/обновлено, provider проверен")
            else:
                self._set_category_result("ONNX", "FAIL", "ошибка проверки provider")

    def _install_triton_packages(self):
        if not self.plan.triton_packages:
            self._set_category_result("Triton", "SKIP", "нет требований")
            return
        packages_to_install = self._filter_packages_to_install(self.plan.triton_packages)
        if not packages_to_install:
            self._set_category_result("Triton", "SKIP", "уже соответствует")
            return
        logging.info("\n<b><i>Установка пакетов Triton...</i></b>")
        cmd = self._build_base_command()
        cmd.append("--upgrade")
        cmd.append("triton-windows")
        self._run_install_command(cmd, "Triton")

    def _install_insightface_packages(self):
        if not self.plan.insightface_packages:
            self._set_category_result("Insightface", "SKIP", "нет требований")
            return
        packages_to_install = self._filter_packages_to_install(self.plan.insightface_packages)
        if not packages_to_install:
            self._set_category_result("Insightface", "SKIP", "уже соответствует")
            return
        logging.info("\n<b><i>Установка пакетов Insightface...</i></b>")
        cmd = self._build_base_command()
        cmd.append("--upgrade")
        logging.info("Insightface требует <b>numpy==1.26.4</b>; эта версия будет установлена принудительно.")
        cmd.extend([self._get_insightface_spec(), "numpy==1.26.4"])
        self._run_install_command(cmd, "Insightface")

    def _get_insightface_spec(self) -> str:
        tag = self._get_python_abi_tag()
        wheel_url = INSIGHTFACE_WINDOWS_WHEEL_URLS.get(tag)
        if wheel_url:
            logging.info(f"Выбран wheel insightface для Python ABI <b>{tag}</b>.")
            return wheel_url

        logging.warning(
            f"Готовый wheel insightface для Python ABI {tag or 'unknown'} не настроен. "
            "Будет использована обычная установка insightface==0.7.3; "
            "она может потребовать Microsoft C++ Build Tools."
        )
        return "insightface==0.7.3"

    def _get_python_abi_tag(self) -> str:
        cmd = [
            str(self.python_executable),
            "-c",
            "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')",
        ]
        success, stdout, stderr = run_command(cmd)
        if not success:
            logging.warning(f"Не удалось определить ABI целевого Python: {stderr}")
            return ""
        return stdout.strip()

    def _get_installed_torch_info(self) -> Dict[str, object]:
        cmd = [
            str(self.python_executable),
            "-c",
            (
                "import json\n"
                "try:\n"
                " import torch\n"
                " print(json.dumps({"
                "'installed': True, "
                "'version': torch.__version__, "
                "'cuda_version': getattr(torch.version, 'cuda', None), "
                "'cuda_available': bool(torch.cuda.is_available())"
                "}))\n"
                "except Exception as e:\n"
                " print(json.dumps({'installed': False, 'error': str(e)}))\n"
            ),
        ]
        success, stdout, stderr = run_command(cmd)
        if not success:
            logging.debug(f"Не удалось проверить установленный Torch: {stderr}")
            return {"installed": False, "error": stderr}
        try:
            info = json.loads(stdout)
        except json.JSONDecodeError:
            return {"installed": False, "error": stdout}
        if info.get("installed"):
            logging.info(
                f"{self._status_prefix('DIRECT')} Torch в целевом Python: <b>{info.get('version')}</b>, "
                f"CUDA build: <b>{info.get('cuda_version') or 'нет'}</b>, "
                f"torch.cuda.is_available(): <b>{info.get('cuda_available')}</b>"
            )
            logging.info("")
        return info

    def _needs_torch_cuda_reinstall(self, pkg: PackageInfo) -> bool:
        """Определяет, нужно ли переустановить Torch-family пакет под выбранную CUDA.

        Для `torch` проверяется фактический CUDA build через импорт модуля.
        Для `torchvision` и `torchaudio` CUDA-сборка видна в local tag версии
        (`+cu128`, `+cpu`), поэтому нельзя подменять их состояние состоянием
        базового `torch`.
        """
        normalized_name = pkg.name.lower().replace("_", "-")
        if normalized_name not in {"torch", "torchvision", "torchaudio"}:
            return False
        if not (self.system_info.cuda and self.system_info.cuda.selected_version):
            return False

        expected_cuda = self.system_info.cuda.selected_version
        expected_tag = self._cuda_local_tag(expected_cuda)

        if normalized_name == "torch":
            if not self.installed_torch_info.get("installed"):
                return False
            current_version = str(self.installed_torch_info.get("version") or "")
            current_cuda = self.installed_torch_info.get("cuda_version")
            if not current_cuda:
                return True
            if str(current_cuda) != str(expected_cuda):
                return True
            if expected_tag and expected_tag not in current_version:
                return True
            return not bool(self.installed_torch_info.get("cuda_available"))

        installed_version = self.installed_packages.get(normalized_name)
        if not installed_version:
            return False
        if not self.installed_torch_info.get("installed"):
            return True

        current_cuda = self.installed_torch_info.get("cuda_version")
        if not current_cuda or str(current_cuda) != str(expected_cuda):
            return True
        if not self.installed_torch_info.get("cuda_available"):
            return True
        if expected_tag and expected_tag not in installed_version:
            return True
        return False

    def _torch_family_needs_cuda_reinstall(self, packages: List[PackageInfo]) -> bool:
        return any(self._needs_torch_cuda_reinstall(pkg) for pkg in packages)

    def _verify_torch_cuda_installation(self, packages: List[PackageInfo]) -> bool:
        """Проверяет, что после установки Torch-family соответствует выбранной CUDA.

        Успешный `pip install` недостаточен: resolver может оставить CPU wheel
        или несовместимый companion wheel. Поэтому после установки проверяется
        импорт `torch`, его CUDA build/availability и версии запрошенных
        `torchvision`/`torchaudio`.
        """
        if not (self.system_info.cuda and self.system_info.cuda.selected_version):
            return True

        family_info = self._get_installed_torch_family_info(packages)
        info = family_info.get("torch", {})
        expected_cuda = self.system_info.cuda.selected_version
        expected_tag = self._cuda_local_tag(expected_cuda)

        if not info.get("installed"):
            self.failures.append("Torch verification")
            logging.error("ОШИБКА: после установки Torch не импортируется в целевом Python.")
            return False
        if str(info.get("cuda_version") or "") != str(expected_cuda):
            self.failures.append("Torch CUDA verification")
            logging.error(
                f"ОШИБКА: после установки Torch имеет CUDA build "
                f"<b>{info.get('cuda_version') or 'нет'}</b>, ожидалось <b>{expected_cuda}</b>."
            )
            return False
        if not info.get("cuda_available"):
            self.failures.append("Torch CUDA availability")
            logging.error("ОШИБКА: Torch установлен с CUDA build, но torch.cuda.is_available() вернул False.")
            return False

        for package_name in self._torch_family_names(packages):
            if package_name == "torch":
                continue
            package_info = family_info.get(package_name, {})
            if not package_info.get("installed"):
                self.failures.append(f"{package_name} verification")
                logging.error(f"ОШИБКА: после установки {package_name} не импортируется в целевом Python.")
                return False
            version = str(package_info.get("version") or "")
            if expected_tag and expected_tag not in version:
                self.failures.append(f"{package_name} CUDA verification")
                logging.error(
                    f"ОШИБКА: после установки {package_name} имеет версию <b>{version or 'неизвестно'}</b>, "
                    f"ожидалась CUDA-сборка <b>{expected_tag}</b>."
                )
                return False
        return True

    def _torch_family_names(self, packages: List[PackageInfo]) -> List[str]:
        """Возвращает уникальные Torch-family имена, которые нужно проверять."""
        names = []
        for pkg in packages:
            normalized_name = pkg.name.lower().replace("_", "-")
            if normalized_name in {"torch", "torchvision", "torchaudio"} and normalized_name not in names:
                names.append(normalized_name)
        if "torch" not in names:
            names.insert(0, "torch")
        return names

    def _get_installed_torch_family_info(self, packages: List[PackageInfo]) -> Dict[str, Dict[str, object]]:
        """Собирает live-информацию о Torch-family из целевого Python.

        `pip list` показывает только версии dist-пакетов. Для `torch` этого
        недостаточно: CUDA build и `torch.cuda.is_available()` доступны только
        через импорт модуля в том Python, куда ставятся зависимости.
        """
        names = self._torch_family_names(packages)
        cmd = [
            str(self.python_executable),
            "-c",
            (
                "import importlib, json\n"
                f"names = {json.dumps(names)}\n"
                "result = {}\n"
                "for name in names:\n"
                " try:\n"
                "  module = importlib.import_module(name)\n"
                "  data = {'installed': True, 'version': getattr(module, '__version__', None)}\n"
                "  if name == 'torch':\n"
                "   data['cuda_version'] = getattr(module.version, 'cuda', None)\n"
                "   data['cuda_available'] = bool(module.cuda.is_available())\n"
                "  result[name] = data\n"
                " except Exception as e:\n"
                "  result[name] = {'installed': False, 'error': str(e)}\n"
                "print(json.dumps(result))\n"
            ),
        ]
        success, stdout, stderr = run_command(cmd)
        if not success:
            logging.debug(f"Не удалось проверить Torch-family пакеты: {stderr}")
            return {name: {"installed": False, "error": stderr} for name in names}
        try:
            return json.loads(stdout)
        except json.JSONDecodeError:
            return {name: {"installed": False, "error": stdout} for name in names}

    def _verify_onnx_runtime_installation(self) -> bool:
        expected_provider = self._expected_onnx_provider()
        if not expected_provider:
            return True

        info = self._get_installed_onnx_info()
        if not info.get("installed"):
            self.failures.append("ONNX Runtime verification")
            logging.error("ОШИБКА: после установки ONNX Runtime не импортируется в целевом Python.")
            return False

        providers = info.get("providers") or []
        if expected_provider not in providers:
            self.failures.append("ONNX Runtime provider verification")
            logging.error(
                f"ОШИБКА: ONNX Runtime не содержит провайдер <b>{expected_provider}</b>. "
                f"Доступные провайдеры: <i>{providers}</i>."
            )
            return False
        return True

    def _get_installed_onnx_info(self) -> Dict[str, object]:
        cmd = [
            str(self.python_executable),
            "-c",
            (
                "import json\n"
                "try:\n"
                " import onnxruntime as ort\n"
                " print(json.dumps({"
                "'installed': True, "
                "'version': getattr(ort, '__version__', None), "
                "'providers': ort.get_available_providers()"
                "}))\n"
                "except Exception as e:\n"
                " print(json.dumps({'installed': False, 'error': str(e)}))\n"
            ),
        ]
        success, stdout, stderr = run_command(cmd)
        if not success:
            return {"installed": False, "error": stderr}
        try:
            info = json.loads(stdout)
        except json.JSONDecodeError:
            return {"installed": False, "error": stdout}
        if info.get("installed"):
            logging.info(
                f"{self._status_prefix('DIRECT')} ONNX Runtime в целевом Python: <b>{info.get('version')}</b>, "
                f"providers: <b>{info.get('providers')}</b>"
            )
            logging.info("")
        return info

    def _expected_onnx_provider(self) -> str:
        target_name = self.plan.onnx_package_name or ""
        if target_name == "onnxruntime-gpu":
            return "CUDAExecutionProvider"
        if target_name == "onnxruntime-directml":
            return "DmlExecutionProvider"
        return ""

    def _cuda_local_tag(self, cuda_version: str) -> str:
        match = re.match(r"(\d+)\.(\d+)", str(cuda_version))
        if not match:
            return ""
        return f"+cu{match.group(1)}{match.group(2)}"

    def _add_reinstall_options(self, cmd: List[str], packages: List[PackageInfo]) -> List[str]:
        if self.use_uv:
            for pkg in packages:
                if not pkg.direct_reference:
                    cmd.extend(["--reinstall-package", pkg.name])
            return cmd
        cmd.append("--force-reinstall")
        return cmd

    def _get_onnx_specs(self, packages: List[PackageInfo]) -> List[str]:
        specs: List[str] = []
        for pkg in packages:
            specs.append(pkg.to_spec())
        return specs

    def _normalize_onnx_packages(self, packages: List[PackageInfo]) -> List[PackageInfo]:
        target_name = self.plan.onnx_package_name or "onnxruntime"
        normalized: List[PackageInfo] = []
        for pkg in packages:
            if pkg.direct_reference:
                normalized.append(pkg)
                continue
            original_spec = pkg.to_spec()
            marker = ""
            if ";" in original_spec:
                _, marker_part = original_spec.split(";", 1)
                marker = f"; {marker_part.strip()}"
            normalized_spec = f"{target_name}{pkg.version or ''}{marker}"
            if pkg.name != target_name or original_spec != normalized_spec:
                self._record_requirement_rewrite(original_spec, normalized_spec, "ONNX Runtime")
            normalized.append(PackageInfo(
                name=target_name,
                original_line=pkg.original_line,
                package_type=pkg.package_type,
                version=pkg.version,
                extras=pkg.extras,
                spec=normalized_spec,
                source_file=pkg.source_file,
                line_number=pkg.line_number,
                direct_reference=False,
            ))
        return normalized

    def _get_torch_specs(self, packages: List[PackageInfo]) -> List[str]:
        return [self._normalize_torch_spec(pkg) for pkg in packages]

    def _normalize_torch_spec(self, pkg: PackageInfo) -> str:
        spec = pkg.to_spec()
        if not (self.system_info.cuda and self.system_info.cuda.selected_version):
            return spec

        expected_tag = self._cuda_local_tag(self.system_info.cuda.selected_version)
        if not expected_tag:
            return spec

        normalized = self._replace_torch_local_tag(pkg, expected_tag)
        if normalized != spec:
            self._record_requirement_rewrite(spec, normalized, f"CUDA {self.system_info.cuda.selected_version}")
        return normalized

    def _replace_torch_local_tag(self, pkg: PackageInfo, expected_tag: str) -> str:
        # If a requirements file pins a CPU or different CUDA local tag, keep the public
        # version but force the CUDA flavor chosen by the system analysis.
        pattern = re.compile(r"(\b" + re.escape(pkg.name) + r"(?:\[[^\]]+\])?==[^;\s+]+)\+(cpu|cu\d+)", re.IGNORECASE)
        return pattern.sub(r"\1" + expected_tag, pkg.to_spec())

    def _record_requirement_rewrite(self, original: str, normalized: str, reason: str) -> None:
        message = f"{original} -> {normalized} ({reason})"
        if message not in self.plan.requirement_rewrites:
            self.plan.requirement_rewrites.append(message)
        logging.warning(f"  [GPU] Требование <i>{original}</i> будет установлено как <b>{normalized}</b> ({reason}).")

    def _validate_torch_direct_references(self, packages: List[PackageInfo]) -> bool:
        if not (self.system_info.cuda and self.system_info.cuda.selected_version):
            return True

        expected_tag = self._cuda_local_tag(self.system_info.cuda.selected_version).lstrip("+")
        if not expected_tag:
            return True

        valid = True
        for pkg in packages:
            if not pkg.direct_reference:
                continue

            spec = pkg.to_spec()
            detected_tag = self._detect_torch_wheel_cuda_tag(spec)
            if detected_tag and detected_tag != expected_tag:
                message = (
                    f"Прямая ссылка/локальный wheel Torch <i>{spec}</i> указывает на <b>{detected_tag}</b>, "
                    f"а выбрана CUDA <b>{expected_tag}</b>. Автоматически переписать wheel/URL нельзя."
                )
                logging.error(message)
                if message not in self.plan.diagnostics:
                    self.plan.diagnostics.append(message)
                self.failures.append("Torch direct reference CUDA mismatch")
                valid = False
        return valid

    def _detect_torch_wheel_cuda_tag(self, spec: str) -> str:
        lower = spec.lower()
        match = re.search(r"(?:\+|-)(cpu|cu\d{3})", lower)
        if match:
            return match.group(1)
        if "/cpu/" in lower or "\\cpu\\" in lower:
            return "cpu"
        match = re.search(r"/(cu\d{3})/", lower)
        if match:
            return match.group(1)
        return ""

    def _without_option(self, cmd: List[str], long_name: str, short_name: str) -> List[str]:
        cleaned: List[str] = []
        idx = 0
        while idx < len(cmd):
            token = cmd[idx]
            if token in (long_name, short_name):
                idx += 2
                continue
            if token.startswith(f"{long_name}="):
                idx += 1
                continue
            cleaned.append(token)
            idx += 1
        return cleaned

    def _with_accelerator_resolver_options(self, cmd: List[str]) -> List[str]:
        if not (self.system_info.cuda and self.system_info.cuda.selected_version):
            return cmd

        if self.use_uv and self.plan.torch_backend:
            if "--torch-backend" not in cmd:
                cmd.extend(["--torch-backend", self.plan.torch_backend])
            return cmd

        if self.plan.torch_index_url and self.plan.torch_index_url not in cmd:
            cmd.extend(["--extra-index-url", self.plan.torch_index_url])
        return cmd
