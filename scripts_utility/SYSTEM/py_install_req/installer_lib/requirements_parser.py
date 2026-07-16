# installer_lib/requirements_parser.py

import logging
import os
import re
import shlex
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .models import InstallationPlan, PackageInfo, PackageType, SystemInfo
from .config import (
    TORCH_FAMILY, ONNXRUNTIME_FAMILY, INSIGHTFACE_FAMILY, TRITON_FAMILY,
    TORCH_INDEX_URLS
)

try:
    import toml
except ImportError:
    toml = None

try:
    from packaging.requirements import InvalidRequirement, Requirement
except ImportError:
    InvalidRequirement = Exception
    Requirement = None


class RequirementsParser:
    """
    Читает requirements.txt / pyproject.toml и создает план установки.

    Для requirements.txt поддерживаются pip-options, include-файлы, constraints,
    URL/VCS/local path-зависимости, line continuation и PEP 508 requirements.
    """

    VALUE_OPTIONS = {
        "-i", "--index-url",
        "--extra-index-url",
        "-f", "--find-links",
        "--trusted-host",
        "--no-binary",
        "--only-binary",
        "--platform",
        "--python-version",
        "--implementation",
        "--abi",
    }
    FLAG_OPTIONS = {
        "--pre",
        "--no-index",
        "--prefer-binary",
        "--use-pep517",
        "--no-use-pep517",
    }
    LOCAL_PATH_OPTIONS = {"-f", "--find-links"}
    REQUIREMENT_OPTIONS = {"-r", "--requirement"}
    CONSTRAINT_OPTIONS = {"-c", "--constraint"}
    EDITABLE_OPTIONS = {"-e", "--editable"}

    def __init__(self, system_info: SystemInfo, marker_environment: Optional[Dict[str, str]] = None):
        self.system_info = system_info
        self.marker_environment = marker_environment
        self._pip_options: List[str] = []
        self._constraint_files: List[str] = []
        self._package_constraints: Dict[str, PackageInfo] = {}
        self._included_files: List[str] = []
        self._diagnostics: List[str] = []

    def parse(self, requirements_path: Path) -> InstallationPlan:
        """Парсит файл зависимостей и возвращает категорийный план установки.

        Метод очищает внутреннее состояние перед каждым запуском, чтобы один
        экземпляр парсера можно было безопасно использовать повторно в тестах
        или диагностике.
        """
        logging.info(f"Чтение файла: <i>{requirements_path}</i>")
        self._pip_options = []
        self._constraint_files = []
        self._package_constraints = {}
        self._included_files = []
        self._diagnostics = []

        if requirements_path.name == "pyproject.toml":
            if not toml:
                raise ImportError("Для парсинга pyproject.toml установите библиотеку 'toml' (`pip install toml`).")
            packages = self._parse_pyproject(requirements_path)
        else:
            packages = self._parse_requirements_txt(requirements_path)

        self._validate_insightface_requirements(packages)

        logging.info(f"Найдено <b>{len(packages)}</b> пакета(ов).<br>")
        if self._included_files:
            logging.info(f"Подключено requirements-файлов: <b>{len(self._included_files)}</b>")
        if self._constraint_files:
            logging.info(f"Найдено constraints-файлов: <b>{len(self._constraint_files)}</b>")
        for diagnostic in self._diagnostics:
            logging.warning(diagnostic)

        return self._create_plan(packages)

    def _validate_insightface_requirements(self, packages: List[PackageInfo]) -> None:
        """Require unambiguous exact InsightFace and NumPy pins.

        InsightFace is installed separately with ``--no-deps``. Therefore its
        own version and the NumPy compatibility version must come directly from
        the parsed project requirements before the installer can change the
        target environment.
        """
        insightface_packages = self._packages_named(packages, "insightface")
        if not insightface_packages:
            return

        if len(insightface_packages) != 1:
            raise ValueError(
                "Для InsightFace найдено несколько требований. Оставьте ровно одну строку "
                "вида insightface==<версия>."
            )

        numpy_packages = self._packages_named(packages, "numpy")
        if not numpy_packages:
            raise ValueError(
                "В requirements присутствует InsightFace, но отсутствует NumPy. "
                "Добавьте точное требование numpy==<версия>."
            )
        if len(numpy_packages) != 1:
            raise ValueError(
                "Для NumPy найдено несколько требований. Оставьте ровно одну строку "
                "вида numpy==<версия>."
            )

        self._require_exact_pin(insightface_packages[0], "InsightFace")
        self._require_exact_pin(numpy_packages[0], "NumPy")

    def _packages_named(self, packages: List[PackageInfo], name: str) -> List[PackageInfo]:
        normalized_name = name.lower().replace("_", "-")
        return [
            package
            for package in packages
            if package.name.lower().replace("_", "-") == normalized_name
        ]

    def _require_exact_pin(self, package: PackageInfo, display_name: str) -> str:
        location = package.source_file or "requirements"
        if package.line_number:
            location = f"{location}:{package.line_number}"

        if package.direct_reference:
            raise ValueError(
                f"{location}: {display_name} должен быть задан точной версией через ==; "
                "URL, wheel и другие прямые ссылки не допускаются."
            )
        if package.extras:
            raise ValueError(
                f"{location}: extras для {display_name} не поддерживаются в специальной "
                f"схеме установки. Используйте {package.name}==<версия>."
            )
        if not Requirement:
            raise ImportError(
                "Для проверки точных версий InsightFace и NumPy требуется библиотека packaging."
            )

        try:
            requirement = Requirement(package.to_spec())
        except Exception as error:
            raise ValueError(
                f"{location}: не удалось разобрать требование {display_name}: {error}"
            ) from None

        specifiers = list(requirement.specifier)
        if (
            len(specifiers) != 1
            or specifiers[0].operator != "=="
            or not specifiers[0].version
            or "*" in specifiers[0].version
        ):
            raise ValueError(
                f"{location}: {display_name} должен быть закреплён одной точной версией "
                f"через ==; получено: {package.to_spec()}"
            )
        return specifiers[0].version

    def _parse_requirements_txt(self, file_path: Path) -> List[PackageInfo]:
        packages: List[PackageInfo] = []
        self._parse_requirements_file(file_path.resolve(), packages, seen=set())
        return packages

    def _parse_requirements_file(
        self,
        file_path: Path,
        packages: List[PackageInfo],
        seen: Set[Path],
    ) -> None:
        resolved = file_path.resolve()
        if resolved in seen:
            self._diagnostics.append(f"Пропущен повторный include requirements: {resolved}")
            return
        if not resolved.is_file():
            self._diagnostics.append(f"Файл requirements не найден: {resolved}")
            return

        seen.add(resolved)
        self._included_files.append(str(resolved))

        for line_number, logical_line in self._read_logical_lines(resolved):
            parsed_packages = self._parse_requirement_entry(logical_line, resolved, line_number, seen)
            packages.extend(parsed_packages)

    def _read_logical_lines(self, file_path: Path) -> List[Tuple[int, str]]:
        logical_lines: List[Tuple[int, str]] = []
        buffer = ""
        start_line = 0

        with open(file_path, "r", encoding="utf-8-sig") as f:
            for line_number, raw_line in enumerate(f, start=1):
                line = raw_line.rstrip("\n")
                stripped = line.rstrip()
                if not buffer:
                    start_line = line_number

                if stripped.endswith("\\"):
                    buffer += stripped[:-1] + " "
                    continue

                buffer += line
                cleaned = self._strip_inline_comment(buffer).strip()
                if cleaned:
                    logical_lines.append((start_line, cleaned))
                buffer = ""

        if buffer.strip():
            cleaned = self._strip_inline_comment(buffer).strip()
            if cleaned:
                logical_lines.append((start_line, cleaned))

        return logical_lines

    def _strip_inline_comment(self, line: str) -> str:
        in_single = False
        in_double = False
        for idx, char in enumerate(line):
            if char == "'" and not in_double:
                in_single = not in_single
            elif char == '"' and not in_single:
                in_double = not in_double
            elif char == "#" and not in_single and not in_double:
                if idx == 0 or line[idx - 1].isspace():
                    return line[:idx]
        return line

    def _parse_requirement_entry(
        self,
        line: str,
        file_path: Path,
        line_number: int,
        seen: Set[Path],
    ) -> List[PackageInfo]:
        """Разбирает одну логическую строку requirements.txt.

        В одной строке могут быть как pip-options (`-r`, `-c`, `--index-url`),
        так и собственно requirement. Опции сохраняются в плане, а include и
        constraints обрабатываются сразу, чтобы предварительный план видел
        реальные ограничения версий.
        """
        try:
            tokens = shlex.split(line, comments=False, posix=False)
        except ValueError as e:
            self._diagnostics.append(f"{file_path}:{line_number}: не удалось разобрать строку requirements: {e}")
            return []

        if not tokens:
            return []

        first_token = self._clean_token(tokens[0])
        if not first_token.startswith("-"):
            requirement_text = self._strip_hash_options(line, file_path, line_number)
            package = self._parse_package(requirement_text, file_path, line_number)
            return [package] if package else []

        packages: List[PackageInfo] = []
        idx = 0
        while idx < len(tokens):
            token = self._clean_token(tokens[idx])

            if token in self.REQUIREMENT_OPTIONS:
                include_path, idx = self._read_option_value(tokens, idx, file_path, line_number, token)
                if include_path:
                    self._parse_requirements_file(self._resolve_relative_path(file_path, include_path), packages, seen)
                continue

            if token.startswith("--requirement="):
                include_path = token.split("=", 1)[1]
                self._parse_requirements_file(self._resolve_relative_path(file_path, include_path), packages, seen)
                idx += 1
                continue

            if token in self.CONSTRAINT_OPTIONS:
                constraint_path, idx = self._read_option_value(tokens, idx, file_path, line_number, token)
                if constraint_path:
                    resolved = self._resolve_relative_path(file_path, constraint_path)
                    self._register_constraint_file(resolved)
                continue

            if token.startswith("--constraint="):
                constraint_path = token.split("=", 1)[1]
                resolved = self._resolve_relative_path(file_path, constraint_path)
                self._register_constraint_file(resolved)
                idx += 1
                continue

            if token in self.EDITABLE_OPTIONS:
                editable_value, idx = self._read_option_value(tokens, idx, file_path, line_number, token)
                if editable_value:
                    editable_spec = f"-e {self._resolve_local_direct_reference(editable_value, file_path)}"
                    package = PackageInfo(
                        name=self._name_from_direct_reference(editable_value),
                        original_line=line,
                        package_type=PackageType.REGULAR,
                        spec=editable_spec,
                        source_file=str(file_path),
                        line_number=line_number,
                        direct_reference=True,
                    )
                    packages.append(package)
                continue

            if token in self.VALUE_OPTIONS:
                value, idx = self._read_option_value(tokens, idx, file_path, line_number, token)
                if value:
                    value = self._normalize_option_value(token, value, file_path)
                    self._add_pip_option([token, value])
                continue

            option_name = self._split_long_option_name(token)
            if option_name in self.VALUE_OPTIONS:
                value = self._normalize_option_value(option_name, token.split("=", 1)[1], file_path)
                self._add_pip_option([option_name, value])
                idx += 1
                continue

            if token in self.FLAG_OPTIONS:
                self._add_pip_option([token])
                idx += 1
                continue

            if token.startswith("-"):
                self._diagnostics.append(f"{file_path}:{line_number}: неподдержанная pip-опция '{token}' пропущена.")
                idx += 1
                continue

            requirement_text = " ".join(self._clean_token(t) for t in tokens[idx:])
            package = self._parse_package(requirement_text, file_path, line_number)
            if package:
                packages.append(package)
            break

        return packages

    def _read_option_value(
        self,
        tokens: List[str],
        idx: int,
        file_path: Path,
        line_number: int,
        option: str,
    ) -> Tuple[Optional[str], int]:
        if idx + 1 >= len(tokens):
            self._diagnostics.append(f"{file_path}:{line_number}: у опции '{option}' нет значения.")
            return None, idx + 1
        return self._clean_token(tokens[idx + 1]), idx + 2

    def _register_constraint_file(self, file_path: Path) -> None:
        resolved = file_path.resolve()
        resolved_str = str(resolved)
        if resolved_str not in self._constraint_files:
            self._constraint_files.append(resolved_str)
            self._add_pip_option(["--constraint", resolved_str])
        self._parse_constraint_file(resolved, seen=set())

    def _parse_constraint_file(self, file_path: Path, seen: Set[Path]) -> None:
        resolved = file_path.resolve()
        if resolved in seen:
            self._diagnostics.append(f"Пропущен повторный include constraints: {resolved}")
            return
        if not resolved.is_file():
            self._diagnostics.append(f"Файл constraints не найден: {resolved}")
            return

        seen.add(resolved)
        for line_number, logical_line in self._read_logical_lines(resolved):
            self._parse_constraint_entry(logical_line, resolved, line_number, seen)

    def _parse_constraint_entry(
        self,
        line: str,
        file_path: Path,
        line_number: int,
        seen: Set[Path],
    ) -> None:
        try:
            tokens = shlex.split(line, comments=False, posix=False)
        except ValueError as e:
            self._diagnostics.append(f"{file_path}:{line_number}: не удалось разобрать строку constraints: {e}")
            return

        if not tokens:
            return

        token = self._clean_token(tokens[0])
        if token in self.CONSTRAINT_OPTIONS:
            constraint_path, _ = self._read_option_value(tokens, 0, file_path, line_number, token)
            if constraint_path:
                nested = self._resolve_relative_path(file_path, constraint_path)
                nested_str = str(nested)
                if nested_str not in self._constraint_files:
                    self._constraint_files.append(nested_str)
                    self._add_pip_option(["--constraint", nested_str])
                self._parse_constraint_file(nested, seen)
            return

        if token.startswith("--constraint="):
            nested = self._resolve_relative_path(file_path, token.split("=", 1)[1])
            nested_str = str(nested)
            if nested_str not in self._constraint_files:
                self._constraint_files.append(nested_str)
                self._add_pip_option(["--constraint", nested_str])
            self._parse_constraint_file(nested, seen)
            return

        if token.startswith("-"):
            self._diagnostics.append(f"{file_path}:{line_number}: опция '{token}' в constraints не применяется к предварительному анализу.")
            return

        constraint_text = self._strip_hash_options(line, file_path, line_number)
        constraint = self._parse_package(constraint_text, file_path, line_number)
        if not constraint:
            return

        normalized_name = constraint.name.lower().replace("_", "-")
        self._package_constraints[normalized_name] = constraint

    def _parse_pyproject(self, file_path: Path) -> List[PackageInfo]:
        packages: List[PackageInfo] = []
        try:
            data = toml.load(file_path)
            dependencies = data.get("project", {}).get("dependencies", [])
            for dep_line in dependencies:
                package = self._parse_package(dep_line, file_path, None)
                if package:
                    packages.append(package)
        except Exception as e:
            logging.error(f"Ошибка при парсинге {file_path}: {e}")
        return packages

    def _parse_package(
        self,
        text: str,
        file_path: Path,
        line_number: Optional[int],
    ) -> Optional[PackageInfo]:
        """Преобразует requirement/direct-reference в PackageInfo.

        Сначала используется строгий PEP 508 parser из `packaging`. Если строка
        не является PEP 508 requirement, включается fallback для прямых URL,
        VCS-ссылок, локальных wheel и путей, которые pip тоже умеет ставить.
        """
        spec = text.strip()
        if not spec:
            return None

        requirement = self._try_parse_pep508(spec)
        if requirement:
            if requirement.marker and not self._marker_matches(requirement, spec, file_path, line_number):
                return None
            return PackageInfo(
                name=requirement.name,
                original_line=spec,
                package_type=self._classify_package(requirement.name),
                version=str(requirement.specifier) if requirement.specifier else None,
                extras=sorted(requirement.extras),
                spec=spec,
                source_file=str(file_path),
                line_number=line_number,
                direct_reference=requirement.url is not None,
            )

        if self._looks_like_direct_reference(spec):
            name = self._name_from_direct_reference(spec)
            return PackageInfo(
                name=name,
                original_line=spec,
                package_type=self._classify_package(name),
                spec=self._resolve_local_direct_reference(spec, file_path),
                source_file=str(file_path),
                line_number=line_number,
                direct_reference=True,
            )

        location = f"{file_path}:{line_number}" if line_number else str(file_path)
        self._diagnostics.append(f"{location}: строка не похожа на поддерживаемое requirement-выражение: {spec}")
        return None

    def _marker_matches(
        self,
        requirement,
        spec: str,
        file_path: Path,
        line_number: Optional[int],
    ) -> bool:
        try:
            matches = requirement.marker.evaluate(environment=self.marker_environment)
        except Exception as e:
            location = f"{file_path}:{line_number}" if line_number else str(file_path)
            self._diagnostics.append(f"{location}: не удалось проверить environment marker для '{spec}': {e}")
            return True

        if not matches:
            location = f"{file_path}:{line_number}" if line_number else str(file_path)
            self._diagnostics.append(f"{location}: requirement пропущен по environment marker: {spec}")
        return matches

    def _strip_hash_options(self, line: str, file_path: Path, line_number: int) -> str:
        if "--hash" not in line:
            return line
        try:
            tokens = shlex.split(line, comments=False, posix=False)
        except ValueError:
            return line

        requirement_tokens: List[str] = []
        idx = 0
        while idx < len(tokens):
            token = self._clean_token(tokens[idx])
            if token == "--hash":
                idx += 2
                continue
            if token.startswith("--hash="):
                idx += 1
                continue
            requirement_tokens.append(tokens[idx])
            idx += 1

        self._diagnostics.append(
            f"{file_path}:{line_number}: --hash найден, но hash-проверка не применяется "
            "в категорийном режиме установки."
        )
        return " ".join(self._clean_token(t) for t in requirement_tokens)

    def _try_parse_pep508(self, spec: str):
        if not Requirement:
            return None
        try:
            return Requirement(spec)
        except InvalidRequirement:
            return None

    def _looks_like_direct_reference(self, spec: str) -> bool:
        lower = spec.lower()
        if re.match(r"^[a-z0-9+.-]+://", lower):
            return True
        if lower.startswith(("git+", "hg+", "svn+", "bzr+")):
            return True
        if lower.endswith((".whl", ".zip", ".tar.gz", ".tgz")):
            return True
        if spec.startswith((".", os.sep)):
            return True
        if re.match(r"^[a-zA-Z]:[\\/]", spec):
            return True
        return False

    def _name_from_direct_reference(self, spec: str) -> str:
        egg_match = re.search(r"[#&]egg=([^&]+)", spec)
        if egg_match:
            return egg_match.group(1)
        path_part = spec.split("#", 1)[0].split("?", 1)[0].rstrip("/\\")
        name = Path(path_part).name
        if name.endswith(".whl"):
            wheel_match = re.match(r"([A-Za-z0-9_.-]+?)-\d", name)
            if wheel_match:
                return wheel_match.group(1).replace("_", "-")
        return name or spec

    def _resolve_local_direct_reference(self, spec: str, file_path: Path) -> str:
        if re.match(r"^[a-zA-Z]:[\\/]", spec) or spec.startswith((".", os.sep)):
            return str(self._resolve_relative_path(file_path, spec))
        return spec

    def _resolve_relative_path(self, base_file: Path, value: str) -> Path:
        cleaned = self._clean_token(value)
        path = Path(cleaned)
        if path.is_absolute():
            return path
        return (base_file.parent / path).resolve()

    def _normalize_option_value(self, option: str, value: str, file_path: Path) -> str:
        if option not in self.LOCAL_PATH_OPTIONS:
            return value
        if re.match(r"^[a-z0-9+.-]+://", value.lower()):
            return value
        return str(self._resolve_relative_path(file_path, value))

    def _clean_token(self, token: str) -> str:
        return token.strip().strip('"').strip("'")

    def _split_long_option_name(self, token: str) -> str:
        if token.startswith("--") and "=" in token:
            return token.split("=", 1)[0]
        return token

    def _add_pip_option(self, option_tokens: List[str]) -> None:
        self._pip_options.extend(option_tokens)

    def _classify_package(self, name: str) -> PackageType:
        name_lower = name.lower().replace("_", "-")
        if name_lower in TORCH_FAMILY:
            return PackageType.TORCH
        if name_lower in ONNXRUNTIME_FAMILY:
            return PackageType.ONNXRUNTIME
        if name_lower in INSIGHTFACE_FAMILY:
            return PackageType.INSIGHTFACE
        if name_lower in TRITON_FAMILY:
            return PackageType.TRITON
        return PackageType.REGULAR

    def _create_plan(self, packages: List[PackageInfo]) -> InstallationPlan:
        """Разносит пакеты по категориям и добавляет системные решения.

        Категории нужны потому, что GPU-зависимости устанавливаются не теми же
        правилами, что обычные пакеты: Torch получает CUDA index/backend, ONNX
        может быть переписан в `onnxruntime-gpu`, а InsightFace устанавливается
        отдельно по точной версии из requirements.
        """
        plan = InstallationPlan()
        for pkg in packages:
            if pkg.package_type == PackageType.TORCH:
                plan.torch_packages.append(pkg)
            elif pkg.package_type == PackageType.ONNXRUNTIME:
                plan.onnx_packages.append(pkg)
            elif pkg.package_type == PackageType.INSIGHTFACE:
                plan.insightface_packages.append(pkg)
            elif pkg.package_type == PackageType.TRITON:
                plan.triton_packages.append(pkg)
            else:
                plan.regular_packages.append(pkg)

        plan.torch_index_url = self._get_torch_index_url()
        plan.torch_backend = self._get_torch_backend()
        plan.onnx_package_name = self._get_onnx_package_name()
        plan.pip_options = self._pip_options.copy()
        plan.constraint_files = self._constraint_files.copy()
        plan.package_constraints = self._package_constraints.copy()
        plan.included_files = self._included_files.copy()
        plan.diagnostics = self._diagnostics.copy()

        logging.info("Сформирован план установки:")
        logging.info(f"  - URL для Torch: <b>{plan.torch_index_url}</b>")
        logging.info(f"  - Torch backend: <b>{plan.torch_backend}</b>")
        logging.info(f"  - Имя пакета ONNX: <b>{plan.onnx_package_name}</b>")
        if plan.pip_options:
            logging.info(f"  - Pip options: <i>{' '.join(plan.pip_options)}</i>")
        if plan.package_constraints:
            logging.info(f"  - Constraints пакетов: <b>{len(plan.package_constraints)}</b>")
        return plan

    def _get_torch_index_url(self) -> str:
        if (
            self.system_info.gpu
            and self.system_info.gpu.vendor == "NVIDIA"
            and self.system_info.cuda
            and self.system_info.cuda.selected_version
        ):
            cuda_version_str = self.system_info.cuda.selected_version
            if cuda_version_str.startswith("12.8"):
                return TORCH_INDEX_URLS["12.8"]
            if cuda_version_str.startswith("12.6"):
                return TORCH_INDEX_URLS["12.6"]
            if cuda_version_str.startswith("12.4"):
                return TORCH_INDEX_URLS["12.4"]
            if cuda_version_str.startswith("12.1"):
                return TORCH_INDEX_URLS["12.1"]
            if cuda_version_str.startswith("11.8"):
                return TORCH_INDEX_URLS["11.8"]
        return TORCH_INDEX_URLS["cpu"]

    def _get_torch_backend(self) -> str:
        if (
            self.system_info.gpu
            and self.system_info.gpu.vendor == "NVIDIA"
            and self.system_info.cuda
            and self.system_info.cuda.selected_version
        ):
            cuda_version_str = self.system_info.cuda.selected_version
            if cuda_version_str.startswith("12.8"):
                return "cu128"
            if cuda_version_str.startswith("12.6"):
                return "cu126"
            if cuda_version_str.startswith("12.4"):
                return "cu124"
            if cuda_version_str.startswith("12.1"):
                return "cu121"
            if cuda_version_str.startswith("11.8"):
                return "cu118"
        return "cpu"

    def _get_onnx_package_name(self) -> str:
        if self.system_info.gpu:
            if self.system_info.gpu.vendor == "NVIDIA":
                return "onnxruntime-gpu"
            if self.system_info.gpu.vendor in ("AMD", "INTEL"):
                return "onnxruntime-directml"
        return "onnxruntime"
