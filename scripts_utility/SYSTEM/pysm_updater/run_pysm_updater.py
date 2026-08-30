"""
Git-обновление portable-установки PySM.

Скрипт намеренно работает консервативно:
- обновляет только файлы, которые отслеживаются удаленным репозиторием;
- не удаляет пользовательские/runtime-файлы вне Git;
- сохраняет обычные данные PySM (`config.toml` и `script_collections`) даже
  при их пересечении с файлами, которые должны прийти с GitHub;
- блокирует обычное обновление при пересечении с неизвестными локальными
  изменениями tracked-кода;
- сохраняет машинно-читаемый отчет в JSON, а полный операторский лог в обычный
  текстовый файл.

Обычно PySM запускает этот файл через свой managed runner, где stdout может
рендериться как HTML. При этом скрипт можно запустить и из обычной консоли,
поэтому логгер всегда хранит plain-text fallback.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from datetime import datetime
from html import escape as html_escape
from pathlib import Path
from urllib.parse import urlparse


try:
    from pysm_lib import pysm_context
    from pysm_lib.pysm_context import ConfigResolver
    from pysm_lib.app_constants import APPLICATION_ROOT_DIR
    from pysm_lib.pysm_icons import icons
    from pysm_lib.pysm_progress_reporter import IS_RUNNING_UNDER_PYSM, JsonProgressReporter
    from pysm_lib.pysm_report_api import ResourceNode, StandardTreeBuilder

    IS_MANAGED_RUN = True
except ImportError:
    IS_MANAGED_RUN = False
    pysm_context = None
    ConfigResolver = None
    icons = None
    IS_RUNNING_UNDER_PYSM = False
    JsonProgressReporter = None
    ResourceNode = None
    StandardTreeBuilder = None
    APPLICATION_ROOT_DIR = Path(os.getcwd()).resolve()


DEFAULT_REMOTE = "origin"
DEFAULT_BRANCH = "main"
DEFAULT_REMOTE_URL = "https://github.com/ExMent-70/PySM.git"
DEFAULT_EXPECTED_REMOTE = "github.com/ExMent-70/PySM"
BACKUP_EXCLUDED_DIRS = {
    ".codex",
    ".git",
    "__pycache__",
    "_backups",
    "_BIN",
    "_Cache",
    "_Embeddings",
    "_OUTPUT",
    "huggingface_home",
}
LOCAL_GIT_EXCLUDE_LINES = [
    "_BIN/",
    "_OUTPUT/",
    "_backups/",
    "huggingface_home/",
    "**/_Cache/",
    "**/_Embeddings/",
    "*.log",
]
LOCAL_STATE_PREVIEW_LIMIT = 25
DEFAULT_CONSOLE_PREVIEW_LIMIT = 8
BACKUP_PROGRESS_STEP = 250
GIT_PATH_CHUNK_SIZE = 100
USER_STATE_EXACT_PATHS = {"config.toml"}
USER_STATE_DIRECTORY_PREFIXES = ("script_collections/",)
GIT_PROGRESS_RE = re.compile(
    r"^(?:remote:\s*)?"
    r"(?P<phase>Counting objects|Compressing objects|Receiving objects|Resolving deltas):\s+"
    r"(?P<percent>\d{1,3})%"
    r"(?:\s+\((?P<current>\d+)/(?P<total>\d+)\))?"
    r"(?P<details>.*)$"
)
GIT_PROGRESS_PHASE_LABELS = {
    "Counting objects": "подсчет объектов",
    "Compressing objects": "сжатие объектов",
    "Receiving objects": "получение объектов",
    "Resolving deltas": "применение изменений",
}


class UpdaterError(RuntimeError):
    """Ожидаемая ошибка updater-а, которую нужно показать без Python traceback."""

    pass


class UpdateLogger:
    """Пишет один отчет обновления в консоль, plain-log и JSON.

    В managed-запуске PySM stdout рендерится как HTML, а файл лога должен
    оставаться обычным читаемым текстом. Поэтому каналы разделены: `write_html`
    печатает HTML в консоль PySM и сохраняет переданный `plain` в файл, а
    `write_file` используется для деталей, которые нужны только в сохраненном
    отчете.
    """

    def __init__(self, report_dir: Path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = report_dir / f"update_{timestamp}.log"
        self.json_path = report_dir / f"update_{timestamp}.json"
        self._file = self.log_path.open("w", encoding="utf-8")

    def close(self):
        self._file.close()

    def write(self, message: str = ""):
        print(message, flush=True)
        self._file.write(message + "\n")
        self._file.flush()

    def write_file(self, message: str = ""):
        self._file.write(message + "\n")
        self._file.flush()

    def write_html(self, html: str, plain: str = ""):
        if plain:
            self.write_file(plain)
        if not IS_MANAGED_RUN:
            if plain:
                print(plain, flush=True)
            return
        clean_html = html.replace("\n", " ").replace("\r", "")
        print(clean_html, flush=True)

    def icon_line(self, message: str, icon_name: str = "INFO", color: str | None = None):
        html = (
            '<div style="margin:2px 0;">'
            f'{icon_html(icon_name, 16, color=color)} '
            f'{html_escape(message)}'
            '</div>'
        )
        self.write_html(html, message)

    def kv_line(self, label: str, value: str, icon_name: str = "INFO"):
        plain = f"{label}: {value}"
        html = (
            '<div style="margin:2px 0;">'
            f'{icon_html(icon_name, 15)} '
            f'<b>{html_escape(label)}:</b> '
            f'<code>{html_escape(str(value))}</code>'
            '</div>'
        )
        self.write_html(html, plain)

    def section(self, title: str, icon_name: str = "INFO"):
        self.write()
        plain = f"=== {title} ==="
        html = (
            '<div style="margin:10px 0 4px 0; padding:4px 0; '
            'border-bottom:1px solid #bdbdbd; font-weight:700;">'
            f'{icon_html(icon_name, 18)} '
            f'{html_escape(title)}'
            '</div>'
        )
        self.write_html(html, plain)

    def write_json(self, payload: dict):
        with self.json_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


def run_git(git_path: Path, target_dir: Path, args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Запустить Git внутри целевого checkout-а и полностью захватить вывод."""

    command = [str(git_path), "-C", str(target_dir), *args]
    result = subprocess.run(
        command,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and result.returncode != 0:
        command_text = "git " + " ".join(args)
        details = (result.stderr or result.stdout or "").strip()
        raise UpdaterError(f"Команда завершилась с ошибкой: {command_text}\n{details}")
    return result


def run_git_streaming(
    git_path: Path,
    target_dir: Path,
    args: list[str],
    logger: UpdateLogger,
    check: bool = True,
    progress_title: str = "Git",
) -> subprocess.CompletedProcess:
    """Запустить Git с потоковым выводом прогресса.

    `git fetch --progress` пишет прогресс в stderr и часто перерисовывает одну
    строку через carriage return. Посимвольное чтение позволяет показывать
    проценты в progress-bar PySM, не засыпая консоль сотнями строк GitHub.
    """

    command = [str(git_path), "-C", str(target_dir), *args]
    process = subprocess.Popen(
        command,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )

    output_parts = []
    buffer = []
    last_console_line = ""
    last_emit_time = 0.0
    last_progress_phase = ""
    last_progress_percent = -1
    progress = None
    if IS_RUNNING_UNDER_PYSM and JsonProgressReporter is not None:
        progress = JsonProgressReporter(total=0, desc=f"{progress_title}: получение данных")

    def progress_match(line: str) -> re.Match[str] | None:
        if line.startswith(f"{progress_title}: "):
            line = line[len(progress_title) + 2 :]
        return GIT_PROGRESS_RE.match(line)

    def update_progress_bar(match: re.Match[str], line: str) -> bool:
        nonlocal last_progress_phase, last_progress_percent, progress
        phase = match.group("phase")
        percent = max(0, min(100, int(match.group("percent"))))
        if phase == last_progress_phase and percent == last_progress_percent:
            return True

        label = GIT_PROGRESS_PHASE_LABELS.get(phase, phase)
        details = match.group("details").strip()
        detail_suffix = f" {details}" if details and phase == "Receiving objects" else ""
        desc = f"{progress_title}: {label}{detail_suffix}"

        if progress is not None:
            if phase != last_progress_phase:
                progress.reset(total=100)
                progress.set_description(desc, refresh=False)
                progress.update(percent)
            else:
                progress.set_description(desc, refresh=False)
                progress.update(percent - last_progress_percent)
        elif phase != last_progress_phase or percent == 100:
            logger.icon_line(f"{progress_title}: {phase} {percent}%", "REFRESH")

        last_progress_phase = phase
        last_progress_percent = percent
        return True

    def emit_buffer(force: bool = False):
        nonlocal buffer, last_console_line, last_emit_time
        line = "".join(buffer).strip()
        buffer = []
        if not line:
            return
        output_parts.append(line)
        logger.write_file(f"{progress_title}: {line}")

        match = progress_match(line)
        if match and update_progress_bar(match, line):
            return

        now = time.monotonic()
        is_final_progress = "done" in line.lower() or "100%" in line
        if force or is_final_progress or (line != last_console_line and now - last_emit_time >= 0.75):
            logger.icon_line(f"{progress_title}: {line}", "REFRESH")
            last_console_line = line
            last_emit_time = now

    try:
        assert process.stdout is not None
        while True:
            char = process.stdout.read(1)
            if char == "" and process.poll() is not None:
                break
            if char == "":
                continue
            if char in ("\r", "\n"):
                emit_buffer(force=char == "\n")
            else:
                buffer.append(char)

        emit_buffer(force=True)
        return_code = process.wait()
    finally:
        if progress is not None:
            progress.close()
    stdout = "\n".join(output_parts)
    result = subprocess.CompletedProcess(command, return_code, stdout=stdout, stderr="")
    if check and return_code != 0:
        command_text = "git " + " ".join(args)
        details = stdout.strip()
        raise UpdaterError(f"Команда завершилась с ошибкой: {command_text}\n{details}")
    return result


def git_output(git_path: Path, target_dir: Path, args: list[str], check: bool = True) -> str:
    """Вернуть обрезанный stdout Git для команд, где пробелы по краям не важны."""

    return run_git(git_path, target_dir, args, check=check).stdout.strip()


def git_output_raw(git_path: Path, target_dir: Path, args: list[str], check: bool = True) -> str:
    """Вернуть stdout Git без изменений.

    Это важно для porcelain-форматов с `-z`: strip может удалить начальный пробел
    из статуса вроде ` M file.py` и сломать разбор.
    """

    return run_git(git_path, target_dir, args, check=check).stdout


def build_argument_parser() -> argparse.ArgumentParser:
    """Build one explicit update-mode contract for every launch environment."""
    parser = argparse.ArgumentParser(description="Обновление PySM через portable Git.")
    parser.add_argument("--target_dir", type=str, help="Папка установки PySM")
    parser.add_argument("--git_path", type=str, help="Путь к git.exe; обычно определяется автоматически")
    parser.add_argument("--remote", type=str, default=DEFAULT_REMOTE, help="Имя Git remote")
    parser.add_argument(
        "--remote_url",
        type=str,
        default=DEFAULT_REMOTE_URL,
        help="URL репозитория GitHub, если remote не настроен",
    )
    parser.add_argument("--branch", type=str, default=DEFAULT_BRANCH, help="Ветка для обновления")
    parser.add_argument(
        "--expected_remote_contains",
        type=str,
        default=DEFAULT_EXPECTED_REMOTE,
        help="Ожидаемый репозиторий remote в формате host/owner/repo; пустое значение отключает проверку",
    )
    parser.add_argument(
        "--allow_untrusted_remote",
        action="store_true",
        help="Разрешить обновление из remote, который не прошел проверку expected_remote_contains",
    )
    parser.add_argument(
        "--update_mode",
        choices=("plan", "apply"),
        default="plan",
        help="Режим работы: plan показывает план, apply применяет обновление",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Принудительно заменить tracked-файлы версией из remote после обязательного системного архива",
    )
    parser.add_argument(
        "--repair_git_state",
        action="store_true",
        help="Синхронизировать HEAD с remote, если нет конфликтов с файлами обновления",
    )
    parser.add_argument(
        "--create_backup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Создавать обычный ZIP-бэкап перед обновлением; системный архив force создается всегда",
    )
    parser.add_argument("--show_stat", action="store_true", help="Сохранить git diff --stat в полном логе")
    parser.add_argument("--max_commits", type=int, default=50, help="Максимум коммитов, получаемых для отчета")
    parser.add_argument("--max_files", type=int, default=300, help="Максимум файлов и записей в полном отчете")
    parser.add_argument(
        "--console_preview_limit",
        type=int,
        default=DEFAULT_CONSOLE_PREVIEW_LIMIT,
        help="Максимум строк предпросмотра для длинных списков в консоли PySM",
    )
    return parser


def get_config():
    """Разобрать параметры из PySM config/context или из командной строки."""

    parser = build_argument_parser()

    if IS_MANAGED_RUN and ConfigResolver:
        return ConfigResolver(parser).resolve_all()
    return parser.parse_args()


def resolve_target_dir(config) -> Path:
    """Вернуть checkout PySM, который нужно обновить.

    В managed-режиме `APPLICATION_ROOT_DIR` указывает на корень установленного
    PySM. Ручной `--target_dir` оставлен для диагностики и прямого запуска из
    терминала.
    """

    if config.target_dir:
        return Path(config.target_dir).resolve()
    return Path(APPLICATION_ROOT_DIR).resolve()


def find_portable_git(target_dir: Path, explicit_git_path: str | None = None) -> Path:
    """Найти Git в portable-раскладке PySM, затем попробовать системный Git.

    Ожидаемая раскладка установки:
        <base>/repos/PySM
        <base>/ps_env/git

    Поэтому `target_dir.parent.parent` является общей папкой `<base>`. Updater не
    зависит от глобально установленного Git, но принимает его как запасной вариант
    на машинах разработки.
    """

    if explicit_git_path:
        git_path = Path(explicit_git_path).resolve()
        if git_path.is_file():
            return git_path
        raise UpdaterError(f"Указанный git.exe не найден: {git_path}")

    base_dir = target_dir.parent.parent
    candidates = [
        base_dir / "ps_env" / "git" / "cmd" / "git.exe",
        base_dir / "ps_env" / "git" / "bin" / "git.exe",
        base_dir / "ps_env" / "git" / "mingw64" / "bin" / "git.exe",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    system_git = shutil.which("git")
    if system_git:
        return Path(system_git).resolve()

    searched = "\n".join(str(path) for path in candidates)
    raise UpdaterError(f"Portable Git не найден. Проверенные пути:\n{searched}")


def assert_git_checkout(git_path: Path, target_dir: Path):
    """Остановиться сразу, если целевая папка не является Git working tree."""

    inside = git_output(git_path, target_dir, ["rev-parse", "--is-inside-work-tree"])
    if inside.lower() != "true":
        raise UpdaterError(f"Целевая папка не является Git checkout: {target_dir}")


def ensure_local_git_excludes(
    git_path: Path,
    target_dir: Path,
    logger: UpdateLogger,
    apply_changes: bool = True,
) -> Path:
    """Добавить локальные ignore-правила для тяжелых runtime-папок.

    Updater пишет в `.git/info/exclude`, а не в `.gitignore`, потому что эти
    правила относятся к конкретной установке. Они скрывают кеши, логи, бэкапы и
    модели в рабочем дереве пользователя, не заставляя каждый clone репозитория
    хранить такую же локальную runtime-политику.
    """

    exclude_path_text = git_output(git_path, target_dir, ["rev-parse", "--git-path", "info/exclude"])
    exclude_path = Path(exclude_path_text)
    if not exclude_path.is_absolute():
        exclude_path = target_dir / exclude_path

    existing_text = ""
    if exclude_path.exists():
        existing_text = exclude_path.read_text(encoding="utf-8", errors="replace")

    existing_lines = {line.strip() for line in existing_text.splitlines()}
    missing_lines = [line for line in LOCAL_GIT_EXCLUDE_LINES if line not in existing_lines]
    if not missing_lines:
        return exclude_path

    if not apply_changes:
        logger.icon_line(
            f"Локальные Git-исключения требуют обновления, но режим plan не меняет .git/info/exclude: {exclude_path}",
            "INFO",
        )
        return exclude_path

    exclude_path.parent.mkdir(parents=True, exist_ok=True)
    with exclude_path.open("a", encoding="utf-8", newline="\n") as f:
        if existing_text and not existing_text.endswith(("\n", "\r")):
            f.write("\n")
        f.write("\n# PySM local runtime files\n")
        for line in missing_lines:
            f.write(line + "\n")

    logger.write(f"Локальные Git-исключения обновлены: {exclude_path}")
    return exclude_path


def remote_identity(url: str) -> tuple[str, str]:
    """Вернуть `(host, owner/repo)` для HTTPS/SSH/scp-like Git URL."""

    raw = url.strip()
    if "://" in raw:
        parsed = urlparse(raw)
        host = (parsed.hostname or "").lower()
        path = parsed.path.lstrip("/")
    elif "@" in raw and ":" in raw:
        _, rest = raw.rsplit("@", 1)
        host, path = rest.split(":", 1)
        host = host.lower()
    else:
        parsed = urlparse("https://" + raw)
        host = (parsed.hostname or "").lower()
        path = parsed.path.lstrip("/")

    return host, path.replace("\\", "/").removesuffix(".git").lower().strip("/")


def assert_trusted_remote(remote_url: str, expected: str | None, allow_untrusted: bool):
    """Защитить пользователя от обновления PySM из неожиданного репозитория."""

    if not expected or allow_untrusted:
        return

    remote_host, remote_path = remote_identity(remote_url)
    expected_host, expected_path = remote_identity(expected)
    if remote_host == expected_host and remote_path == expected_path:
        return

    raise UpdaterError(
        "Remote не прошел проверку безопасности.\n"
        f"Remote URL: {remote_url}\n"
        f"Ожидался репозиторий: {expected_host}/{expected_path}\n"
        "Если это осознанно, включите allow_untrusted_remote."
    )


def validate_git_ref_name(git_path: Path, target_dir: Path, kind: str, value: str):
    """Проверить пользовательское имя Git ref до сборки refspec."""

    if not value or value.startswith("-") or "\\" in value:
        raise UpdaterError(f"Некорректное имя {kind}: {value}")

    if kind == "remote":
        if "/" in value:
            raise UpdaterError(f"Некорректное имя {kind}: {value}")
        args = ["check-ref-format", "--allow-onelevel", value]
    else:
        args = ["check-ref-format", f"refs/heads/{value}"]

    result = run_git(git_path, target_dir, args, check=False)
    if result.returncode != 0:
        raise UpdaterError(f"Некорректное имя {kind}: {value}")


def resolve_fetch_source(git_path: Path, target_dir: Path, remote: str, remote_url: str | None) -> tuple[str, str]:
    """Вернуть источник для `git fetch` и URL, который показывается в логах.

    Если `origin` существует, Git использует его штатно. Если remote отсутствует,
    прямой `remote_url` передается только в текущую команду fetch. Так portable-
    установка получает безопасный default без скрытой записи remote в
    `.git/config`.
    """

    result = run_git(git_path, target_dir, ["remote", "get-url", remote], check=False)
    if result.returncode == 0 and result.stdout.strip():
        return remote, result.stdout.strip()
    if remote_url:
        return remote_url, remote_url
    details = (result.stderr or result.stdout or "").strip()
    raise UpdaterError(
        f"Remote '{remote}' не найден, а remote_url не указан.\n"
        f"{details}"
    )


def create_backup(target_dir: Path, logger: UpdateLogger, include_paths: list[str] | None = None) -> Path:
    """Создать ZIP-снимок перед изменением tracked-файлов.

    Бэкап намеренно пропускает Git-метаданные и большие runtime-директории: эти
    данные либо восстанавливаемы, либо слишком велики для практичного updater-
    бэкапа, либо находятся вне области обновления репозитория.
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = target_dir / "_backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_file = backup_dir / f"pysm_backup_{timestamp}.zip"

    files_to_backup = []
    seen_rel_paths = set()

    def add_backup_file(abs_path: Path, rel_path: Path):
        rel_key = rel_path.as_posix()
        if rel_key in seen_rel_paths or not abs_path.is_file():
            return
        seen_rel_paths.add(rel_key)
        files_to_backup.append((abs_path, rel_path))

    for root, dirs, files in os.walk(target_dir):
        dirs[:] = [name for name in dirs if name not in BACKUP_EXCLUDED_DIRS]
        root_path = Path(root)
        for file_name in files:
            abs_path = root_path / file_name
            try:
                rel_path = abs_path.relative_to(target_dir)
            except ValueError:
                continue
            add_backup_file(abs_path, rel_path)

    extra_count = 0
    for rel_text in include_paths or []:
        rel_path = Path(rel_text)
        if rel_path.is_absolute():
            continue
        abs_path = target_dir / rel_path
        before_count = len(files_to_backup)
        add_backup_file(abs_path, rel_path)
        if len(files_to_backup) > before_count:
            extra_count += 1

    logger.kv_line("Файл бэкапа", str(backup_file), "FILE_ARCHIVE")
    logger.kv_line("Файлов к упаковке", str(len(files_to_backup)), "LIST")
    if extra_count:
        logger.kv_line("Tracked-файлов из исключенных папок", str(extra_count), "INFO")
    with zipfile.ZipFile(backup_file, "w", zipfile.ZIP_DEFLATED) as zf:
        for index, (abs_path, rel_path) in enumerate(files_to_backup, start=1):
            zf.write(abs_path, rel_path)
            if index == len(files_to_backup) or index % BACKUP_PROGRESS_STEP == 0:
                logger.write(f"  Упаковано: {index}/{len(files_to_backup)}")

    logger.icon_line("Резервная копия создана.", "OK")
    return backup_file


def get_tracked_file_paths(git_path: Path, target_dir: Path) -> list[str]:
    """Вернуть пути файлов, известных Git в текущем индексе."""

    return split_nul(git_output_raw(git_path, target_dir, ["ls-files", "-z"]))


def create_force_system_archive(git_path: Path, target_dir: Path, logger: UpdateLogger) -> Path:
    """Заархивировать существующие tracked-файлы до разрушительного `force`.

    В отличие от обычного бэкапа, сюда не входят untracked-файлы, логи и
    пользовательские результаты. Архив нужен, чтобы вручную вернуть именно
    состояние файлов системы до `git reset --hard`.
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = target_dir / "_backups"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / f"pysm_force_system_{timestamp}.zip"

    tracked_paths = get_tracked_file_paths(git_path, target_dir)
    files_to_archive = []
    missing_paths = []
    for relative_path in tracked_paths:
        absolute_path = target_dir / relative_path
        if absolute_path.is_file():
            files_to_archive.append((absolute_path, Path(relative_path)))
        else:
            missing_paths.append(relative_path)

    logger.section("Архив системы перед force", "FILE_ARCHIVE")
    logger.kv_line("Файл архива", str(archive_path), "FILE_ARCHIVE")
    logger.kv_line("Tracked-файлов к упаковке", str(len(files_to_archive)), "LIST")
    if missing_paths:
        logger.kv_line("Отсутствующих tracked-файлов", str(len(missing_paths)), "WARNING")

    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for index, (absolute_path, relative_path) in enumerate(files_to_archive, start=1):
            archive.write(absolute_path, relative_path)
            if index == len(files_to_archive) or index % BACKUP_PROGRESS_STEP == 0:
                logger.write(f"  Упаковано системных файлов: {index}/{len(files_to_archive)}")

    logger.icon_line("Архив tracked-файлов перед force создан.", "OK")
    return archive_path


def split_lines(value: str) -> list[str]:
    return [line for line in value.splitlines() if line.strip()]


def split_nul(value: str) -> list[str]:
    return [item for item in value.split("\0") if item]


def icon_html(name: str, size: int = 16, color: str | None = None) -> str:
    if not icons:
        return ""
    try:
        return getattr(icons, name)(size=size, color=color)
    except Exception:
        return ""


def html_badge(text: str, color: str, background: str) -> str:
    return (
        f'<span style="display:inline-block; min-width:125px; padding:1px 5px; '
        f'border-radius:4px; color:{color}; background:{background}; '
        f'font-weight:600;">{html_escape(text)}</span>'
    )


def remote_status_label(status: str) -> str:
    code = status[0] if status else ""
    labels = {
        "A": "Добавлен в обновлении",
        "M": "Изменен в обновлении",
        "D": "Удален в обновлении",
        "R": "Переименован в обновлении",
        "C": "Скопирован в обновлении",
    }
    return labels.get(code, f"Git status {status}")


def local_status_label(xy: str) -> str:
    if xy == "??":
        return "Новый файл вне Git"

    index_code = xy[0] if len(xy) > 0 else " "
    worktree_code = xy[1] if len(xy) > 1 else " "
    if "U" in xy:
        return "Конфликт"
    if "D" in xy:
        return "Удален локально"
    if "A" in xy:
        return "Добавлен локально"
    if "R" in xy:
        return "Переименован локально"
    if "M" in xy:
        return "Изменен локально"

    parts = []
    code_labels = {
        "M": "изменен",
        "A": "добавлен",
        "D": "удален",
        "R": "переименован",
        "C": "скопирован",
        "U": "конфликт",
    }
    if index_code != " ":
        parts.append(f"индекс: {code_labels.get(index_code, index_code)}")
    if worktree_code != " ":
        parts.append(f"рабочая папка: {code_labels.get(worktree_code, worktree_code)}")
    return "; ".join(parts) if parts else "Без изменений"


def format_remote_display(status: str, paths: list[str]) -> str:
    label = remote_status_label(status)
    if len(paths) == 2:
        return f"{label}: {paths[0]} -> {paths[1]}"
    return f"{label}: {paths[-1]}"


def format_local_display(xy: str, paths: list[str]) -> str:
    label = local_status_label(xy)
    if len(paths) == 2:
        return f"{label}: {paths[0]} -> {paths[1]}"
    return f"{label}: {paths[-1]}"


def format_local_item_html(display: str, path: str, status_html: str, icon_name: str = "FILE") -> str:
    return (
        '<tr>'
        '<td width="18" style="padding:1px 4px 1px 18px; vertical-align:top;">'
        f'{icon_html(icon_name, 14)}'
        '</td>'
        '<td width="150" style="padding:1px 6px 1px 0; vertical-align:top;">'
        f'{status_html}'
        '</td>'
        '<td style="padding:1px 0; vertical-align:top;">'
        f'<code>{html_escape(path)}</code>'
        '</td>'
        '</tr>'
    )


def display_to_status_and_path(display: str) -> tuple[str, str]:
    if ": " not in display:
        return "Файл", display
    status, path = display.split(": ", 1)
    return status, path


def plain_status_badge(status: str) -> str:
    if "Новый" in status:
        return html_badge(status, "#e65100", "#fff3e0")
    if "конфликт" in status.lower():
        return html_badge(status, "#b71c1c", "#ffebee")
    if "Удален" in status or "удален" in status:
        return html_badge(status, "#b71c1c", "#ffebee")
    if "Добавлен" in status or "добавлен" in status:
        return html_badge(status, "#1b5e20", "#e8f5e9")
    if "Переименован" in status or "переименован" in status:
        return html_badge(status, "#4a148c", "#f3e5f5")
    if "Изменен" in status or "изменен" in status:
        return html_badge(status, "#0d47a1", "#e3f2fd")
    return html_badge(status, "#424242", "#eeeeee")


def parse_name_status_z(value: str) -> list[dict]:
    """Разобрать вывод `git diff --name-status -z`.

    Записи rename/copy содержат два пути, поэтому требуют отдельной обработки.
    """

    tokens = split_nul(value)
    entries = []
    index = 0
    while index < len(tokens):
        status = tokens[index]
        index += 1
        if status.startswith(("R", "C")):
            if index + 1 >= len(tokens):
                break
            old_path = tokens[index]
            new_path = tokens[index + 1]
            index += 2
            entries.append({
                "status": status,
                "paths": [old_path, new_path],
                "display": format_remote_display(status, [old_path, new_path]),
            })
        else:
            if index >= len(tokens):
                break
            path = tokens[index]
            index += 1
            entries.append({"status": status, "paths": [path], "display": format_remote_display(status, [path])})
    return entries


def parse_status_z(value: str) -> list[dict]:
    """Разобрать `git status --porcelain=v1 -z` без потери XY-колонок."""

    tokens = split_nul(value)
    entries = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        index += 1
        if len(token) < 4:
            continue
        xy = token[:2]
        first_path = token[3:]
        paths = [first_path]
        if "R" in xy or "C" in xy:
            if index < len(tokens):
                paths.append(tokens[index])
                index += 1
        entries.append({
            "xy": xy,
            "paths": paths,
            "path": paths[-1],
            "display": format_local_display(xy, paths),
        })
    return entries


def get_remote_changes(git_path: Path, target_dir: Path, remote_ref: str) -> dict:
    """Собрать файловую поверхность обновления между local HEAD и remote."""

    raw = git_output_raw(git_path, target_dir, ["diff", "--name-status", "-z", f"HEAD..{remote_ref}"], check=False)
    entries = parse_name_status_z(raw)
    changed_paths = set()
    added_paths = set()
    deleted_paths = set()
    for entry in entries:
        status = entry["status"]
        paths = entry["paths"]
        for path in paths:
            changed_paths.add(path)
        if status.startswith("A"):
            added_paths.add(paths[-1])
        elif status.startswith("D"):
            deleted_paths.add(paths[-1])
        elif status.startswith("R"):
            deleted_paths.add(paths[0])
            added_paths.add(paths[-1])
    return {
        "entries": entries,
        "changed_paths": changed_paths,
        "added_paths": added_paths,
        "deleted_paths": deleted_paths,
    }


def get_update_plan(git_path: Path, target_dir: Path, remote_ref: str, max_commits: int, max_files: int) -> dict:
    """Построить человекочитаемый и JSON-план обновления после fetch."""

    current_commit = git_output(git_path, target_dir, ["rev-parse", "--short", "HEAD"])
    current_commit_full = git_output(git_path, target_dir, ["rev-parse", "HEAD"])
    remote_commit = git_output(git_path, target_dir, ["rev-parse", "--short", remote_ref])
    remote_commit_full = git_output(git_path, target_dir, ["rev-parse", remote_ref])
    ahead_behind = git_output(git_path, target_dir, ["rev-list", "--left-right", "--count", f"HEAD...{remote_ref}"])
    ahead_text, behind_text = ahead_behind.split()
    ahead = int(ahead_text)
    behind = int(behind_text)

    commits = split_lines(
        git_output(
            git_path,
            target_dir,
            ["log", "--oneline", f"--max-count={max_commits}", f"HEAD..{remote_ref}"],
            check=False,
        )
    )
    remote_changes = get_remote_changes(git_path, target_dir, remote_ref)
    files = [entry["display"] for entry in remote_changes["entries"]]
    file_report_limit = max(0, int(max_files))
    stat = git_output(git_path, target_dir, ["diff", "--stat", f"HEAD..{remote_ref}"], check=False)

    return {
        "current_commit": current_commit,
        "current_commit_full": current_commit_full,
        "remote_commit": remote_commit,
        "remote_commit_full": remote_commit_full,
        "ahead": ahead,
        "behind": behind,
        "commits": commits,
        "changed_files": files[:file_report_limit],
        "changed_files_total": len(files),
        "changed_paths": sorted(remote_changes["changed_paths"]),
        "added_paths": sorted(remote_changes["added_paths"]),
        "deleted_paths": sorted(remote_changes["deleted_paths"]),
        "stat": stat,
    }


def write_console_preview_list(
    logger: UpdateLogger,
    title: str,
    items: list[str],
    icon_name: str = "FILE",
    preview_limit: int = DEFAULT_CONSOLE_PREVIEW_LIMIT,
    total_count: int | None = None,
):
    """Показать короткий список в консоли, а полный список сохранить в plain-log."""

    total = len(items) if total_count is None else total_count
    display_limit = max(0, min(len(items), preview_limit))
    plain_lines = [f"{title}: {total}"]
    html_parts = [
        '<div style="margin:6px 0 4px 0;">',
        f'{icon_html(icon_name, 16)} <b>{html_escape(title)}</b>: ',
        f'<span style="font-weight:600;">{total}</span>',
        '</div>',
        '<table cellspacing="0" cellpadding="0" border="0" style="margin:0 0 2px 0; border-collapse:collapse;">',
    ]
    if items:
        for item in items[:display_limit]:
            plain_lines.append(f"  - {item}")
            html_parts.append(
                '<tr>'
                f'<td style="padding:1px 6px 1px 0;">{icon_html(icon_name, 14)}</td>'
                f'<td style="padding:1px 0;"><code>{html_escape(item)}</code></td>'
                '</tr>'
            )
        hidden_count = total - display_limit
        if hidden_count > 0:
            if total > len(items):
                hidden_message = (
                    f"... скрыто строк: {hidden_count}. "
                    f"В логе сохранено записей: {len(items)} из {total}; для большего списка увеличьте max_files."
                )
            else:
                hidden_message = f"... скрыто строк: {hidden_count}. Полный список в логе обновления."
            plain_lines.append(f"  {hidden_message}")
            html_parts.append(
                '<tr><td></td><td style="padding:2px 0; color:#666;">'
                f'{html_escape(hidden_message)}</td></tr>'
            )
    else:
        plain_lines.append("  Нет записей.")
        html_parts.append('<tr><td></td><td style="padding:1px 0; color:#666;">Нет записей.</td></tr>')
    html_parts.append("</table>")
    logger.write_html("".join(html_parts), "\n".join(plain_lines))

    if len(items) > display_limit:
        logger.write_file("")
        logger.write_file(f"Полный список: {title} ({len(items)} из {total})")
        for item in items:
            logger.write_file(f"  - {item}")
    if total > len(items):
        logger.write_file(f"  ... не сохранено из-за max_files: {total - len(items)}")


def write_update_plan(logger: UpdateLogger, plan: dict, show_stat: bool, console_preview_limit: int):
    logger.section("План обновления", "LIST")
    logger.kv_line("Текущий commit", plan["current_commit"], "INFO")
    logger.kv_line("Remote commit", plan["remote_commit"], "INFO")
    logger.kv_line("Локально впереди", str(plan["ahead"]), "ARROW_SUB")
    logger.kv_line("Remote впереди", str(plan["behind"]), "REFRESH")

    if plan["behind"] == 0 and plan["ahead"] == 0:
        logger.icon_line("Новых commit нет.", "OK")
        return
    if plan["ahead"] > 0 and plan["behind"] > 0:
        logger.icon_line("Истории разошлись. Автоматическое обновление будет заблокировано.", "WARNING")
    elif plan["ahead"] > 0:
        logger.icon_line("Локальная ветка уже содержит коммиты, которых нет в remote.", "WARNING")

    write_console_preview_list(logger, "Новые коммиты", plan["commits"], "COMMIT", console_preview_limit)
    write_console_preview_list(
        logger,
        "Измененные файлы",
        plan["changed_files"],
        "FILE",
        console_preview_limit,
        total_count=plan["changed_files_total"],
    )

    if show_stat and plan["stat"]:
        logger.write_file()
        logger.write_file("Статистика diff:")
        for line in plan["stat"].splitlines():
            logger.write_file(f"  {line}")
        logger.icon_line("Статистика diff сохранена в полном логе.", "REPORT")


def get_status_entries(git_path: Path, target_dir: Path) -> list[dict]:
    """Вернуть изменения working tree/index, включая обычные untracked-файлы."""

    status = git_output_raw(git_path, target_dir, ["status", "--porcelain=v1", "-z", "--untracked-files=normal"], check=False)
    return parse_status_z(status)


def remote_blob(git_path: Path, target_dir: Path, remote_ref: str, path: str) -> str | None:
    """Вернуть blob id пути в remote или None, если remote этот путь не отслеживает."""

    result = run_git(git_path, target_dir, ["rev-parse", f"{remote_ref}:{path}"], check=False)
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def worktree_blob(git_path: Path, target_dir: Path, path: str) -> str | None:
    """Вернуть blob id текущего содержимого файла в working tree."""

    abs_path = target_dir / path
    if not abs_path.is_file():
        return None
    result = run_git(git_path, target_dir, ["hash-object", "--", path], check=False)
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def path_matches_remote(git_path: Path, target_dir: Path, remote_ref: str, path: str) -> bool:
    """Проверить совпадение содержимого с remote независимо от состояния index."""

    return worktree_blob(git_path, target_dir, path) == remote_blob(git_path, target_dir, remote_ref, path)


def path_collides_with_added(untracked_path: str, added_paths: set[str]) -> bool:
    """Найти untracked-файлы/папки, которые помешают добавить файлы из remote."""

    normalized = untracked_path.rstrip("/")
    for added_path in added_paths:
        if added_path == normalized or added_path.startswith(normalized + "/"):
            return True
    return False


def is_user_state_path(path: str) -> bool:
    """Определить встроенные пути, которые PySM меняет при обычной работе.

    Это намеренно короткий список. Он защищает настройки приложения и коллекции
    пользователя, но не маскирует локальные правки Python-кода, тем и ресурсов,
    для которых обновление должно остановиться и показать конфликт.
    """

    normalized_path = path.replace("\\", "/").lstrip("/")
    return normalized_path in USER_STATE_EXACT_PATHS or normalized_path.startswith(USER_STATE_DIRECTORY_PREFIXES)


def entry_has_user_state_path(entry: dict) -> bool:
    """Вернуть true, если все затронутые записью Git пути являются user-state."""

    paths = entry.get("paths", [])
    return bool(paths) and all(is_user_state_path(path) for path in paths)


def entry_intersects_paths(entry: dict, paths: set[str]) -> bool:
    """Проверить все пути записи, включая обе стороны rename/copy."""

    return any(path in paths for path in entry.get("paths", []))


def analyze_local_state(git_path: Path, target_dir: Path, remote_ref: str, plan: dict) -> dict:
    """Классифицировать локальные изменения по риску для updater-а.

    Git отвечает на два разных вопроса:
    - `HEAD..remote` говорит, есть ли на GitHub новые commit;
    - `git status` говорит, отличается ли конкретное рабочее дерево от HEAD.

    Пользователь может удалить tracked-файл темы, когда новых commit на GitHub
    нет. Это все равно локальное расхождение, которое нужно показать/исправить,
    даже если план обновления говорит "загружать нечего".
    """

    status_entries = get_status_entries(git_path, target_dir)
    remote_paths = set(plan["changed_paths"])
    added_paths = set(plan["added_paths"])
    already_remote = []
    already_remote_paths = []
    already_remote_index_dirty = 0
    already_remote_worktree_dirty = 0
    conflicts = []
    missing_tracked = []
    missing_tracked_paths = []
    user_state = []
    user_state_conflicts = []
    user_state_conflict_entries = []
    local_outside_update = []
    untracked_collisions = []
    untracked_other = []

    for entry in status_entries:
        xy = entry["xy"]
        path = entry["path"]
        is_user_state = entry_has_user_state_path(entry)
        if xy == "??":
            # Пользовательские файлы вне Git сохраняются по умолчанию. Блокируем
            # только случай, когда GitHub собирается добавить тот же путь и Git
            # не сможет выполнить checkout.
            if path_collides_with_added(path, added_paths):
                untracked_collisions.append(entry["display"])
            else:
                untracked_other.append(entry["display"])
            continue

        # PySM пишет настройки и коллекции непосредственно в working tree. Такие
        # правки не должны выглядеть как конфликт кода; staged-состояние намеренно
        # остается блокирующим, потому что его создает пользователь вручную.
        if is_user_state and xy[0] == " ":
            user_state.append(entry["display"])
            if entry_intersects_paths(entry, remote_paths):
                user_state_conflicts.append(entry["display"])
                user_state_conflict_entries.append(entry)
            continue

        if "D" in xy and remote_blob(git_path, target_dir, remote_ref, path) is not None:
            # Удаленный локально tracked-файл не является конфликтом обновления:
            # это повреждение локального checkout-а, которое можно восстановить.
            missing_tracked.append(entry["display"])
            missing_tracked_paths.append(path)
            continue

        if path_matches_remote(git_path, target_dir, remote_ref, path):
            # Старые ZIP-обновления могли оставить Git в состоянии, где многие
            # файлы считаются modified, хотя байты уже совпадают с remote. Такие
            # статусы можно исправить без перезаписи настоящих user-изменений.
            already_remote.append(entry["display"])
            already_remote_paths.append(path)
            if xy[0] != " ":
                already_remote_index_dirty += 1
            if xy[1] != " ":
                already_remote_worktree_dirty += 1
            continue

        if entry_intersects_paths(entry, remote_paths):
            conflicts.append(entry["display"])
        else:
            local_outside_update.append(entry["display"])

    return {
        "entries": [entry["display"] for entry in status_entries],
        "already_remote": already_remote,
        "already_remote_paths": already_remote_paths,
        "already_remote_index_dirty": already_remote_index_dirty,
        "already_remote_worktree_dirty": already_remote_worktree_dirty,
        "conflicts": conflicts,
        "missing_tracked": missing_tracked,
        "missing_tracked_paths": missing_tracked_paths,
        "user_state": user_state,
        "user_state_conflicts": user_state_conflicts,
        "user_state_conflict_entries": user_state_conflict_entries,
        "local_outside_update": local_outside_update,
        "untracked_collisions": untracked_collisions,
        "untracked_other": untracked_other,
        "has_blocking_conflicts": bool(conflicts or untracked_collisions),
    }


def write_limited_list(
    logger: UpdateLogger,
    title: str,
    items: list[str],
    max_items: int,
    icon_name: str = "FILE",
    console_preview_limit: int = DEFAULT_CONSOLE_PREVIEW_LIMIT,
):
    """Вывести короткий список в HTML-консоль и полный список в plain-log."""

    total = len(items)
    report_limit = max(0, int(max_items))
    report_items = items[:report_limit]
    display_limit = max(0, min(len(report_items), LOCAL_STATE_PREVIEW_LIMIT, console_preview_limit))
    plain_lines = [f"{title}: {total}"]
    html_parts = [
        '<div style="margin:6px 0 4px 0;">',
        f'{icon_html(icon_name, 16)} <b>{html_escape(title)}</b>: ',
        f'<span style="font-weight:600;">{total}</span>',
        '</div>',
        '<table cellspacing="0" cellpadding="0" border="0" style="margin:0 0 2px 0; border-collapse:collapse;">',
    ]
    for item in report_items[:display_limit]:
        plain_lines.append(f"  - {item}")
        item_status, item_path = display_to_status_and_path(item)
        html_parts.append(format_local_item_html(item, item_path, plain_status_badge(item_status), icon_name))
    hidden_count = total - min(total, display_limit)
    if hidden_count > 0:
        if total > len(report_items):
            hidden_message = (
                f"... скрыто строк: {hidden_count}. "
                f"В логе сохранено записей: {len(report_items)} из {total}; для большего списка увеличьте max_files."
            )
        else:
            hidden_message = f"... скрыто строк: {hidden_count}. Полный список в логе обновления."
        plain_lines.append(f"  {hidden_message}")
        html_parts.append(
            '<tr><td></td><td colspan="2" style="padding:2px 0 2px 0; color:#666;">'
            f'{html_escape(hidden_message)}</td></tr>'
        )
    html_parts.append("</table>")
    logger.write_html("".join(html_parts), "\n".join(plain_lines))

    if len(report_items) > display_limit:
        logger.write_file("")
        logger.write_file(f"Полный список: {title} ({len(report_items)} из {total})")
        for item in report_items:
            logger.write_file(f"  - {item}")


def write_local_state(
    logger: UpdateLogger,
    state: dict,
    max_items: int,
    title: str = "Локальное состояние",
    console_preview_limit: int = DEFAULT_CONSOLE_PREVIEW_LIMIT,
):
    """Показать классификацию локального состояния из `analyze_local_state`."""

    logger.section(title, "TARGET")
    write_limited_list(logger, "Файлы уже совпадают с remote", state["already_remote"], max_items, "REFRESH", console_preview_limit)
    if state["already_remote"]:
        logger.write(
            "  Диагностика этой группы: "
            f"index={state['already_remote_index_dirty']}, "
            f"worktree={state['already_remote_worktree_dirty']}"
        )
    write_limited_list(logger, "Отсутствующие файлы из репозитория", state["missing_tracked"], max_items, "WARNING", console_preview_limit)
    write_limited_list(logger, "Локальные настройки и коллекции", state["user_state"], max_items, "FILE", console_preview_limit)
    write_limited_list(
        logger,
        "Настройки и коллекции, которые будут сохранены при обновлении",
        state["user_state_conflicts"],
        max_items,
        "LOCK",
        console_preview_limit,
    )
    write_limited_list(logger, "Локальные изменения вне обновления", state["local_outside_update"], max_items, "FILE", console_preview_limit)
    write_limited_list(logger, "Файлы вне Git", state["untracked_other"], max_items, "ADD", console_preview_limit)
    write_limited_list(logger, "Локальные конфликты с обновлением", state["conflicts"], max_items, "WARNING", console_preview_limit)
    write_limited_list(logger, "Файлы вне Git, мешающие обновлению", state["untracked_collisions"], max_items, "LOCK", console_preview_limit)

    if state["untracked_collisions"]:
        logger.icon_line("Есть файлы вне Git, которые мешают обновлению.", "WARNING")
    elif state["conflicts"]:
        logger.icon_line("Есть tracked-конфликты. Обычное обновление заблокировано; force может их перезаписать.", "WARNING")
    elif state["user_state_conflicts"]:
        logger.icon_line("Настройки и коллекции будут сохранены автоматически; обновление не заблокировано.", "OK")
    else:
        logger.icon_line("Блокирующих локальных конфликтов не найдено.", "OK")


def chunked(items: list[str], size: int):
    for index in range(0, len(items), size):
        yield items[index : index + size]


def refresh_already_remote_worktree(git_path: Path, target_dir: Path, paths: list[str], logger: UpdateLogger):
    """Освежить файлы, которые уже равны remote, но Git еще считает их dirty."""

    if not paths:
        return

    logger.write(f"git checkout-index -f -- <already-remote files> ({len(paths)})")
    for chunk in chunked(paths, GIT_PATH_CHUNK_SIZE):
        run_git(git_path, target_dir, ["checkout-index", "-f", "--", *chunk])


def restore_missing_tracked_files(git_path: Path, target_dir: Path, paths: list[str], logger: UpdateLogger):
    """Восстановить tracked-файлы, удаленные локально.

    Это намеренно уже, чем `force`: функция возвращает только файлы, которые
    remote отслеживает, а локальному checkout-у их не хватает. Она не
    перезаписывает измененные файлы и не удаляет untracked user-data.
    """

    if not paths:
        return

    logger.section("Восстановление отсутствующих файлов", "WRENCH")
    logger.icon_line(f"Будут восстановлены tracked-файлы из репозитория: {len(paths)}", "INFO")
    for chunk in chunked(paths, GIT_PATH_CHUNK_SIZE):
        logger.write(f"git reset -- <missing files> ({len(chunk)})")
        run_git(git_path, target_dir, ["reset", "--", *chunk])
        logger.write(f"git checkout-index -f -- <missing files> ({len(chunk)})")
        run_git(git_path, target_dir, ["checkout-index", "-f", "--", *chunk])

    logger.write("git update-index --refresh")
    run_git(git_path, target_dir, ["update-index", "--refresh"], check=False)
    logger.icon_line("Отсутствующие tracked-файлы восстановлены.", "OK")


def resolve_repo_file_path(target_dir: Path, relative_path: str) -> Path:
    """Безопасно разрешить относительный путь, полученный из Git."""

    candidate = (target_dir / relative_path).resolve()
    try:
        candidate.relative_to(target_dir.resolve())
    except ValueError as error:
        raise UpdaterError(f"Git вернул путь вне папки PySM: {relative_path}") from error
    return candidate


def get_user_state_conflict_paths(entries: list[dict]) -> list[str]:
    """Собрать уникальные user-state пути, пересекающиеся с remote-обновлением."""

    paths = {
        path
        for entry in entries
        for path in entry.get("paths", [])
        if is_user_state_path(path)
    }
    return sorted(paths)


def create_user_state_archive(target_dir: Path, logger: UpdateLogger, entries: list[dict]) -> dict | None:
    """Сохранить локальные настройки и коллекции до временного очистки Git.

    Этот небольшой архив создается только для файлов, которые одновременно
    изменены пользователем и обновляются на GitHub. Он остается рядом с отчетом
    как аварийная копия, пока обычный ZIP-бэкап может быть отключен параметром
    `create_backup`.
    """

    paths = get_user_state_conflict_paths(entries)
    if not paths:
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = logger.log_path.parent / f"preserved_user_state_{timestamp}.zip"
    manifest = []

    logger.section("Сохранение настроек и коллекций", "LOCK")
    logger.kv_line("Файл сохраненного состояния", str(archive_path), "FILE_ARCHIVE")
    logger.kv_line("Файлов и удалений", str(len(paths)), "LIST")
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for index, relative_path in enumerate(paths, start=1):
            absolute_path = resolve_repo_file_path(target_dir, relative_path)
            exists = absolute_path.is_file()
            manifest.append({"path": relative_path, "exists": exists})
            if exists:
                archive.write(absolute_path, relative_path)
            if index == len(paths) or index % BACKUP_PROGRESS_STEP == 0:
                logger.write(f"  Сохранено пользовательских записей: {index}/{len(paths)}")
        archive.writestr("_pysm_updater_manifest.json", json.dumps({"paths": manifest}, ensure_ascii=False, indent=2))

    logger.icon_line("Локальные настройки и коллекции сохранены перед обновлением.", "OK")
    return {"archive_path": archive_path, "manifest": manifest}


def temporarily_reset_user_state_paths(
    git_path: Path,
    target_dir: Path,
    archive_info: dict,
    logger: UpdateLogger,
):
    """Вернуть защищенные пути к HEAD, чтобы `git merge --ff-only` мог пройти.

    Архив уже содержит актуальную пользовательскую версию. Изменения в index не
    поддерживаются этим путем намеренно: такая ситуация не возникает при обычной
    работе PySM и должна оставаться видимым ручным конфликтом.
    """

    paths = [entry["path"] for entry in archive_info["manifest"]]
    logger.icon_line("Временная подготовка сохраненных файлов к Git-обновлению.", "INFO")
    for chunk in chunked(paths, GIT_PATH_CHUNK_SIZE):
        logger.write(f"git reset -- <saved user-state files> ({len(chunk)})")
        run_git(git_path, target_dir, ["reset", "--", *chunk])
        logger.write(f"git checkout-index -f -- <saved user-state files> ({len(chunk)})")
        run_git(git_path, target_dir, ["checkout-index", "-f", "--", *chunk])


def restore_user_state_archive(target_dir: Path, archive_info: dict, logger: UpdateLogger):
    """Вернуть пользовательские файлы либо сохраненное локальное удаление."""

    archive_path = Path(archive_info["archive_path"])
    restored_count = 0
    with zipfile.ZipFile(archive_path, "r") as archive:
        for item in archive_info["manifest"]:
            relative_path = item["path"]
            absolute_path = resolve_repo_file_path(target_dir, relative_path)
            if item["exists"]:
                if absolute_path.exists() and not absolute_path.is_file():
                    raise UpdaterError(
                        "Не удалось безопасно вернуть пользовательский файл: "
                        f"путь занят папкой {relative_path}. Архив сохранен: {archive_path}"
                    )
                absolute_path.parent.mkdir(parents=True, exist_ok=True)
                temporary_path = None
                try:
                    with tempfile.NamedTemporaryFile(
                        mode="wb",
                        prefix=f".{absolute_path.name}.pysm_updater_",
                        suffix=".tmp",
                        dir=absolute_path.parent,
                        delete=False,
                    ) as destination:
                        temporary_path = Path(destination.name)
                        source = archive.open(relative_path)
                        try:
                            shutil.copyfileobj(source, destination)
                        finally:
                            source.close()
                    os.replace(temporary_path, absolute_path)
                finally:
                    if temporary_path is not None and temporary_path.exists():
                        temporary_path.unlink()
            elif absolute_path.is_file() or absolute_path.is_symlink():
                absolute_path.unlink()
            elif absolute_path.exists():
                raise UpdaterError(
                    "Не удалось безопасно восстановить локальное удаление: "
                    f"путь занят папкой {relative_path}. Архив сохранен: {archive_path}"
                )
            restored_count += 1

    logger.icon_line(f"Локальные настройки и коллекции возвращены: {restored_count}.", "OK")


def merge_fast_forward_preserving_user_state(
    git_path: Path,
    target_dir: Path,
    remote_ref: str,
    local_state: dict,
    logger: UpdateLogger,
) -> str | None:
    """Выполнить fast-forward и вернуть встроенное пользовательское состояние."""

    archive_info = create_user_state_archive(target_dir, logger, local_state["user_state_conflict_entries"])
    if archive_info:
        temporarily_reset_user_state_paths(git_path, target_dir, archive_info, logger)

    try:
        logger.write(f"git merge --ff-only {remote_ref}")
        merge_result = run_git(git_path, target_dir, ["merge", "--ff-only", remote_ref], check=False)
        if merge_result.returncode != 0:
            details = (merge_result.stderr or merge_result.stdout or "").strip()
            if local_state["already_remote"]:
                raise UpdaterError(
                    "Git не смог выполнить fast-forward из-за локальных изменений, хотя часть файлов уже совпадает с remote.\n"
                    "Если режим plan показывает, что локальных конфликтов с обновлением нет, включите repair_git_state.\n"
                    f"{details}"
                )
            raise UpdaterError(f"Команда завершилась с ошибкой: git merge --ff-only {remote_ref}\n{details}")
    except Exception as error:
        if archive_info:
            try:
                restore_user_state_archive(target_dir, archive_info, logger)
            except Exception as restore_error:
                raise UpdaterError(
                    f"{error}\n\n"
                    "Git-обновление не применено, а автоматическое возвращение пользовательских данных не удалось. "
                    f"Используйте сохраненный архив: {archive_info['archive_path']}\n{restore_error}"
                ) from restore_error
        raise

    if archive_info:
        restore_user_state_archive(target_dir, archive_info, logger)
        return str(archive_info["archive_path"])
    return None


def repair_git_state(git_path: Path, target_dir: Path, remote_ref: str, logger: UpdateLogger, local_state: dict | None = None):
    """Синхронизировать Git-метаданные с remote без force для всех файлов.

    `reset --mixed` переносит HEAD/index на fetched remote, но оставляет рабочие
    файлы на месте. Затем освежаются только пути, которые уже побайтово совпадают
    с remote. Это убирает ложные `modified` статусы после ZIP-обновлений и
    сохраняет config/context/user-файлы.
    """

    logger.section("Восстановление Git-состояния", "WRENCH")
    logger.write(f"git reset --mixed {remote_ref}")
    run_git(git_path, target_dir, ["reset", "--mixed", remote_ref])

    if local_state:
        refresh_already_remote_worktree(git_path, target_dir, local_state.get("already_remote_paths", []), logger)

    logger.write("git update-index --refresh")
    refresh_result = run_git(git_path, target_dir, ["update-index", "--refresh"], check=False)
    if refresh_result.returncode != 0:
        logger.icon_line("Часть файлов осталась локально измененной; это нормально для config/context/user-файлов.", "INFO")


def force_update(git_path: Path, target_dir: Path, remote_ref: str, logger: UpdateLogger):
    """Заменить tracked-файлы версией из remote через `git reset --hard`.

    Это намеренно не удаляет untracked-файлы. Удаление untracked user-data было
    бы отдельной разрушительной операцией `clean` и находится вне safety-модели
    этого updater-а.
    """

    logger.section("Принудительное обновление tracked-файлов", "WARNING")
    logger.write(f"git reset --hard {remote_ref}")
    run_git(git_path, target_dir, ["reset", "--hard", remote_ref])


def requirements_sha256(target_dir: Path) -> str | None:
    """Return the exact requirements file signature without resolving packages."""

    requirements_file = target_dir / "requirements.txt"
    if not requirements_file.is_file():
        return None
    return hashlib.sha256(requirements_file.read_bytes()).hexdigest()


def portable_setup_state_dir(target_dir: Path) -> Path | None:
    """Find the state directory owned by a neighboring portable pysm-setup.

    A development checkout must not gain installer metadata, so the marker is
    available only for the portable layout `<root>/repos/PySM` with an existing
    `<root>/ps_env/state` directory.
    """

    if target_dir.name.lower() != "pysm" or target_dir.parent.name.lower() != "repos":
        return None
    state_dir = target_dir.parent.parent / "ps_env" / "state"
    return state_dir if state_dir.is_dir() else None


def record_pending_dependency_sync(
    target_dir: Path,
    requirements_before: str | None,
    requirements_after: str | None,
    after_commit: str,
    logger: UpdateLogger,
) -> str | None:
    """Create a launcher-readable marker only when requirements changed.

    The updater never invokes pip and never changes the active PySM venv. The
    external launcher consumes this marker and asks `pysm-setup` to prepare a
    separately verified candidate environment before the next PySM launch.
    """

    if requirements_before == requirements_after:
        logger.icon_line("requirements.txt не изменился: синхронизация зависимостей не требуется.", "INFO")
        return None

    state_dir = portable_setup_state_dir(target_dir)
    if state_dir is None:
        logger.icon_line(
            "requirements.txt изменился. Это не portable-установка с pysm-setup, поэтому marker синхронизации не создан.",
            "INFO",
        )
        return None

    marker_path = state_dir / "pending-pysm-dependency-sync.json"
    temporary_path = marker_path.with_suffix(f"{marker_path.suffix}.tmp")
    payload = {
        "schema_version": 1,
        "requirements_before_sha256": requirements_before,
        "requirements_after_sha256": requirements_after,
        "after_commit": after_commit,
    }
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as marker_file:
            json.dump(payload, marker_file, ensure_ascii=False, indent=2)
            marker_file.write("\n")
        os.replace(temporary_path, marker_path)
    except OSError as error:
        logger.icon_line(
            f"requirements.txt изменился, но marker синхронизации не удалось сохранить: {error}",
            "WARNING",
        )
        return None

    logger.icon_line(
        "requirements.txt изменился. Перед следующим запуском PySM launcher предложит безопасно синхронизировать зависимости.",
        "WARNING",
    )
    logger.write(f"Создан marker синхронизации зависимостей: {marker_path}")
    return str(marker_path)


def log_report_links(logger: UpdateLogger):
    """Показать сохраненные отчеты в PySM как кликабельные ресурсы.

    Plain-пути пишутся в `.log` для полноты отчета. В managed-запуске PySM
    консоль получает дерево ResourceNode вместо дублирующих текстовых строк.
    Запуск из терминала сохраняет текстовый fallback.
    """

    logger.write_file()
    logger.write_file(f"Полный лог: {logger.log_path}")
    logger.write_file(f"JSON-отчет: {logger.json_path}")

    if not (IS_MANAGED_RUN and pysm_context and ResourceNode and StandardTreeBuilder):
        logger.write()
        logger.write(f"Полный лог: {logger.log_path}")
        logger.write(f"JSON-отчет: {logger.json_path}")
        return

    try:
        tv_builder = StandardTreeBuilder(icon_size=28)
        report_node = ResourceNode("Отчеты<br>обновления", logger.log_path.parent, "folder", "Папка отчетов updater")
        report_node.children.append(ResourceNode(logger.log_path.name, logger.log_path, "txt", "Полный текстовый лог"))
        report_node.children.append(ResourceNode(logger.json_path.name, logger.json_path, "code", "JSON-отчет"))
        tv_builder.add_section("", [report_node])
        pysm_context.log_html(tv_builder.get_html())
    except Exception as exc:
        logger.write(f"Не удалось вывести HTML-ссылки на отчеты: {type(exc).__name__}: {exc}")


def main():
    """Выполнить полный цикл обновления.

    Порядок важен:
    1. сначала fetch remote-состояния, чтобы сравнения использовали актуальный
       GitHub;
    2. затем план remote-обновления;
    3. затем анализ локального checkout-а даже при отсутствии новых commit;
    4. затем узкое восстановление локальных повреждений вроде missing tracked;
    5. затем применение fast-forward, repair или force по выбранному режиму.
    """

    config = get_config()
    target_path = resolve_target_dir(config)

    if not target_path.exists():
        print(f"Целевая папка не найдена: {target_path}", flush=True)
        sys.exit(1)
    if not target_path.is_dir():
        print(f"Целевой путь не является папкой: {target_path}", flush=True)
        sys.exit(1)

    report_dir = target_path / "_OUTPUT" / "pysm_updater"
    logger = UpdateLogger(report_dir)
    payload = {
        "result": "started",
        "target_dir": str(target_path),
        "report_log": str(logger.log_path),
        "report_json": str(logger.json_path),
    }

    try:
        console_preview_limit = max(0, int(config.console_preview_limit))
        apply_update = config.update_mode == "apply"
        payload["update_mode"] = config.update_mode

        logger.section("ЗАПУСК ОБНОВЛЕНИЯ PySM", "ROCKET")
        logger.kv_line("Целевая папка", str(target_path), "FOLDER_OPEN")
        logger.kv_line("Лог обновления", str(logger.log_path), "REPORT")
        logger.kv_line(
            "Режимы",
            (
                f"update_mode={config.update_mode}, "
                f"repair_git_state={bool(config.repair_git_state)}, "
                f"force={bool(config.force)}, "
                f"create_backup={bool(config.create_backup)}"
            ),
            "SLIDERS",
        )

        git_path = find_portable_git(target_path, config.git_path)
        payload["git_path"] = str(git_path)
        logger.kv_line("Git", str(git_path), "CONSOLE")

        assert_git_checkout(git_path, target_path)
        requirements_before = requirements_sha256(target_path)
        payload["requirements_before_sha256"] = requirements_before

        remote = config.remote or DEFAULT_REMOTE
        branch = config.branch or DEFAULT_BRANCH
        validate_git_ref_name(git_path, target_path, "remote", remote)
        validate_git_ref_name(git_path, target_path, "branch", branch)
        remote_ref = f"{remote}/{branch}"
        payload["remote"] = remote
        payload["branch"] = branch
        payload["remote_ref"] = remote_ref

        fetch_source, remote_url = resolve_fetch_source(git_path, target_path, remote, config.remote_url)
        payload["remote_url"] = remote_url
        payload["fetch_source"] = fetch_source
        if fetch_source == remote:
            logger.kv_line("Remote", f"{remote} -> {remote_url}", "REFRESH")
        else:
            logger.kv_line("Remote", f"'{remote}' не настроен; используется URL {remote_url}", "REFRESH")
        assert_trusted_remote(remote_url, config.expected_remote_contains, bool(config.allow_untrusted_remote))

        local_exclude_path = ensure_local_git_excludes(
            git_path,
            target_path,
            logger,
            apply_changes=apply_update,
        )
        payload["local_git_exclude"] = str(local_exclude_path)

        # Fetch обновляет только локальный remote-tracking ref. Рабочие файлы на
        # этом этапе не меняются, поэтому следующий plan/state отчет безопасен.
        logger.section("Получение данных с GitHub", "REFRESH")
        fetch_refspec = f"refs/heads/{branch}:refs/remotes/{remote}/{branch}"
        logger.write(f"git fetch --progress --prune <source> {fetch_refspec}")
        run_git_streaming(
            git_path,
            target_path,
            ["fetch", "--progress", "--prune", fetch_source, fetch_refspec],
            logger,
            progress_title="GitHub",
        )

        plan = get_update_plan(git_path, target_path, remote_ref, int(config.max_commits), int(config.max_files))
        payload["plan"] = plan
        write_update_plan(logger, plan, bool(config.show_stat), console_preview_limit)

        local_state = analyze_local_state(git_path, target_path, remote_ref, plan)
        payload["local_state"] = local_state
        write_local_state(logger, local_state, int(config.max_files), console_preview_limit=console_preview_limit)

        if config.force and not apply_update:
            logger.write()
            logger.icon_line(
                "Force включен, но режим plan не меняет файлы. "
                "Для применения выберите update_mode=apply.",
                "INFO",
            )

        if not apply_update:
            payload["result"] = "plan"
            logger.section("Итог", "OK")
            logger.icon_line("Режим plan завершен. Файлы не изменялись.", "OK")
            return

        if plan["ahead"] > 0:
            raise UpdaterError(
                "Локальная ветка содержит commit, которых нет в remote. "
                "Updater не должен автоматически сбрасывать локальную историю. "
                "Сначала сохраните, отправьте или вручную удалите локальные commit."
            )

        restored_missing_files = False
        if local_state["missing_tracked"] and not config.force:
            # Missing tracked можно восстановить до решения об обновлении: это
            # файлы репозитория, удаленные локально, а не untracked runtime-data.
            restore_missing_tracked_files(git_path, target_path, local_state["missing_tracked_paths"], logger)
            local_state = analyze_local_state(git_path, target_path, remote_ref, plan)
            payload["local_state_after_missing_restore"] = local_state
            write_local_state(
                logger,
                local_state,
                int(config.max_files),
                "Локальное состояние после восстановления файлов",
                console_preview_limit,
            )
            restored_missing_files = True

        if config.repair_git_state and not config.force and plan["behind"] > 0:
            update_paths = set(plan["changed_paths"])
            already_remote_paths = set(local_state["already_remote_paths"])
            if not update_paths.issubset(already_remote_paths):
                raise UpdaterError(
                    "repair_git_state нельзя использовать как обычное обновление: "
                    "часть файлов из remote еще не находится в рабочей папке. "
                    "Сначала выполните обычное обновление без repair_git_state."
                )

        if plan["behind"] == 0 and not config.force:
            # "Новых commit нет" не означает "checkout чистый". К этому моменту
            # missing tracked уже могли быть восстановлены, а optional repair все
            # еще может убрать ложные modified-статусы.
            if config.repair_git_state and apply_update:
                repair_git_state(git_path, target_path, remote_ref, logger, local_state)
                after_repair_state = analyze_local_state(git_path, target_path, remote_ref, plan)
                payload["local_state_after_repair"] = after_repair_state
                write_local_state(
                    logger,
                    after_repair_state,
                    int(config.max_files),
                    "Локальное состояние после repair",
                    console_preview_limit,
                )
                payload["result"] = "repaired"
                logger.section("Итог", "OK")
                logger.icon_line("Новых commit нет. Git-состояние синхронизировано с remote.", "OK")
                return

            payload["result"] = "no_updates"
            logger.section("Итог", "OK")
            if restored_missing_files:
                payload["result"] = "restored_missing_files"
                logger.icon_line("Новых commit нет. Отсутствующие файлы из репозитория восстановлены.", "OK")
            else:
                logger.icon_line("Обновление не требуется.", "OK")
            return

        if local_state["untracked_collisions"]:
            # Force обновляет tracked-файлы, но Git все равно не перезапишет
            # untracked-файл, который занимает путь, добавляемый обновлением.
            raise UpdaterError(
                "Есть untracked-файлы, которые мешают добавить файлы из обновления. "
                "Force не удаляет untracked-файлы, поэтому такие конфликты нужно разобрать вручную."
            )

        if local_state["conflicts"] and not config.force:
            raise UpdaterError(
                "Есть локальные изменения, которые пересекаются с файлами обновления. "
                "Автоматическое обновление отменено, чтобы не потерять данные."
            )

        if config.create_backup:
            logger.section("Бэкап", "FILE_ARCHIVE")
            backup_file = create_backup(target_path, logger, plan.get("changed_paths", []))
            payload["backup_file"] = str(backup_file)
        else:
            logger.section("Бэкап", "FILE_ARCHIVE")
            logger.icon_line("Обычный бэкап отключен параметром create_backup=False.", "INFO")

        logger.section("Применение обновления", "WRENCH")
        if config.force:
            # Даже при create_backup=False force обязан оставить отдельный снимок только
            # tracked-системы: именно эти файлы сейчас будут заменены hard reset.
            force_system_archive = create_force_system_archive(git_path, target_path, logger)
            payload["force_system_archive"] = str(force_system_archive)
            force_update(git_path, target_path, remote_ref, logger)
        elif config.repair_git_state:
            if local_state["has_blocking_conflicts"]:
                raise UpdaterError(
                    "repair_git_state невозможен: есть локальные изменения, которые пересекаются "
                    "с файлами обновления, или untracked-файлы мешают добавлению файлов."
                )
            repair_git_state(git_path, target_path, remote_ref, logger, local_state)
        else:
            # Штатный путь сохраняет встроенные пользовательские данные, но
            # по-прежнему использует только fast-forward и не меняет историю.
            preserved_user_state_archive = merge_fast_forward_preserving_user_state(
                git_path,
                target_path,
                remote_ref,
                local_state,
                logger,
            )
            if preserved_user_state_archive:
                payload["preserved_user_state_archive"] = preserved_user_state_archive

        after_commit = git_output(git_path, target_path, ["rev-parse", "--short", "HEAD"])
        payload["after_commit"] = after_commit
        payload["result"] = "updated"
        requirements_after = requirements_sha256(target_path)
        payload["requirements_after_sha256"] = requirements_after
        payload["pending_dependency_sync_marker"] = record_pending_dependency_sync(
            target_path,
            requirements_before,
            requirements_after,
            after_commit,
            logger,
        )

        logger.section("Итог", "OK")
        logger.icon_line(f"Обновление завершено успешно. Текущий commit: {after_commit}", "OK")
        logger.icon_line("Перезапустите PySM, чтобы применить изменения.", "INFO")

    except UpdaterError as exc:
        payload["result"] = "error"
        payload["error"] = str(exc)
        logger.section("Ошибка", "ERROR")
        logger.write(str(exc))
        sys.exit(1)
    except Exception as exc:
        payload["result"] = "error"
        payload["error"] = f"{type(exc).__name__}: {exc}"
        logger.section("Ошибка", "ERROR")
        logger.write(f"{type(exc).__name__}: {exc}")
        sys.exit(1)
    finally:
        try:
            logger.write_json(payload)
            log_report_links(logger)
        except Exception as report_exc:
            try:
                logger.write(f"Не удалось сохранить финальный отчет: {type(report_exc).__name__}: {report_exc}")
            except Exception:
                print(f"Не удалось сохранить финальный отчет: {type(report_exc).__name__}: {report_exc}", flush=True)
        finally:
            logger.close()


if __name__ == "__main__":
    main()
