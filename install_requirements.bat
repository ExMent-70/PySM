@echo off

set "SCRIPT_DIR=%~dp0"

REM Устанавливаем переменную для пути к python.exe
set "PYTHON_PATH=%~dp0_BIN\python3.11.9"

REM Устанавливаем путь к Python, добавляя папку portable_python
SET "PYTHON_SCRIPT=%PYTHON_PATH%\Scripts"

REM Добавляем путь к Python в системные переменные PATH для текущей сессии
SET "PATH=%PYTHON_PATH%;%PYTHON_SCRIPT%;%SCRIPT_DIR%;%PATH%"

REM Секция параметров командной строки

REM Укажите путь к вашему Python-скрипту
set "SCRIPT_NAME=scripts_utility\install_requirements\run_install_requirements.py"
echo %SCRIPT_DIR%

REM Запуск Python-скрипта с параметрами
%PYTHON_PATH%\python %SCRIPT_DIR%%SCRIPT_NAME% --requirements-file %SCRIPT_DIR%requirements.txt
pause

