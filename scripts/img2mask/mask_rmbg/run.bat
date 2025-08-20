@echo off

set "SCRIPT_DIR=%~dp0"

REM Устанавливаем переменную для пути к python.exe
set "PYTHON_PATH=D:\WORK_PYTHON2025\PySM\_BIN\python3.11.9"

REM Устанавливаем путь к Python, добавляя папку portable_python
SET "PYTHON_SCRIPT=%PYTHON_PATH%\Scripts"

REM Добавляем путь к Python в системные переменные PATH для текущей сессии
SET "PATH=%PYTHON_PATH%;%PYTHON_SCRIPT%;%SCRIPT_DIR%;%PATH%"

REM Секция параметров командной строки

REM Укажите путь к вашему Python-скрипту
set "SCRIPT_NAME=run_mask_rmbg.py"
%PYTHON_PATH%\python %SCRIPT_NAME% -o ../../../../_output ../../../../_input
REM %PYTHON_PATH%\python %SCRIPT_NAME% -h
pause

