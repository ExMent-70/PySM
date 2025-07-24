@echo off

set SCRIPT_DIR=%~dp0

REM Устанавливаем переменную для пути к python.exe
set PYTHON_PATH=%~dp0..\..\..\_BIN\python3.11.9
set TENSORRT_PATH=%~dp0..\..\..\_BIN\TensorRT\Lib

REM Устанавливаем путь к Python, добавляя папку portable_python
SET PYTHON_SCRIPT=%PYTHON_PATH%\Scripts

REM Добавляем путь к Python в системные переменные PATH для текущей сессии
SET PATH=%TENSORRT_PATH%;%PYTHON_PATH%;%PYTHON_SCRIPT%;%SCRIPT_DIR%;%PATH%

REM Секция параметров командной строки

REM Укажите путь к вашему Python-скрипту
set SCRIPT_NAME=run_wf_raw_analysis.py

REM Запуск Python-скрипта с параметрами
%PYTHON_PATH%\python %SCRIPT_DIR%\%SCRIPT_NAME%

pause
