@echo off

set "SCRIPT_DIR=%~dp0"

REM ������������� ���������� ��� ���� � python.exe
set "PYTHON_PATH=D:\WORK_PYTHON2025\PySM\_BIN\python3.11.9"

REM ������������� ���� � Python, �������� ����� portable_python
SET "PYTHON_SCRIPT=%PYTHON_PATH%\Scripts"

REM ��������� ���� � Python � ��������� ���������� PATH ��� ������� ������
SET "PATH=%PYTHON_PATH%;%PYTHON_SCRIPT%;%SCRIPT_DIR%;%PATH%"

REM ������ ���������� ��������� ������

REM ������� ���� � ������ Python-�������
set "SCRIPT_NAME=run_mask_rmbg.py"
%PYTHON_PATH%\python %SCRIPT_NAME% -o ../../../../_output ../../../../_input
REM %PYTHON_PATH%\python %SCRIPT_NAME% -h
pause

