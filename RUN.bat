@echo off

set "SCRIPT_DIR=%~dp0"

REM ������������� ���������� ��� ���� � python.exe
set "PYTHON_PATH=%~dp0_BIN\python3.11.9"

REM ������������� ���� � Python, �������� ����� portable_python
SET "PYTHON_SCRIPT=%PYTHON_PATH%\Scripts"

REM ��������� ���� � Python � ��������� ���������� PATH ��� ������� ������
SET "PATH=%PYTHON_PATH%;%PYTHON_SCRIPT%;%SCRIPT_DIR%;%PATH%"

REM ������ ���������� ��������� ������

REM ������� ���� � ������ Python-�������
set "SCRIPT_NAME=main.py"


REM ������ Python-������� � �����������
%PYTHON_PATH%\python %SCRIPT_DIR%\%SCRIPT_NAME%
pause

