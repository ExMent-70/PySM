@echo off

set SCRIPT_DIR=%~dp0

REM ������������� ���������� ��� ���� � python.exe
set PYTHON_PATH=%~dp0..\..\..\_BIN\python3.11.9
set TENSORRT_PATH=%~dp0..\..\..\_BIN\TensorRT\Lib

REM ������������� ���� � Python, �������� ����� portable_python
SET PYTHON_SCRIPT=%PYTHON_PATH%\Scripts

REM ��������� ���� � Python � ��������� ���������� PATH ��� ������� ������
SET PATH=%TENSORRT_PATH%;%PYTHON_PATH%;%PYTHON_SCRIPT%;%SCRIPT_DIR%;%PATH%

REM ������ ���������� ��������� ������

REM ������� ���� � ������ Python-�������
set SCRIPT_NAME=run_wf_raw_analysis.py

REM ������ Python-������� � �����������
%PYTHON_PATH%\python %SCRIPT_DIR%\%SCRIPT_NAME%

pause
