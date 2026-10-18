@echo off
for /d /r %%D in (__pycache__) do (
    if exist "%%D\" rd /s /q "%%D"
)