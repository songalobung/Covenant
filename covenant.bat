@echo off
if exist "%~dp0.venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" -m sovdistress.cli %*
) else (
    python -m sovdistress.cli %*
)
