@echo off
title AI Coding Assistant Launcher
cd /d "%~dp0"

set "PYEXE=C:\Users\Gurnoor\AppData\Local\Programs\Python\Python313\pythonw.exe"
if not exist "%PYEXE%" (
    where pythonw >nul 2>nul
    if %ERRORLEVEL% equ 0 (
        set "PYEXE=pythonw.exe"
    ) else (
        where python >nul 2>nul
        if %ERRORLEVEL% equ 0 (
            set "PYEXE=python.exe"
        ) else (
            echo [ERROR] Python not found! Please ensure Python is installed.
            pause
            exit /b 1
        )
    )
)

start "" "%PYEXE%" "%~dp0run_app.py"
exit /b 0
