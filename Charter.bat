@echo off
setlocal
set "ROOT=%~dp0"
if not exist "%ROOT%runtime\venv\Scripts\pythonw.exe" (
    echo Charter no esta instalado todavia. Ejecuta primero "Instalar Charter.bat".
    echo Charter is not installed yet. Run "Instalar Charter.bat" first.
    pause
    exit /b 1
)
set "PYTHONPATH=%ROOT%app"
set "CHARTER_ROOT=%ROOT:~0,-1%"
start "" "%ROOT%runtime\venv\Scripts\pythonw.exe" -m charter
