@echo off
chcp 65001 >nul
title Instalador de Charter
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0instalador\instalar.ps1"
echo.
pause
