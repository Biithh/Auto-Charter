@echo off
chcp 65001 >nul
title Instalador de Charter - Charter installer
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0instalador\instalar.ps1"
echo.
pause
