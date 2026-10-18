@echo off
rem AlterVox V1 启动：后端 + 自动打开页面
cd /d %~dp0
start "" http://127.0.0.1:8765
.venv\Scripts\python.exe server.py
