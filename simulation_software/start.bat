@echo off
echo 启动仿真软件管理系统...
echo.

:: 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误：未找到Python，请先安装Python 3.7+
    pause
    exit /b 1
)

:: 检查依赖包
echo 检查依赖包...
pip show PyQt5 >nul 2>&1
if errorlevel 1 (
    echo 正在安装依赖包...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo 错误：依赖包安装失败
        pause
        exit /b 1
    )
)

:: 启动程序
echo 启动程序...
python main.py

pause