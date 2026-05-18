@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ========================================
echo Computer Vision Textbook - Environment Setup
echo ========================================
echo.

set "PYTHON_DIR=%~dp0python_portable"
set "PYTHON_EXE=%PYTHON_DIR%\python.exe"
set "VENV_DIR=%~dp0venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
set "USE_PORTABLE=0"

REM Use portable Python if available
if exist "%PYTHON_EXE%" (
    echo [INFO] Portable Python detected.
    set "USE_PORTABLE=1"
    goto :install_deps
)

REM Check system Python
set "SYS_VERSION="
python --version >nul 2>&1
if %errorlevel% equ 0 (
    for /f "delims=" %%v in ('python --version 2^>^&1') do set "SYS_VERSION=%%v"
    echo [INFO] System Python: !SYS_VERSION!
    python -c "import sys; v=sys.version_info; sys.exit(0 if v.major==3 and 10<=v.minor<=12 else 1)" >nul 2>&1
    if !errorlevel! equ 0 (
        echo [INFO] System Python version OK, creating venv...
        echo.
        goto :create_venv_system
    )
    echo [WARN] System Python version not in 3.10-3.12 range.
    echo [INFO] Will download portable Python 3.12.4.
    echo.
    goto :download_portable
)

REM No system Python
echo [INFO] Python not found in PATH.
echo [INFO] Will download portable Python 3.12.4.
echo.
goto :download_portable

:create_venv_system
if exist "%VENV_PYTHON%" (
    echo [INFO] Virtual environment already exists.
    set "PYTHON_EXE=%VENV_PYTHON%"
    goto :install_deps
)
python -m venv "%VENV_DIR%"
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
)
set "PYTHON_EXE=%VENV_PYTHON%"
goto :install_deps

:download_portable
echo [DOWNLOAD] Python 3.12.4 portable...
echo.

set "PYTHON_VERSION=3.12.4"
set "PYTHON_ZIP=python-%PYTHON_VERSION%-embed-amd64.zip"
set "DOWNLOAD_URL=https://www.python.org/ftp/python/%PYTHON_VERSION%/%PYTHON_ZIP%"

powershell -Command "& { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%DOWNLOAD_URL%' -OutFile '%~dp0%PYTHON_ZIP%' -UseBasicParsing }"
if errorlevel 1 (
    echo [ERROR] Download failed, check your network.
    pause
    exit /b 1
)

echo [EXTRACT] Python...
if exist "%PYTHON_DIR%" rmdir /s /q "%PYTHON_DIR%"
mkdir "%PYTHON_DIR%"
powershell -Command "Expand-Archive -Path '%~dp0%PYTHON_ZIP%' -DestinationPath '%PYTHON_DIR%' -Force"
del "%~dp0%PYTHON_ZIP%" 2>nul

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Extraction failed.
    pause
    exit /b 1
)

REM Enable site-packages for embedded Python
echo [CONFIG] Python environment...
set "PTH_FILE=%PYTHON_DIR%\python312._pth"
if exist "%PTH_FILE%" (
    echo import site>> "%PTH_FILE%"
)

echo [INSTALL] pip...
powershell -Command "& { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile '%PYTHON_DIR%\get-pip.py' -UseBasicParsing }"
if exist "%PYTHON_DIR%\get-pip.py" (
    "%PYTHON_EXE%" "%PYTHON_DIR%\get-pip.py" --no-warn-script-location -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
    del "%PYTHON_DIR%\get-pip.py" 2>nul
)

if not exist "%PYTHON_DIR%\Scripts\pip.exe" (
    echo [ERROR] pip installation failed.
    pause
    exit /b 1
)

echo [DONE] Portable Python installed successfully.
set "USE_PORTABLE=1"
echo.
goto :install_deps

:install_deps
for /f "delims=" %%v in ('"%PYTHON_EXE%" --version 2^>^&1') do set "ACTUAL_VERSION=%%v"
echo [INFO] Using: !ACTUAL_VERSION!
if not "%USE_PORTABLE%"=="1" (
    "%PYTHON_EXE%" -c "import sys; v=sys.version_info; sys.exit(0 if v.major==3 and 10<=v.minor<=12 else 1)" >nul 2>&1
    if !errorlevel! neq 0 (
        echo [WARN] Python 3.10-3.12 recommended, current: !ACTUAL_VERSION!
        echo [WARN] Some dependencies may not be compatible. Continue? (Y/N)
        set /p "CONTINUE="
        if /i "!CONTINUE!" neq "Y" (
            echo [INFO] Cancelled.
            pause
            exit /b 1
        )
    )
)
echo.
echo [INSTALL] Dependencies (first install may take a while)...

if "%USE_PORTABLE%"=="1" (
    set "PYTHONHOME=%PYTHON_DIR%"
) else (
    set "PYTHONHOME="
)

"%PYTHON_EXE%" -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn >nul 2>&1
"%PYTHON_EXE%" -m pip install -r "%~dp0requirements.txt" -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
if errorlevel 1 (
    echo [ERROR] Dependency installation failed.
    echo [HINT] Try manually: %PYTHON_EXE% -m pip install -r requirements.txt
    pause
    exit /b 1
)

echo.
echo [VERIFY] Importing core libraries...
"%PYTHON_EXE%" -c "import numpy; import cv2; import torch; print('[OK] numpy', numpy.__version__); print('[OK] opencv', cv2.__version__); print('[OK] torch', torch.__version__)"
if errorlevel 1 (
    echo [WARN] Some core libraries failed to import, check errors above.
)

echo.
echo ========================================
echo Environment setup complete!
echo ========================================
echo.
echo Next step: run run.bat to execute chapter code.
echo.

pause
