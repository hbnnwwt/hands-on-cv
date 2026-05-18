@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

set "PYTHON_DIR=%~dp0python_portable"
set "PYTHON_EXE=%PYTHON_DIR%\python.exe"
set "VENV_PYTHON=%~dp0venv\Scripts\python.exe"

if exist "%VENV_PYTHON%" (
    set "RUN_PYTHON=%VENV_PYTHON%"
) else if exist "%PYTHON_EXE%" (
    set "RUN_PYTHON=%PYTHON_EXE%"
) else (
    echo [ERROR] Python not found. Run setup.bat first.
    pause
    exit /b 1
)

set "CHAPTERS[01]=hello_opencv.py"
set "CHAPTERS[02]=color_space.py read_show.py"
set "CHAPTERS[03]=numpy_image_ops.py"
set "CHAPTERS[04]=prompt_examples.txt"
set "CHAPTERS[05]=preprocessing.py prompt_notes.txt"
set "CHAPTERS[06]=perspective_correction.py prompt_notes.txt"
set "CHAPTERS[07]=canny_and_contours.py prompt_notes.txt"
set "CHAPTERS[08]=layout_analysis.py line_segmentation.py prompt_notes.txt"
set "CHAPTERS[09]=omr_pipeline.py prompt_notes.txt"
set "CHAPTERS[10]=template_and_shapes.py prompt_notes.txt"
set "CHAPTERS[11]=mlp_numpy.py prompt_notes.txt"
set "CHAPTERS[12]=lenet_mnist.py prompt_notes.txt"
set "CHAPTERS[13]=paddleocr_demo.py prompt_notes.txt"
set "CHAPTERS[14]=trocr_finetune.py handwriting_inference.py prompt_notes.txt"
set "CHAPTERS[15]=iou_nms.py prompt_notes.txt"
set "CHAPTERS[16]=diffusion_model.py generative_models.py prompt_notes.txt"
set "CHAPTERS[17]=pipeline.py prompt_notes.txt"
set "CHAPTERS[18]=test_omr.py prompt_notes.txt"

:menu
powershell -NoProfile -Command "Get-Content -LiteralPath '%~dp0menu.txt' -Encoding UTF8"
echo  Python: %RUN_PYTHON%
echo.

set /p "CHOICE=Select chapter (01-18, A, 0): "

if /i "%CHOICE%"=="0" exit /b 0
if /i "%CHOICE%"=="A" goto :run_all

set "PAD=%CHOICE%"
if "%PAD:~1,1%"=="" set "PAD=0%CHOICE%"

set "FILES=!CHAPTERS[%PAD%]!"
if not defined FILES (
    echo.
    echo [INFO] Chapter %PAD% has no code files.
    pause
    goto :menu
)

echo.
set "CH_DIR=%~dp0chapter%PAD%"
set "ROOT_DIR=%~dp0"
cd /d "%CH_DIR%"
for %%f in (!FILES!) do (
    echo --- %%f ---
    if "%%~xf"==".txt" (
        echo [INFO] Opening %%f in Notepad...
        notepad "%%f"
    ) else (
        "%RUN_PYTHON%" "%%f"
        if errorlevel 1 (
            echo [ERROR] %%f exited with error.
        )
    )
    echo.
)
cd /d "%ROOT_DIR%"

echo ========================================
echo  Done.
echo ========================================
pause
goto :menu

:run_all
echo.
echo ========================================
echo  Running ALL chapters ...
echo ========================================
echo.

for /L %%i in (101,1,118) do (
    set /a "CH=%%i-100"
    set "PAD=0!CH!"
    set "PAD=!PAD:~-2!"
    set "FILES=!CHAPTERS[!PAD!]!"
    if defined FILES (
        echo --- Chapter !PAD! ---
        cd /d "%~dp0chapter!PAD!"
        for %%f in (!FILES!) do (
            echo   %%f
            if "%%~xf"==".txt" (
                echo   [INFO] Opening %%f in Notepad...
                notepad "%%f"
            ) else (
                "%RUN_PYTHON%" "%%f"
                if errorlevel 1 (
                    echo   [ERROR] %%f exited with error.
                )
            )
        )
        cd /d "%~dp0"
    )
)

echo.
echo ========================================
echo  All chapters done.
echo ========================================
pause
goto :menu
