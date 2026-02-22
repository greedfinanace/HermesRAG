@echo off
setlocal EnableDelayedExpansion
title Offline RAG Analyzer - Control Panel

:: Setup console colors (Green text on black background)
color 0A

:: Header
echo ========================================================
echo         OFFLINE RAG RESEARCH ANALYZER LAUNCHER
echo ========================================================
echo.

:: 1. Check if Ollama is installed
where ollama >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Ollama is not installed or not in PATH.
    echo Please install Ollama from https://ollama.com/
    pause
    exit /b
)

:: 2. Check if Ollama server is running, if not start it silently in background
echo [INFO] Checking Ollama Engine Status...
curl -s -o nul http://localhost:11434/api/tags
if %errorlevel% neq 0 (
    echo [INFO] Ollama is not running. Starting engine in background...
    :: We start the `ollama serve` process without opening a new visible window
    start /B "Ollama Core" ollama serve >nul 2>&1
    
    :: Wait a few seconds for engine to boot
    timeout /t 3 /nobreak >nul
) else (
    echo [OK] Ollama Engine is already running.
)
echo.

:: 3. Fetch installed models and build an interactive menu
echo [INFO] Fetching installed local AI models...
echo.

:: We use a temporary file to store the names safely since FOR /F loops
:: run in a child cmd process and array variables can be tricky.
set counter=0
if exist "%TEMP%\ollama_models.txt" del "%TEMP%\ollama_models.txt"

:: Parse `ollama list`. Skip the header row (NAME ID SIZE...)
for /f "skip=1 tokens=1" %%a in ('ollama list') do (
    set /a counter+=1
    echo !counter!=%%a >> "%TEMP%\ollama_models.txt"
    echo   [!counter!] %%a
)

if %counter% == 0 (
    echo [ERROR] No models found!
    echo Please download a model first, for example: `ollama run qwen2.5:3b`
    pause
    exit /b
)
echo.

:: 4. Prompt user for model selection by number
:SelectModel
set /p selection="Select a model number to use for analysis (1-%counter%): "

:: Validate input is a number
echo %selection%|findstr /r "^[1-9][0-9]*$" >nul
if %errorlevel% neq 0 (
    echo Invalid input. Please enter a valid number.
    goto SelectModel
)
if %selection% gtr %counter% (
    echo Choice out of range. Please select between 1 and %counter%.
    goto SelectModel
)

:: Retrieve the actual model name based on selection
set SELECTED_MODEL=
for /f "tokens=1,2 delims==" %%A in (%TEMP%\ollama_models.txt) do (
    if "%%A"=="%selection%" set SELECTED_MODEL=%%B
)

:: Cleanup temp file
del "%TEMP%\ollama_models.txt"

:: 4.5 Find an open dynamic port to prevent connection refusal crashes
for /f "delims=" %%a in ('python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()"') do set SERVER_PORT=%%a

echo.
echo ========================================================
echo [SYSTEM SUCCESS] Environment configured.
echo - Active Model: %SELECTED_MODEL%
echo - Vector DB   : FAISS CPU
echo - Base URL    : http://127.0.0.1:%SERVER_PORT%
echo ========================================================
echo.

:: 5. Set Environment Variable for the FastAPI Backend
set OLLAMA_MODEL=%SELECTED_MODEL%

:: 6. Launch the Server
echo [INFO] Booting FastAPI Server...
echo [INFO] Press Ctrl+C in this console to shut down.
echo.

:: Open the default browser after a short delay to the dynamic port
start http://127.0.0.1:%SERVER_PORT%

:: Run uvicorn in the foreground so the console acts as the server log
uvicorn app:app --host 127.0.0.1 --port %SERVER_PORT%

endlocal
