@echo off
setlocal enabledelayedexpansion

:: Nunno Finance - Full Setup and Run Script (Python 3.13 Compatible)
:: This script will install all dependencies and run the application

color 0A
echo ====================================
echo   NUNNO FINANCE - SETUP ^& RUN
echo ====================================
echo.

:: Check if Python is installed
echo [1/7] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)

:: Get Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [OK] Python %PYTHON_VERSION% is installed
echo.

:: Check if Node.js is installed
echo [2/7] Checking Node.js installation...
node --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js is not installed or not in PATH
    echo Please install Node.js 16+ from https://nodejs.org/
    pause
    exit /b 1
)
echo [OK] Node.js is installed
echo.

:: Check if .env file exists
echo [3/7] Checking API key configuration...
if not exist ".env" (
    echo [WARNING] No .env file found!
    echo.
    echo Creating .env file from template...
    copy ".env.example" ".env" >nul
    echo.
    echo ====================================
    echo   ACTION REQUIRED!
    echo ====================================
    echo.
    echo Please add your OpenRouter API key to:
    echo   .env
    echo.
    echo 1. Get your API key from: https://openrouter.ai/
    echo 2. Open .env in notepad
    echo 3. Replace 'your_openrouter_api_key_here' with your actual key
    echo.
    notepad .env
    echo.
    set /p continue="Press ENTER when you've added your API key, or type 'skip' to continue anyway: "
    if /i "!continue!"=="skip" (
        echo [WARNING] Continuing without API key - chat features will not work
    )
    echo.
) else (
    echo [OK] .env file exists
)
echo.

:: Upgrade pip first
echo [4/7] Upgrading pip...
python -m pip install --upgrade pip --quiet
echo [OK] pip upgraded
echo.

:: Install core backend dependencies
echo [5/7] Installing core backend dependencies...
echo This may take a few minutes...
REM cd backend
pip install fastapi uvicorn[standard] python-dotenv requests pydantic python-multipart --quiet --disable-pip-version-check
if errorlevel 1 (
    echo [ERROR] Failed to install core dependencies
    cd ..
    pause
    exit /b 1
)
echo [OK] Core dependencies installed
echo.

:: Install data science packages with pre-built wheels
echo [6/7] Installing data science packages...
echo Note: Using pre-built wheels for Python 3.13 compatibility
pip install --only-binary :all: numpy pandas ta 2>nul
if errorlevel 1 (
    echo [WARNING] Pre-built wheels not available, trying normal install...
    pip install numpy pandas ta --quiet
    if errorlevel 1 (
        echo [ERROR] Failed to install numpy/pandas/ta
        echo.
        echo SOLUTION: Install Visual Studio Build Tools or use Python 3.11
        echo Download: https://visualstudio.microsoft.com/downloads/
        echo Or install Python 3.11: https://www.python.org/downloads/release/python-3119/
        cd ..
        pause
        exit /b 1
    )
)
REM cd ..
echo [OK] Data science packages installed
echo.

:: Install frontend dependencies
echo [7/7] Installing frontend dependencies...
echo This may take a few minutes...
cd ..\nunno-frontend2

:: Check if node_modules exists
if exist "node_modules" (
    echo [INFO] node_modules folder exists, skipping install
    echo [INFO] Delete node_modules folder to force reinstall
) else (
    call npm install
    if errorlevel 1 (
        echo [ERROR] Failed to install frontend dependencies
        echo Try running manually: npm install
        cd ..
        pause
        exit /b 1
    )
)
cd ..\nunno-backend2
echo [OK] Frontend dependencies installed
echo.

:: Start the application
echo ====================================
echo   STARTING SERVERS
echo ====================================
echo.
echo Backend will start on: http://localhost:8000
echo Frontend will start on: http://localhost:5173
echo.
echo Two new windows will open:
echo   1. Backend Server (FastAPI)
echo   2. Frontend Server (Vite)
echo.
echo Your browser should open automatically to:
echo   http://localhost:5173
echo.
echo To stop the servers, close this window or press Ctrl+C
echo.

timeout /t 3 /nobreak >nul

:: Start backend server
echo Starting backend server...
start "Nunno Finance - Backend" cmd /k "echo Backend Server Running on http://localhost:8000 && echo. && python main.py"

:: Wait a bit for backend to start
timeout /t 5 /nobreak >nul

:: Start frontend server
echo Starting frontend server...
start "Nunno Finance - Frontend" cmd /k "cd ..\nunno-frontend2 && echo Frontend Server Running on http://localhost:5173 && echo. && npm run dev"

:: Wait for frontend to start
timeout /t 5 /nobreak >nul

:: Open browser
echo Opening browser...
start http://localhost:5173

echo.
echo ====================================
echo   NUNNO FINANCE IS RUNNING!
echo ====================================
echo.
echo Backend API: http://localhost:8000
echo Frontend UI: http://localhost:5173
echo API Docs: http://localhost:8000/docs
echo.
echo TIP: Try asking "Is Bitcoin a good buy right now?"
echo.
echo To stop all servers, close this window.
echo.
pause

:: Cleanup on exit
echo.
echo Stopping servers...
taskkill /FI "WindowTitle eq Nunno Finance - Backend*" /T /F >nul 2>&1
taskkill /FI "WindowTitle eq Nunno Finance - Frontend*" /T /F >nul 2>&1
echo Servers stopped.
echo.
echo Thank you for using Nunno Finance!
pause
