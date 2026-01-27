@echo off
echo ====================================
echo   Starting Nunno Finance
echo ====================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from https://www.python.org/
    pause
    exit /b 1
)

REM Check if Node.js is installed
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Node.js is not installed or not in PATH
    echo Please install Node.js from https://nodejs.org/
    pause
    exit /b 1
)

REM Check if backend dependencies are installed
if not exist "nunno.db" (
    echo WARNING: Database not found
    echo Please run setup-and-run.bat first to install dependencies
    pause
)

REM Check if frontend dependencies are installed
if not exist "..\nunno-frontend2\node_modules" (
    echo Installing frontend dependencies...
    pushd "..\nunno-frontend2"
    call npm install
    popd
)

echo [1/2] Starting Backend Server...
start "Nunno Backend" cmd /k "python main.py || pause"

echo Waiting for backend to initialize...
timeout /t 5 /nobreak >nul

echo [2/2] Starting Frontend Server...
start "Nunno Frontend" cmd /k "cd ..\nunno-frontend2 && npm run dev"

echo Waiting for frontend to start...
timeout /t 3 /nobreak >nul

echo.
echo ====================================
echo   Nunno Finance is starting!
echo ====================================
echo.
echo Backend:  http://localhost:8000
echo Frontend: http://localhost:5173
echo.
echo Opening browser...
timeout /t 2 /nobreak >nul
start http://localhost:5173

echo.
echo Both servers are running in separate windows.
echo Close those windows to stop the servers.
echo.
echo Press any key to stop all servers and exit...
pause >nul

echo.
echo Stopping servers...
taskkill /FI "WindowTitle eq Nunno Backend*" /T /F >nul 2>&1
taskkill /FI "WindowTitle eq Nunno Frontend*" /T /F >nul 2>&1

echo Servers stopped.
timeout /t 1 /nobreak >nul
