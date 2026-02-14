@echo off
setlocal ENABLEDELAYEDEXPANSION

title Umaplay Launcher

:: --- cd to script folder ---
cd /d "%~dp0"

echo ========================================
echo    Umaplay Launcher
echo ========================================
echo.

:: --- Check if conda is available ---
where conda >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
  echo [Error] Anaconda/Conda not found!
  echo.
  echo Please install Anaconda from:
  echo https://www.anaconda.com/download
  echo.
  echo After installing, make sure to:
  echo 1. Restart your computer
  echo 2. Run this script again
  echo.
  pause
  exit /b 1
)

echo [OK] Conda found
echo.

:: --- Initialize conda for this script ---
call conda activate base 2>nul
if %ERRORLEVEL% NEQ 0 (
  echo [Warning] Could not activate conda base environment
  echo Initializing conda...
  call conda init cmd.exe
  echo.
  echo [Info] Please close this window and run the script again.
  pause
  exit /b 0
)

:: --- Check if env_uma environment exists ---
conda env list | findstr /C:"env_uma" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
  echo [Setup] Creating conda environment 'env_uma'...
  echo This is a ONE-TIME setup and may take several minutes.
  echo.
  call conda create -n env_uma python=3.10 -y
  if %ERRORLEVEL% NEQ 0 (
    echo [Error] Failed to create conda environment
    pause
    exit /b 1
  )
  echo.
  echo [OK] Environment created
  echo.
)

:: --- Activate the env_uma environment ---
echo [Activate] Activating env_uma environment...
call conda activate env_uma
if %ERRORLEVEL% NEQ 0 (
  echo [Error] Failed to activate env_uma environment
  echo.
  echo Try running this manually:
  echo   conda activate env_uma
  echo.
  pause
  exit /b 1
)

echo [OK] Environment activated
echo.

:: --- Check if dependencies are installed ---
if not exist ".deps_installed" (
  echo [Setup] Installing Python dependencies...
  echo This may take several minutes on first run.
  echo.
  
  python -m pip install --upgrade pip
  
  if exist requirements.txt (
    pip install -r requirements.txt
    if %ERRORLEVEL% NEQ 0 (
      echo.
      echo [Error] Failed to install requirements.
      echo.
      echo Try running these commands manually:
      echo   conda activate env_uma
      echo   pip install -r requirements.txt
      echo.
      pause
      exit /b 1
    )
  ) else (
    echo [Warning] requirements.txt not found
  )
  
  echo done > ".deps_installed"
  echo.
  echo [OK] Dependencies installed
  echo.
)

:: --- Check for AMD GPU and install DirectML packages ---
if not exist ".amd_checked" (
  echo [GPU] Checking for AMD GPU...
  wmic path win32_VideoController get name 2>nul | findstr /I "AMD Radeon" >nul 2>&1
  if !ERRORLEVEL! EQU 0 (
    echo [GPU] AMD Radeon GPU detected!
    if exist requirements-amd.txt (
      echo [GPU] Installing AMD GPU acceleration packages...
      echo       This enables DirectML for GPU-accelerated inference.
      echo.
      pip install -r requirements-amd.txt --force-reinstall --no-deps
      if !ERRORLEVEL! NEQ 0 (
        echo.
        echo [Warning] AMD GPU packages failed to install.
        echo           The bot will still work, but inference runs on CPU.
        echo.
      ) else (
        echo.
        echo [OK] AMD GPU packages installed
        echo.
      )
    )
  ) else (
    echo [GPU] No AMD GPU detected, skipping DirectML setup.
    echo.
  )
  echo done > ".amd_checked"
)

:: --- Open Web UI ---
echo [Launch] Opening Web UI...
start "" "http://127.0.0.1:8000/"
timeout /t 2 /nobreak >nul

:: --- Run the application ---
echo [Run] Starting Umaplay...
echo.
echo ========================================
echo Press Ctrl+C to stop the bot
echo ========================================
echo.

python main.py

echo.
echo ========================================
echo [Exit] Umaplay stopped.
echo ========================================
pause
