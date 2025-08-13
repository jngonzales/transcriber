@echo off
title Whisper Local Backend Server

REM Navigate to the directory where this script is located
cd /d "%~dp0"

echo Activating virtual environment...
call .\venv\Scripts\activate

IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Failed to activate virtual environment.
    echo Make sure the 'venv' folder exists and is set up correctly.
    echo Press any key to exit...
    pause > NUL
    exit /b %ERRORLEVEL%
)

echo.
echo Starting Flask server (app.py)...
echo.
python app.py

REM The server will keep this window open. If the server exits, this will pause.
echo.
echo Server has stopped or encountered an error.
echo Press any key to close this window...
pause > NUL