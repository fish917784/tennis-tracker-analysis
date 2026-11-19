@echo off
setlocal
if not defined TENNIS_PYTHON if exist "%~dp0.venv\Scripts\python.exe" set "TENNIS_PYTHON=%~dp0.venv\Scripts\python.exe"
if not defined TENNIS_PYTHON set "TENNIS_PYTHON=python"
pushd "%~dp0" || exit /b 1
"%TENNIS_PYTHON%" -X utf8 -B "working\desktop\tennis_tacker_live.py" %*
set "TENNIS_EXIT=%ERRORLEVEL%"
popd
exit /b %TENNIS_EXIT%
