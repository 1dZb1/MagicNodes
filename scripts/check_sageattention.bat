@echo off
setlocal
set PS=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe
"%PS%" -ExecutionPolicy Bypass -File "%~dp0\check_sageattention.ps1" %*
echo.
pause
endlocal
