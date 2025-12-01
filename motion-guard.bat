@echo off
set ENV_NAME=motion-guard
where conda >nul 2>nul
if %errorlevel% equ 0 (
    echo ### Attempting Conda environment activation: %ENV_NAME%
    call conda activate %ENV_NAME%
    goto execute
)
for %%D in ("%USERPROFILE%\anaconda3" "%USERPROFILE%\miniconda3" "C:\ProgramData\anaconda3" "C:\ProgramData\miniconda3") do (
    if exist "%%D\Scripts\activate.bat" (
        echo ### Conda not initialized. Searching for activate.bat to activate %ENV_NAME%.
	call %%D\Scripts\activate.bat %ENV_NAME%
	goto execute
    )
)
:execute
python "%~dp0\motion-guard.py" %*