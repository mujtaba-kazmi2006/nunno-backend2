@echo off
REM Copy optimized files from main app to GitHub app

echo Copying optimized prediction module...

REM Copy the advanced prediction module
copy "c:\Users\CSC\Desktop\Nunno Streamlit\NunnoFinance\backend\services\betterpredictormodule.py" "c:\Users\CSC\Desktop\Nunno Streamlit\NunnoFinance\Nunno-GitHub-Setup\nunno-backend2\services\betterpredictormodule.py"

REM Copy the optimized technical analysis
copy "c:\Users\CSC\Desktop\Nunno Streamlit\NunnoFinance\backend\services\technical_analysis.py" "c:\Users\CSC\Desktop\Nunno Streamlit\NunnoFinance\Nunno-GitHub-Setup\nunno-backend2\services\technical_analysis.py"

echo.
echo ✓ Files copied successfully!
echo.
pause
