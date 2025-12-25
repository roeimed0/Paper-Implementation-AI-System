@echo off
REM Activation script for Paper-Implementation-AI-System
REM Run this before working on the project

echo Activating Paper-to-Code environment...

REM Activate conda environment
call conda activate paper-to-code

REM Activate venv (if you created both)
REM Uncomment if you're using venv inside conda:
REM call .venv\Scripts\activate

echo.
echo Environment activated!
echo Python location:
where python

echo.
echo To test the mock LLM:
echo   python examples\test_mock_llm.py
echo.
