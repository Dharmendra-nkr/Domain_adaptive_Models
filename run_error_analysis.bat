@echo off
REM Activate conda environment and run error analysis
call %CONDA_PREFIX%\..\Scripts\activate.bat biomedical-ner
cd /d "d:\OpenLab 3\biomedical-ner-dapt"
python error_analysis.py
pause
