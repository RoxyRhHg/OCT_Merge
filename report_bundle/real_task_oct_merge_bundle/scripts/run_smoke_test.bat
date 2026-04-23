@echo off
set SCRIPT_DIR=%~dp0
set BUNDLE_DIR=%SCRIPT_DIR%..
python "%SCRIPT_DIR%run_real_pipeline.py" --volume-a "%BUNDLE_DIR%\smoke_data\volume_a.npy" --volume-b "%BUNDLE_DIR%\smoke_data\volume_b.npy" --output-dir "%BUNDLE_DIR%\smoke_run" --brick-size 7,6,5 --overlap-range 0.05,0.20
pause
