@echo off
rmdir /s /q .\build >nul 2>&1
rmdir /s /q .\dist >nul 2>&1
rmdir /s /q .\MAB_algorithm.egg-info >nul 2>&1
del .\MAB_algorithm\mabCutils.cpp >nul 2>&1
@REM del .\MAB_algorithm\*.pyd
