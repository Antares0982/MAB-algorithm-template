@echo off
python cythonbuild.py build_ext
del .\MAB_algorithm\mabCutils.cpp
