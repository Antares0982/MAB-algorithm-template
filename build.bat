@echo off
python setup.py sdist bdist_wheel --plat manylinux2014_x86_64
del .\MAB_algorithm\mabCutils.cpp
