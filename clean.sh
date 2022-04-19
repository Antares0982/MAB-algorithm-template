#!/bin/bash

# to build wheel:
# python3 setup.py sdist bdist_wheel --plat manylinux2014_x86_64

cdnm="$(basename `readlink -f .`)"

if [ $cdnm != "MAB_algorithm_template" ]; then
    echo "wrong directory"
    exit
fi

rm -rf dist
rm -rf build
rm -rf wheelhouse
rm -rf MAB_algorithm.egg-info
rm -f MAB_algorithm/mabCutils.cpp
rm MAB_algorithm/*.so
