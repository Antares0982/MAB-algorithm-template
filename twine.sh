#!/bin/bash
cd $(dirname $0)
./clean.sh
./build.sh
file=$(ls dist | grep tar.gz)
python3 -m twine upload --repository pypi dist/$file
