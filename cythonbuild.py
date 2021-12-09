# /usr/bin/python3
from distutils.core import setup
from Cython.Build import cythonize
import numpy as np
from distutils.extension import Extension

setup(
    ext_modules=cythonize([Extension("mabCutils", [
        "MAB_algorithm/mabCutils.pyx",
        "MAB_algorithm/src/cutils.cpp"
    ])]),
    include_dirs=np.get_include(),
    options={'build_ext': {"build_lib": "MAB_algorithm"}}
)
