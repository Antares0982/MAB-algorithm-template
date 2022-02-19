#!/usr/bin/python3 -O
from setuptools import setup
from Cython.Build import cythonize
import numpy as np
from distutils.extension import Extension

_PACK_NAME = "MAB_algorithm"


def main():
    with open('README.md', 'r', encoding='utf-8') as f:
        long_description = f.read()

    setup(
        name=_PACK_NAME,
        version="0.0.1",
        author="Antares",
        author_email="Antares0982@gmail.com",
        license="MIT",
        platforms=["Windows", "Linux"],
        description="A modern multi-armed bandits library.",
        long_description=long_description,
        long_description_content_type='text/markdown',
        url="https://github.com/Antares0982/MAB-algorithm-template",
        ext_modules=cythonize([Extension("mabCutils", [
            f"{_PACK_NAME}/mabCutils.pyx",
            f"{_PACK_NAME}/src/cutils.cpp"
        ])]),
        include_dirs=np.get_include(),
        options={'build_ext': {"build_lib": _PACK_NAME}},
        install_requires=[
            'numpy',
            'scipy',
            'pandas',
            'matplotlib'
        ],
        package_data={_PACK_NAME: ["*.pyi"]},
        packages=[_PACK_NAME]
    )


if __name__ == "__main__":
    main()
