#!/usr/bin/python3 -O
from setuptools import setup
from Cython.Build import cythonize
import numpy as np
from distutils.extension import Extension
import os


_PACK_NAME_ = "MAB_algorithm"
_VERSION_ = "0.0.5"


def main():
    with open('README.md', 'r', encoding='utf-8') as f:
        long_description = f.read()
    with open(f"{os.path.dirname(os.path.abspath(__file__))}/setup.py", 'r', encoding='utf-8') as f:
        content = f.read()

    content = content.split('\n')
    content[0] = f"_PACK_NAME_ = \"{_PACK_NAME_}\""
    content[1] = f"_VERSION_ = \"{_VERSION_}\""
    with open(f"{os.path.dirname(os.path.abspath(__file__))}/setup.py", 'w', encoding='utf-8') as f:
        f.write('\n'.join(content))

    setup(
        name=_PACK_NAME_,
        version=_VERSION_,
        author="Antares",
        author_email="Antares0982@gmail.com",
        license="MIT",
        platforms=["Windows", "Linux"],
        description="A modern multi-armed bandits library.",
        long_description=long_description,
        long_description_content_type='text/markdown',
        url="https://github.com/Antares0982/MAB-algorithm-template",
        ext_modules=cythonize([Extension("mabCutils", [
            f"{_PACK_NAME_}/mabCutils.pyx"
        ], language="c++")]),
        include_dirs=np.get_include(),
        options={'build_ext': {"build_lib": _PACK_NAME_}},
        install_requires=[
            'numpy',
            'scipy',
            'pandas',
            'matplotlib'
        ],
        package_data={_PACK_NAME_: ["*.cpp", "*.h", "*.pyi"]},
        packages=[_PACK_NAME_]
    )


if __name__ == "__main__":
    main()
