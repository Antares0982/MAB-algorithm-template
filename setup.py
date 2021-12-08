#!/usr/bin/python3 -O
# from setuptools import Extension
from setuptools import find_packages, setup
from Cython.Build import cythonize
import numpy as np
from distutils.extension import Extension


def main():
    # cwd = os.path.abspath(os.path.dirname(__file__))
    # subprocess.check_call([sys.executable, "build_src/setup.py", "build_ext", "-b",
    #                        "MAB_algorithm"], cwd=cwd)

    with open('README.md', 'r', encoding='utf-8') as f:
        long_description = f.read()

    setup(
        name="MAB_algorithm",
        version="0.0.1",
        packages=find_packages(),
        author="Antares",
        author_email="Antares0982@gmail.com",
        license="MIT",
        platforms=["Windows", "Linux"],
        description="A modern multi-armed bandits library.",
        long_description=long_description,
        long_description_content_type='text/markdown',
        url="https://github.com/Antares0982/MAB-algorithm-template",
        ext_modules=cythonize([Extension("mabCutils",["MAB_algorithm/mabCutils.pyx", "MAB_algorithm/src/cutils.cpp","MAB_algorithm/src/cutils.h"])]),
        include_dirs=np.get_include(),
        install_requires=[
            'numpy',
            'scipy',
            'pandas',
            # 'PyObjC;platform_system=="Darwin"',
            # 'PyGObject;platform_system=="Linux"'
        ]
    )


if __name__ == "__main__":
    main()
