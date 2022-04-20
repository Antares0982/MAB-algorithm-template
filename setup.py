_PACK_NAME_ = "MAB_algorithm"
_VERSION_ = "0.0.5"
# don't modify packname and version above
# PACK_NAME and VERSION should be declared in first two lines
if _VERSION_:
    import os

    from setuptools import Extension, find_packages, setup


try:
    from Cython.Build import cythonize
except ImportError:
    cythonize = None

extensions = [
    Extension("MAB_algorithm.mabCutils", [
              "MAB_algorithm/mabCutils.cpp", "MAB_algorithm/cutils.cpp"], language="c++"),
]

CYTHONIZE = bool(int(os.getenv("CYTHONIZE", 0))) and cythonize is not None

if CYTHONIZE:
    compiler_directives = {"language_level": 3, "embedsignature": True}
    extensions = cythonize(extensions, compiler_directives=compiler_directives)


install_requires = "numpy scipy pandas matplotlib".split()

setup(
    name=_PACK_NAME_,
    version=_VERSION_,
    author="Antares",
    author_email="Antares0982@gmail.com",
    license="MIT",
    platforms=["Windows", "Linux"],
    description="A modern multi-armed bandits library.",
    url="https://github.com/Antares0982/MAB-algorithm-template",
    ext_modules=extensions,
    install_requires=install_requires,
    packages=[_PACK_NAME_],
)
