from distutils.core import setup, Extension
from Cython.Build import cythonize

ext = Extension("InferModuleX", sources=["InferModuleX.pyx", "InferFunctions.cpp", "SFMT\SFMT.c"], language="c++")

setup(
  ext_modules = cythonize([ext])
)

# 137.88