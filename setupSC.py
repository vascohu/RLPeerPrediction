from distutils.core import setup, Extension
from Cython.Build import cythonize

ext = Extension("InferModuleSC", sources=["InferModuleSC.pyx", "InferFunctionsSC.cpp", "SFMT\SFMT.c"], language="c++")

setup(
  ext_modules = cythonize([ext])
)