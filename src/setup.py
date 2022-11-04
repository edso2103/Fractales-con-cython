## Cython: setup.py
from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

exts= (cythonize("mandel_cy.pyx"))
dirs=[numpy.get_include()]
setup(ext_modules = exts,include_dirs=dirs)

