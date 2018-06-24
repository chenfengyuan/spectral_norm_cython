#!/usr/bin/env python
# encoding: utf-8

from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize(["impl.pyx"],
        annotate=True
        )
)
