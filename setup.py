from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "impl",
        ["impl.pyx"],
        extra_compile_args=['-march=native', '-msse', '-msse2', '-mfma', '-mfpmath=sse', '-fopenmp'],
        extra_link_args=['-march=native', '-msse', '-msse2', '-mfma', '-mfpmath=sse', '-fopenmp'],
    )
]

setup(
    name='spectral-norm',
    ext_modules=cythonize(ext_modules),
)
