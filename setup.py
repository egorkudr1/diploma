from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension

import numpy

extensions = [Extension('rank',
  sources=['src/rank.pyx', 'src/cplusplusrank.cpp'],
  language='c++', 
  include_dirs=['src', numpy.get_include()], 
  extra_compile_args=["-std=c++11"])]


setup(
    ext_modules = cythonize(extensions)
)

# setup(ext_modules = cythonize(
#            "src/rank.pyx",                 # our Cython source
#            sources=["src/cplusplusrank.cpp"],  # additional source file(s)
#            language="c++",             # generate C++ code
#       ))