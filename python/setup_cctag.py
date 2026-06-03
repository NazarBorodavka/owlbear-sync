import os
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11

class get_pybind_include(object):
    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        return pybind11.get_include(self.user)

ext_modules = [
    Extension(
        'cctag_ext',
        ['cctag_ext.cpp'],
        include_dirs=[
            get_pybind_include(),
            get_pybind_include(user=True),
            '../CCTag-develop/src',
            '/tmp/eigen_install/include/eigen3',
            '/usr/include/eigen3',
            '/usr/include/opencv4'
        ],
        library_dirs=['../CCTag-develop/build/Linux-x86_64'],
        libraries=['CCTag', 'opencv_core', 'opencv_imgproc', 'boost_serialization', 'tbb'],
        language='c++'
    ),
]

def has_flag(compiler, flagname):
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True

class BuildExt(build_ext):
    c_opts = {
        'msvc': ['/EHsc'],
        'unix': ['-std=c++14', '-O3', '-Wall', '-shared', '-fPIC'],
    }

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        for ext in self.extensions:
            ext.extra_compile_args = opts
        super().build_extensions()

setup(
    name='cctag_ext',
    version='1.0.0',
    author='CCTag',
    description='CCTag detector python wrapper',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
)
