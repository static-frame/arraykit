import site
import os
import typing as tp
from setuptools import Extension  # type: ignore
from setuptools import setup
from pathlib import Path

AK_VERSION = '0.4.6'

def get_long_description() -> str:
    return '''The ArrayKit library provides utilities for creating and transforming NumPy arrays, implementing performance-critical StaticFrame operations as Python C extensions.

Code: https://github.com/static-frame/arraykit

Packages: https://pypi.org/project/arraykit
'''

# NOTE: we do this to avoid importing numpy: https://stackoverflow.com/questions/54117786/add-numpy-get-include-argument-to-setuptools-without-preinstalled-numpy
# we used to import the following to get directories:
# from numpy.distutils.misc_util import get_info
# import numpy as np  # type: ignore
# get_info('npymath')['library_dirs']
# get_info('npymath')['libraries']

def get_ext_dir(*components: tp.Iterable[str]) -> tp.Sequence[str]:
    dirs = []
    for sp in site.getsitepackages():
        fp = os.path.join(sp, *components)
        if os.path.exists(fp):
            dirs.append(fp)
    return dirs

ak_extension = Extension(
        name='arraykit._arraykit', # build into module
        sources=['src/_arraykit.c'],
        include_dirs=get_ext_dir('numpy', 'core', 'include'),
        library_dirs=get_ext_dir('numpy', 'core', 'lib'),
        define_macros=[("AK_VERSION", AK_VERSION)],
        libraries=['npymath'], # not including mlib at this time
        )


setup(
    name='arraykit',
    version=AK_VERSION,
    description='Array utilities for StaticFrame',
    long_description=get_long_description(),
    python_requires='>3.7.0',
    install_requires=['numpy>=1.18.5'],
    url='https://github.com/static-frame/arraykit',
    author='Christopher Ariza, Brandt Bucher, Charles Burkland',
    license='MIT',
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Topic :: Software Development',
        'License :: OSI Approved :: MIT License',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    keywords='numpy array',
    packages=['arraykit'],
    package_dir={'arraykit': 'src'},
    package_data={'arraykit': ['__init__.pyi', 'py.typed']},
    include_package_data=True,
    ext_modules=[ak_extension],
)
