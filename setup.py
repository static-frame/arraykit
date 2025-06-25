import site
import os
from os import path
import typing as tp
from setuptools import Extension  # type: ignore
from setuptools import setup

AK_VERSION = '1.1.0'

ROOT_DIR_FP = path.abspath(path.dirname(__file__))

def get_long_description() -> str:
    with open(path.join(ROOT_DIR_FP, 'README.rst'), encoding='utf-8') as f:
        msg = []
        collect = False
        start = -1
        for i, line in enumerate(f):
            if line.startswith('arraykit'):
                start = i + 2 # skip this line and the next
            if i == start:
                collect = True
            if collect:
                msg.append(line)
    return ''.join(msg).strip()


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
        sources=[
            'src/_arraykit.c',
            'src/array_go.c',
            'src/array_to_tuple.c',
            'src/block_index.c',
            'src/delimited_to_arrays.c',
            'src/methods.c',
            'src/tri_map.c',
            'src/auto_map.c',
        ],
        include_dirs=get_ext_dir('numpy', '_core', 'include') + ['src'],
        library_dirs=get_ext_dir('numpy', '_core', 'lib'),
        define_macros=[("AK_VERSION", AK_VERSION)],
        libraries=['npymath'], # not including mlib at this time
        )

setup(
    name='arraykit',
    version=AK_VERSION,
    description='Array utilities for StaticFrame',
    long_description=get_long_description(),
    long_description_content_type='text/x-rst', # use text/x-rst
    python_requires='>=3.10',
    install_requires=['numpy>=1.24.3'],
    url='https://github.com/static-frame/arraykit',
    author='Christopher Ariza, Brandt Bucher, Charles Burkland',
    license='MIT',
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Topic :: Software Development',
        "Programming Language :: C",
        "Programming Language :: Python :: Implementation :: CPython",
        'License :: OSI Approved :: MIT License',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Typing :: Typed',
    ],
    keywords='numpy array',
    packages=['arraykit'],
    package_dir={'arraykit': 'src'},
    package_data={'arraykit': ['__init__.pyi', 'py.typed']},
    include_package_data=True,
    ext_modules=[ak_extension],
)
