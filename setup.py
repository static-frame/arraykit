# Always prefer setuptools over distutils
from setuptools import Extension  # type: ignore
from setuptools import setup
# To use a consistent encoding
from codecs import open
from os import path
import numpy as np  # type: ignore


root_dir_fp = path.abspath(path.dirname(__file__))

def get_long_description() -> str:
    return '''The ArrayKit library provides utilities for creating and transforming NumPy arrays, implementing performance critical StaticFrame operations as Python C extensions.

Code: https://github.com/InvestmentSystems/arraykit

Docs: http://arraykit.readthedocs.io

Packages: https://pypi.org/project/arraykit
'''

def get_version() -> str:
    with open(path.join(root_dir_fp, 'arraykit', '__init__.py'), encoding='utf-8') as f:
        for l in f:
            if l.startswith('__version__'):
                if '#' in l:
                    l = l.split('#')[0].strip()
                return l.split('=')[-1].strip()[1:-1]
    raise ValueError("__version__ not found!")

include_dirs = [np.get_include(), 'arraykit/core']

setup(
    name='arraykit',
    version=get_version(),
    description='Array utilities for StaticFrame',
    long_description=get_long_description(),
    python_requires='>3.6.0',
    install_requires=['numpy>=1.14.2'],
    url='https://github.com/InvestmentSystems/arraykit',
    author='Christopher Ariza',
    license='MIT',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Developers',
            'Topic :: Software Development',
            'License :: OSI Approved :: MIT License',
            'Operating System :: MacOS :: MacOS X',
            'Operating System :: Microsoft :: Windows',
            'Operating System :: POSIX',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
    ],
    keywords='numpy array',
    packages=[
    ],
    package_data={'arraykit.core': ['*.h', '*.pyi']},
    ext_modules=[
            Extension(
                    'arraykit.core.array_go',
                    [
                            'arraykit/core/array_go.c',
                            'arraykit/core/SF.c'
                    ],
                    include_dirs=include_dirs,
            ),
            Extension(
                    'arraykit.core.util',
                    [
                            'arraykit/core/util.c',
                            'arraykit/core/SF.c'
                    ],
                    include_dirs=include_dirs,
            ),
    ],
)
