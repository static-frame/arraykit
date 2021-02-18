from setuptools import Extension  # type: ignore
from setuptools import setup
import numpy as np  # type: ignore


AK_VERSION = '0.1.0'


def get_long_description() -> str:
    return '''The ArrayKit library provides utilities for creating and transforming NumPy arrays, implementing performance-critical StaticFrame operations as Python C extensions.

Code: https://github.com/InvestmentSystems/arraykit

Docs: http://arraykit.readthedocs.io

Packages: https://pypi.org/project/arraykit
'''


setup(
    name='arraykit',
    version=AK_VERSION,
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
    packages=[],
    package_data={'': ['*.pyi']},
    ext_modules=[
        Extension(
            name='arraykit',
            sources=['arraykit.c'],
            include_dirs=[np.get_include()],
            define_macros=[("AK_VERSION", AK_VERSION)],
        ),
    ],
)
