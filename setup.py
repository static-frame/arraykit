from setuptools import Extension  # type: ignore
from setuptools import setup
from numpy.distutils.misc_util import get_info
import numpy as np  # type: ignore


AK_VERSION = '0.1.9'


def get_long_description() -> str:
    return '''The ArrayKit library provides utilities for creating and transforming NumPy arrays, implementing performance-critical StaticFrame operations as Python C extensions.

Code: https://github.com/InvestmentSystems/arraykit

Packages: https://pypi.org/project/arraykit
'''

additional_info = get_info('npymath') # We need this for various numpy C math APIs to work

# Update the dictionary to include configuration we want.
additional_info['include_dirs'] = [np.get_include()] + additional_info['include_dirs']
additional_info['define_macros'] = [("AK_VERSION", AK_VERSION)] + additional_info['define_macros']

ak_extension = Extension(
        name='arraykit._arraykit', # build into module
        sources=['src/_arraykit.c'],
        **additional_info,
)

setup(
    name='arraykit',
    version=AK_VERSION,
    description='Array utilities for StaticFrame',
    long_description=get_long_description(),
    python_requires='>3.6.0',
    install_requires=['numpy>=1.17.4'],
    url='https://github.com/InvestmentSystems/arraykit',
    author='Christopher Ariza, Brandt Bucher, Charles Burkland',
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
    packages=['arraykit'],
    package_dir={'arraykit': 'src'},
    package_data={'arraykit': ['__init__.pyi', 'py.typed']},
    include_package_data=True,
    ext_modules=[ak_extension],
)
