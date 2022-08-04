import site
import os
from setuptools import Extension  # type: ignore
from setuptools import setup
from pathlib import Path

AK_VERSION = '0.1.12'

def get_long_description() -> str:
    return '''The ArrayKit library provides utilities for creating and transforming NumPy arrays, implementing performance-critical StaticFrame operations as Python C extensions.

Code: https://github.com/InvestmentSystems/arraykit

Packages: https://pypi.org/project/arraykit
'''

# NOTE: we do this to avoid importing numpy: https://stackoverflow.com/questions/54117786/add-numpy-get-include-argument-to-setuptools-without-preinstalled-numpy

# NOTE site.getsitepackages()
# ['C:\\Users\\runneradmin\\AppData\\Local\\Temp\\cibw-run-zn5jvx1r\\cp37-win32\\build\\venv', 'C:\\Users\\runneradmin\\AppData\\Local\\Temp\\cibw-run-zn5jvx1r\\cp37-win32\\build\\venv\\lib\\site-packages']

# site.getsitepackages() ['C:A', 'C:A/lib/site-packages']
# np.get_include() C:A/lib/site-packages/numpy/core/include
# get_info('npymath')['library_dirs'] ['C:A/lib/site-packages/numpy/core/lib']


site_packages = site.getsitepackages()
ak_extension = Extension(
        name='arraykit._arraykit', # build into module
        sources=['src/_arraykit.c'],
        include_dirs=[os.path.join(fp, 'numpy', 'core', 'include') for fp in site_packages],
        library_dirs=[os.path.join(fp, 'numpy', 'core', 'lib') for fp in site_packages],
        define_macros=[("AK_VERSION", AK_VERSION)],
        libraries=['npymath'], # as observed from get_info('npymath')['libraries']
        )

# old approach that imported numpy
# from numpy.distutils.misc_util import get_info
# import numpy as np  # type: ignore
# print('NOTE', 'site.getsitepackages()', site.getsitepackages(), 'np.get_include()', np.get_include(), "get_info('npymath')['library_dirs']", get_info('npymath')['library_dirs'])

# ak_extension = Extension(
#         name='arraykit._arraykit', # build into module
#         sources=['src/_arraykit.c'],
#         include_dirs=[np.get_include()],
#         library_dirs=get_info('npymath')['library_dirs'],
#         # include_dirs=[os.path.join(site_pkg, 'numpy', 'core', 'include')],
#         # library_dirs=[os.path.join(site_pkg, 'numpy', 'core', 'lib')],
#         define_macros=[("AK_VERSION", AK_VERSION)],
#         libraries=['npymath'], # as observed from get_info('npymath')['libraries']
#         )

setup(
    name='arraykit',
    version=AK_VERSION,
    description='Array utilities for StaticFrame',
    long_description=get_long_description(),
    python_requires='>3.7.0',
    install_requires=['numpy>=1.18.5'],
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
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    keywords='numpy array',
    packages=['arraykit'],
    package_dir={'arraykit': 'src'},
    package_data={'arraykit': ['__init__.pyi', 'py.typed']},
    include_package_data=True,
    ext_modules=[ak_extension],
)
