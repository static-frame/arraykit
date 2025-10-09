from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import site, os

AK_VERSION = "1.2.0"

def get_ext_dir(*components: tp.Iterable[str]) -> tp.Sequence[str]:
    dirs = []
    for sp in site.getsitepackages():
        fp = os.path.join(sp, *components)
        if os.path.exists(fp):
            dirs.append(fp)
    return dirs

# subclass build_ext to append NumPy's include path at build time
# class build_ext_numpy(build_ext):
#     def finalize_options(self):
#         super().finalize_options()
#         import numpy  # available thanks to [build-system].requires
#         self.include_dirs.append(numpy.get_include())

#         # If (and only if) you truly need to link npymath, try to discover it.
#         # Many extensions don't need this anymore. Safe to remove if unused.
#         try:
#             import os
#             import numpy.core as npcore
#             get_lib = getattr(npcore, "get_lib", None)
#             if callable(get_lib):
#                 libdir = get_lib()
#                 if libdir and os.path.isdir(libdir):
#                     self.library_dirs.append(libdir)
#                     # add once
#                     libs = set(self.libraries or [])
#                     if "npymath" not in libs:
#                         self.libraries = list(libs | {"npymath"})
#         except Exception:
#             pass

ext_modules = [
    Extension(
        name="arraykit._arraykit",
        sources=[
            "src/_arraykit.c",
            "src/array_go.c",
            "src/array_to_tuple.c",
            "src/block_index.c",
            "src/delimited_to_arrays.c",
            "src/methods.c",
            "src/tri_map.c",
            "src/auto_map.c",
        ],

        include_dirs=get_ext_dir('numpy', '_core', 'include') + ['src'],
        library_dirs=get_ext_dir('numpy', '_core', 'lib'),
        define_macros=[("AK_VERSION", AK_VERSION)],
        libraries=["npymath"],
    )
]


setup(ext_modules=ext_modules)


# no metadata here—keep that in pyproject.toml
# setup(ext_modules=ext_modules, cmdclass={"build_ext": build_ext_numpy})




# import site
# import os
# from os import path
# import typing as tp
# from setuptools import Extension  # type: ignore
# from setuptools import setup

# AK_VERSION = '1.2.0'

# ROOT_DIR_FP = path.abspath(path.dirname(__file__))

# def get_long_description() -> str:
#     with open(path.join(ROOT_DIR_FP, 'README.rst'), encoding='utf-8') as f:
#         msg = []
#         collect = False
#         start = -1
#         for i, line in enumerate(f):
#             if line.startswith('arraykit'):
#                 start = i + 2 # skip this line and the next
#             if i == start:
#                 collect = True
#             if collect:
#                 msg.append(line)
#     return ''.join(msg).strip()


# # NOTE: we do this to avoid importing numpy: https://stackoverflow.com/questions/54117786/add-numpy-get-include-argument-to-setuptools-without-preinstalled-numpy
# # we used to import the following to get directories:
# # from numpy.distutils.misc_util import get_info
# # import numpy as np  # type: ignore
# # get_info('npymath')['library_dirs']
# # get_info('npymath')['libraries']

# def get_ext_dir(*components: tp.Iterable[str]) -> tp.Sequence[str]:
#     dirs = []
#     for sp in site.getsitepackages():
#         fp = os.path.join(sp, *components)
#         if os.path.exists(fp):
#             dirs.append(fp)
#     return dirs

# ak_extension = Extension(
#         name='arraykit._arraykit', # build into module
#         sources=[
#             'src/_arraykit.c',
#             'src/array_go.c',
#             'src/array_to_tuple.c',
#             'src/block_index.c',
#             'src/delimited_to_arrays.c',
#             'src/methods.c',
#             'src/tri_map.c',
#             'src/auto_map.c',
#         ],
#         include_dirs=get_ext_dir('numpy', '_core', 'include') + ['src'],
#         library_dirs=get_ext_dir('numpy', '_core', 'lib'),
#         define_macros=[("AK_VERSION", AK_VERSION)],
#         libraries=['npymath'], # not including mlib at this time
#         )

# setup(
#     name='arraykit',
#     version=AK_VERSION,
#     description='Array utilities for StaticFrame',
#     long_description=get_long_description(),
#     long_description_content_type='text/x-rst', # use text/x-rst
#     python_requires='>=3.10',
#     install_requires=['numpy>=1.24.3'],
#     url='https://github.com/static-frame/arraykit',
#     author='Christopher Ariza, Brandt Bucher, Charles Burkland',
#     license='MIT',
#     # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
#     classifiers=[
#         'Development Status :: 5 - Production/Stable',
#         'Intended Audience :: Developers',
#         'Topic :: Software Development',
#         "Programming Language :: C",
#         "Programming Language :: Python :: Implementation :: CPython",
#         'License :: OSI Approved :: MIT License',
#         'Operating System :: MacOS :: MacOS X',
#         'Operating System :: Microsoft :: Windows',
#         'Operating System :: POSIX',
#         'Programming Language :: Python :: 3.10',
#         'Programming Language :: Python :: 3.11',
#         'Programming Language :: Python :: 3.12',
#         'Programming Language :: Python :: 3.13',
#         'Programming Language :: Python :: 3.14',
#         'Programming Language :: Python :: Free Threading',
#         'Typing :: Typed',
#     ],
#     keywords='numpy array',
#     packages=['arraykit'],
#     package_dir={'arraykit': 'src'},
#     package_data={'arraykit': ['__init__.pyi', 'py.typed']},
#     include_package_data=True,
#     ext_modules=[ak_extension],
# )
