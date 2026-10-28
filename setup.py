import typing as tp
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import site, os
import shlex
import subprocess
from pathlib import Path

AK_VERSION = Path('VERSION').read_text(encoding='utf-8').strip()


def get_ext_dir(*components: tp.Iterable[str]) -> tp.Sequence[str]:
    dirs = []
    # Check user site-packages
    user_site = site.getusersitepackages()
    if user_site:
        fp = os.path.join(user_site, *components)
        if os.path.exists(fp):
            dirs.append(fp)
    # Check system site-packages
    for sp in site.getsitepackages():
        fp = os.path.join(sp, *components)
        if os.path.exists(fp):
            dirs.append(fp)
    return dirs


ext_modules = [
    Extension(
        name='arraykit._arraykit',
        sources=[
            'src/_arraykit.c',
            'src/array_go.c',
            'src/array_to_tuple.c',
            'src/block_index.c',
            'src/delimited_to_arrays.c',
            'src/methods.c',
            'src/parquet_to_arrays.c',
            'src/tri_map.c',
            'src/auto_map.c',
        ],
        include_dirs=get_ext_dir('numpy', '_core', 'include') + ['src'],
        library_dirs=get_ext_dir('numpy', '_core', 'lib'),
        define_macros=[('AK_VERSION', AK_VERSION)],
        libraries=['npymath'],
    )
]


def get_parquet_extension() -> Extension:
    prefix = os.environ.get('ARROW_HOME')
    include_dir = os.environ.get('ARROW_INCLUDE_DIR')
    library_dir = os.environ.get('ARROW_LIBRARY_DIR')
    if prefix:
        include_dir = include_dir or os.path.join(prefix, 'include')
        library_dir = library_dir or os.path.join(prefix, 'lib')
    compile_args = ['/std:c++20'] if os.name == 'nt' else ['-std=c++20']
    link_args = []
    library_dirs = []
    include_dirs = get_ext_dir('numpy', '_core', 'include')
    if include_dir or library_dir:
        if not (include_dir and library_dir):
            raise RuntimeError('Set both ARROW_INCLUDE_DIR and ARROW_LIBRARY_DIR')
        include_dirs.append(include_dir)
        library_dirs.append(library_dir)
    else:
        try:
            subprocess.check_call(
                ['pkg-config', '--atleast-version=24', 'arrow', 'parquet']
            )
            compile_args.extend(shlex.split(subprocess.check_output(
                ['pkg-config', '--cflags', 'arrow', 'parquet'], text=True
            )))
            link_args = shlex.split(subprocess.check_output(
                ['pkg-config', '--libs', 'arrow', 'parquet'], text=True
            ))
        except (OSError, subprocess.CalledProcessError) as error:
            raise RuntimeError(
                'Native Parquet support requires Arrow/Parquet >=24 development '
                'libraries. Install their pkg-config files or set ARROW_HOME '
                '(or ARROW_INCLUDE_DIR and ARROW_LIBRARY_DIR).'
            ) from error
    runtime_dirs = library_dirs + [
        flag[2:] for flag in link_args if flag.startswith('-L')
    ]
    return Extension(
        name='arraykit._parquet',
        sources=['src/parquet_reader.cpp'],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        runtime_library_dirs=runtime_dirs if os.name != 'nt' else [],
        libraries=['parquet', 'arrow'],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        language='c++',
    )


if os.environ.get('ARRAYKIT_WITH_PARQUET') == '1':
    ext_modules.append(get_parquet_extension())

setup(ext_modules=ext_modules)
