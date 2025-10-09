from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import site, os
from pathlib import Path

AK_VERSION = Path("VERSION").read_text(encoding="utf-8").strip()

def get_ext_dir(*components: tp.Iterable[str]) -> tp.Sequence[str]:
    dirs = []
    for sp in site.getsitepackages():
        fp = os.path.join(sp, *components)
        if os.path.exists(fp):
            dirs.append(fp)
    return dirs

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

