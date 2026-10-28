from contextlib import chdir
import os
from pathlib import Path
import runpy
import unittest
from unittest import mock


class TestUnit(unittest.TestCase):
    @staticmethod
    def get_extensions(environment, output=None):
        path = Path(__file__).resolve().parents[1] / 'setup.py'
        with mock.patch.dict(os.environ, environment, clear=True), chdir(path.parent):
            with mock.patch('setuptools.setup') as setup:
                with mock.patch('subprocess.check_call'):
                    with mock.patch('subprocess.check_output', side_effect=output):
                        runpy.run_path(str(path))
                return setup.call_args.kwargs['ext_modules']

    def test_parquet_build_default(self):
        self.assertEqual(len(self.get_extensions({})), 1)

    def test_parquet_build_prefix(self):
        extensions = self.get_extensions(
            {'ARRAYKIT_WITH_PARQUET': '1', 'ARROW_HOME': '/opt/arrow'}
        )
        native = extensions[-1]
        self.assertEqual(native.name, 'arraykit._parquet')
        self.assertIn('/opt/arrow/include', native.include_dirs)
        self.assertEqual(native.library_dirs, ['/opt/arrow/lib'])
        self.assertEqual(native.libraries, ['parquet', 'arrow'])
        self.assertEqual(native.language, 'c++')

    def test_parquet_build_explicit_directories(self):
        extensions = self.get_extensions(
            {
                'ARRAYKIT_WITH_PARQUET': '1',
                'ARROW_INCLUDE_DIR': '/native/include',
                'ARROW_LIBRARY_DIR': '/native/lib64',
            }
        )
        self.assertIn('/native/include', extensions[-1].include_dirs)
        self.assertEqual(extensions[-1].library_dirs, ['/native/lib64'])

    def test_parquet_build_incomplete_directories(self):
        with self.assertRaisesRegex(RuntimeError, 'Set both'):
            self.get_extensions(
                {'ARRAYKIT_WITH_PARQUET': '1', 'ARROW_INCLUDE_DIR': '/native/include'}
            )

    def test_parquet_build_pkg_config(self):
        extensions = self.get_extensions(
            {'ARRAYKIT_WITH_PARQUET': '1'},
            output=['-I/native/include', '-L/native/lib -lparquet -larrow'],
        )
        native = extensions[-1]
        self.assertIn('-I/native/include', native.extra_compile_args)
        self.assertIn('-L/native/lib', native.extra_link_args)

    def test_parquet_build_missing_libraries(self):
        with self.assertRaisesRegex(RuntimeError, 'Native Parquet support requires'):
            self.get_extensions(
                {'ARRAYKIT_WITH_PARQUET': '1'}, output=FileNotFoundError()
            )


if __name__ == '__main__':
    unittest.main()
