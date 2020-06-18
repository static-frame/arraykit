import sys
import os
import typing as tp

import invoke

# -------------------------------------------------------------------------------


@invoke.task
def clean(context):
    '''Clean doc and build artifacts
    '''
    # context.run('rm -rf htmlcov')
    # context.run('rm -rf doc/build')
    context.run('rm -rf build')
    context.run('rm -rf dist')
    context.run('rm -rf *.egg-info')
    context.run('rm -rf .mypy_cache')
    context.run('rm -rf .pytest_cache')
    context.run('rm -rf .hypothesis')
    context.run('rm -rf .ipynb_checkpoints')


@invoke.task(pre=(clean,))
def build(context):
    '''Build packages
    '''
    context.run(f'{sys.executable} setup.py sdist bdist_wheel')
