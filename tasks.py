import sys

import invoke


ARTIFACTS = (
    '*.egg-info',
    '.hypothesis',
    'build',
    'dist',
    'src/*.so'
)


@invoke.task
def clean(context):
    '''Clean doc and build artifacts
    '''
    cmd = f'{sys.executable} -m pip uninstall --yes arraykit'
    context.run(cmd, echo=True, pty=True)

    for artifact in sorted(ARTIFACTS):
        context.run(f'rm -rf {artifact}', echo=True, pty=True)


@invoke.task(clean)
def build(context):
    context.run('pip install -r requirements.txt', echo=True, pty=True)
    context.run(f'{sys.executable} -m pip install .', echo=True, pty=True)


@invoke.task(build)
def test(context):
    cmd = 'pytest -s --color no --disable-pytest-warnings --tb=native'
    context.run(cmd, echo=True, pty=True)


@invoke.task(build)
def performance(context):
    context.run(f'{sys.executable} -m performance', echo=True, pty=True)


@invoke.task
def lint(context):
    '''Run pylint static analysis.
    '''
    cmd = 'pylint -f colorized *.py performance src test'
    context.run(cmd, echo=True, pty=True)
