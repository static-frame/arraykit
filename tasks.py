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
    # context.run('pip install -r requirements-test.txt', echo=True, pty=True)
    # keep verbose to see warnings
    context.run(f'{sys.executable} -m pip -v install .', echo=True, pty=True)


@invoke.task(build)
def test(context):
    cmd = 'pytest -s --disable-pytest-warnings --tb=native'
    context.run(cmd, echo=True, pty=True)


@invoke.task(build)
def performance(context, names=''):
    context.run(f'{sys.executable} -m performance {"--names" if names else ""} {names}', echo=True, pty=True)


@invoke.task
def lint(context):
    '''Run pylint static analysis.
    '''
    cmd = 'pylint -f colorized *.py performance src test'
    context.run(cmd, echo=True, pty=True)

