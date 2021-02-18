import sys
import os
import typing as tp

import invoke



@invoke.task
def clean(context):
    '''Clean doc and build artifacts
    '''
    context.run(f"{sys.executable} setup.py develop --uninstall", echo=True)

    for artifact in ("*.egg-info", "*.so", "build", "dist"):
        context.run(f"rm -rf {artifact}", echo=True)

    # context.run("black .", echo=True)@task(clean)


@invoke.task
def build(context):
    context.run(f"pip install -r requirements.txt", echo=True)
    context.run(f"{sys.executable} setup.py develop", echo=True)



