#!/usr/bin/env python3
import glob
from fabric import Connection
from invoke import task

HOST        = 'ec2-13-250-110-92.ap-southeast-1.compute.amazonaws.com'
USER        = 'ubuntu'
ROOT        = 'cash'
REMOTE      = '{user}@{host}:{root}'.format(user=USER, host=HOST, root=ROOT)
VENV        = 'tensorflow2_p36'
MODEL       = 'models'
OUTPUT      = 'output_tests'

PYTHON_SCRIPTS = [
    'aug.py',
    'common.py',
    'data.py',
    'generator.py',
    'metrics.py',
    'model.py',
    'render.py',
    'train.py',
    'test.py'
]

FOLDERS = [
    'models',
    'data'
]

@task
def connect(ctx):
    ctx.conn = Connection(host=HOST, user=USER)

@task
def close(ctx):
    ctx.conn.close()

@task(pre=[connect], post=[close])
def setup(ctx):
    ctx.conn.run('mkdir -p {}'.format(ROOT))
    with ctx.conn.cd(ROOT):
        ctx.conn.run('mkdir -p {}'.format(MODEL))
        ctx.conn.run('mkdir -p {}'.format(OUTPUT))
        ctx.conn.run('sudo apt install -y dtach')
    # PIP
    ctx.conn.put('requirements.in', remote='{}/requirements.in'.format(ROOT))
    with ctx.conn.cd(ROOT):
        with ctx.conn.prefix('source activate {}'.format(VENV)):
            ctx.conn.run('pip install -U pip')
            ctx.conn.run('pip install -U -r requirements.in')

@task
def push(ctx, model=''):
    ctx.run('rsync -rv --progress {files} {remote}'.format(files=' '.join(LOCAL_FILES).join(FOLDERS), remote=REMOTE))
    model = sorted([fp for fp in glob.glob('models/*') if model and model in fp], reverse=True)
    if model:
        ctx.run('rsync -rv {folder}/ {remote}/{folder}'.format(remote=REMOTE, folder=model[0]))

@task
def pull(ctx):
    ctx.run('rsync -r {remote}/{folder}/ {folder}'.format(remote=REMOTE, folder=MODEL))
    ctx.run('rsync -r {remote}/{folder}/ {folder}'.format(remote=REMOTE, folder=OUTPUT))

@task(pre=[connect], post=[close])
def generate(ctx, model=''):
    ctx.run('rsync -rv {files} {remote}'.format(files=' '.join(PYTHON_SCRIPTS), remote=REMOTE))
    with ctx.conn.cd(ROOT):
        with ctx.conn.prefix('source activate tensorflow2_p36'):
            ctx.conn.run('dtach -A /tmp/{} python generator.py -c 1000'.format(ROOT), pty=True)


@task(pre=[connect], post=[close])
def train(ctx, model=''):
    ctx.run('rsync -rv {files} {remote}'.format(files=' '.join(PYTHON_SCRIPTS), remote=REMOTE))
    model = sorted([fp for fp in glob.glob('models/*') if model and model in fp], reverse=True)
    if model:
        ctx.run('rsync -rv {folder}/ {remote}/{folder}'.format(remote=REMOTE, folder=model[0]))

    with ctx.conn.cd(ROOT):
        with ctx.conn.prefix('source activate tensorflow2_p36'):
            ctx.conn.run('dtach -A /tmp/{} python train.py'.format(ROOT), pty=True)

    ctx.run('rsync -r {remote}/{folder}/ {folder}'.format(remote=REMOTE, folder=MODEL))
    ctx.run('rsync -r {remote}/{folder}/ {folder}'.format(remote=REMOTE, folder=OUTPUT))

@task(pre=[connect], post=[close])
def resume(ctx):
    ctx.conn.run('dtach -a /tmp/{}'.format(ROOT), pty=True)

@task(pre=[connect], post=[close])
def test(ctx, model=''):
    ctx.run('rsync -rv {files} {remote}'.format(files=' '.join(PYTHON_SCRIPTS), remote=REMOTE))
    model = sorted([fp for fp in glob.glob('models/*') if model and model in fp], reverse=True)
    if model:
        ctx.run('rsync -rv {folder}/ {remote}/{folder}'.format(remote=REMOTE, folder=model[0]))

    with ctx.conn.cd(ROOT):
        with ctx.conn.prefix('source activate tensorflow2_p36'):
            ctx.conn.run('python test.py {}'.format(model), pty=True)

    ctx.run('rsync -r {remote}/{folder}/ {folder}'.format(remote=REMOTE, folder=MODEL))
    ctx.run('rsync -r {remote}/{folder}/ {folder}'.format(remote=REMOTE, folder=OUTPUT))

@task(pre=[connect], post=[close])
def clean(ctx):
    with ctx.conn.cd(ROOT):
        ctx.conn.run('rm -rf *'.format(MODEL), pty=True)