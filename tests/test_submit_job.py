"""aws/submit_job.sh end to end against a fake ``aws`` CLI: S3 is a directory and writing
jobs/<id>/job.json runs the real handler, as the S3 trigger would. No AWS calls."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from galbot_cv5.bundle import save_bundle
from galbot_cv5.synthetic import make_samples

HERE = Path(__file__).resolve().parents[1]
SCRIPT = HERE / 'aws/submit_job.sh'
ONNX = HERE / 'models/cube_yolo11n_seg.onnx'

FAKE_AWS = f'''#!{sys.executable}
import os, shutil, sys
from pathlib import Path
sys.path.insert(0, {str(HERE)!r})
root = Path(os.environ['FAKE_BUCKET_DIR'])
args = [a for a in sys.argv[1:] if a not in ('--only-show-errors',)]
assert args[0] == 's3', args

def local(uri):
    assert uri.startswith('s3://'), uri
    return root / uri[5:].split('/', 1)[1]

if args[1] == 'ls':
    prefix = args[2][5:].split('/', 1)[1]
    hits = sorted(str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()
                  and str(p.relative_to(root)).startswith(prefix))
    print('\\n'.join(hits))
    sys.exit(0 if hits else 1)
if args[1] == 'cp' and args[2] == '--recursive':
    shutil.copytree(local(args[3]), args[4], dirs_exist_ok=True)
    sys.exit(0)
if args[1] == 'cp':
    dest = local(args[3])
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args[2], dest)
    key = str(dest.relative_to(root))
    if key.startswith('jobs/') and key.endswith('job.json'):
        from galbot_cv5.aws_handler import LocalBucket, handler
        handler({{'Records': [{{'s3': {{'bucket': {{'name': 'b'}}, 'object': {{'key': key}}}}}}]}},
                client=LocalBucket(root))
    sys.exit(0)
sys.exit(f'unsupported: {{args}}')
'''


@pytest.fixture()
def env(tmp_path):
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    (bin_dir / 'aws').write_text(FAKE_AWS)
    (bin_dir / 'aws').chmod(0o755)
    bucket = tmp_path / 'bucket'
    bucket.mkdir()
    return dict(os.environ, PATH=f'{bin_dir}:{os.environ["PATH"]}', FAKE_BUCKET_DIR=str(bucket)), bucket


def job_dir(tmp_path, job):
    path = tmp_path / 'job'
    path.mkdir()
    save_bundle(path / 'synthetic.npz', make_samples(2), dict(source='synthetic'))
    (path / 'job.json').write_text(json.dumps(job))
    return path


def submit(env, path):
    return subprocess.run(['bash', str(SCRIPT), 'b', str(path), '--allow-project-data'], env=env,
                          capture_output=True, text=True, timeout=120)


def cleanup(job_id):
    shutil.rmtree(HERE / 'results/cloud' / job_id, ignore_errors=True)


def test_submit_downloads_complete_report(tmp_path, env):
    environ, bucket = env
    path = job_dir(tmp_path, dict(job_id='pytest-submit-ok', bundles=['synthetic.npz'], mask_sources=['gt']))
    try:
        result = submit(environ, path)
        assert result.returncode == 0, result.stderr
        out = HERE / 'results/cloud/pytest-submit-ok'
        assert (out / 'report.json').exists() and (out / 'summary.md').exists()
    finally:
        cleanup('pytest-submit-ok')


def test_existing_report_is_refused(tmp_path, env):
    environ, bucket = env
    (bucket / 'reports/pytest-submit-old').mkdir(parents=True)
    (bucket / 'reports/pytest-submit-old/error.json').write_text('{}')
    path = job_dir(tmp_path, dict(job_id='pytest-submit-old', bundles=['synthetic.npz'], mask_sources=['gt']))
    result = submit(environ, path)
    assert result.returncode == 2 and 'use a new job_id' in result.stderr
    assert not (bucket / 'jobs').exists()


def test_failed_job_exits_nonzero_with_error(tmp_path, env):
    environ, bucket = env
    path = job_dir(tmp_path, dict(job_id='pytest-submit-err', bundles=['synthetic.npz'], mask_sources=['bogus']))
    try:
        result = submit(environ, path)
        assert result.returncode == 1 and '"status": "error"' in result.stderr
    finally:
        cleanup('pytest-submit-err')


@pytest.mark.skipif(not ONNX.exists(), reason='local ONNX not available')
def test_model_and_sidecar_are_uploaded(tmp_path, env):
    environ, bucket = env
    path = job_dir(tmp_path, dict(job_id='pytest-submit-dnn', bundles=['synthetic.npz'], mask_sources=['dnn'],
                                  model='cube.onnx'))
    shutil.copyfile(ONNX, path / 'cube.onnx')
    shutil.copyfile(ONNX.with_suffix('.json'), path / 'cube.json')
    try:
        result = submit(environ, path)
        assert result.returncode == 0, result.stderr
        assert (bucket / 'models/cube.onnx').exists() and (bucket / 'models/cube.json').exists()
        report = json.loads((HERE / 'results/cloud/pytest-submit-dnn/report.json').read_text())
        assert report['input_shape'] == [480, 640]
    finally:
        cleanup('pytest-submit-dnn')


def test_live_missions_file_is_uploaded(tmp_path, env):
    environ, bucket = env
    path = job_dir(tmp_path, dict(job_id='pytest-submit-live', bundles=['synthetic.npz'], mask_sources=['gt'],
                                  live_missions=['live.json']))
    (path / 'live.json').write_text(json.dumps(dict(format='galbot-cv5-live-missions/1', missions=[dict(
        trial='live/yaw+0-1', pick=dict(t=1.0, p=[0.5, 0.2, 0.8], q=[0, 0, 0, 1]),
        truth=dict(t=1.0, p=[0.5, 0.2, 0.8], q=[0, 0, 0, 1]))])))
    try:
        result = submit(environ, path)
        assert result.returncode == 0, result.stderr
        assert (bucket / 'inputs/live.json').exists()
        report = json.loads((HERE / 'results/cloud/pytest-submit-live/report.json').read_text())
        assert report['live']['summary']['evaluated'] == 1
    finally:
        cleanup('pytest-submit-live')
