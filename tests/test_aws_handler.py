"""Lambda handler with a directory-backed bucket: no AWS calls, no network."""
import json
from pathlib import Path
import shutil

import pytest

from galbot_cv5.aws_handler import LocalBucket, check_key, handler
from galbot_cv5.bundle import save_bundle
from galbot_cv5.synthetic import make_samples


def s3_event(bucket, key):
    return {'Records': [{'s3': {'bucket': {'name': bucket}, 'object': {'key': key}}}]}


@pytest.fixture()
def bucket(tmp_path):
    (tmp_path / 'inputs').mkdir()
    save_bundle(tmp_path / 'inputs/synthetic.npz', make_samples(3), dict(source='synthetic'))
    return tmp_path


def put_job(root, job_id, job):
    path = root / 'jobs' / job_id / 'job.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(job))
    return f'jobs/{job_id}/job.json'


def test_job_produces_report(bucket):
    key = put_job(bucket, 'smoke', dict(job_id='smoke', bundles=['inputs/synthetic.npz'], mask_sources=['hsv', 'gt']))
    result = handler(s3_event('b', key), client=LocalBucket(bucket))
    assert result['status'] == 'ok'
    report = json.loads((bucket / 'reports/smoke/report.json').read_text())
    assert report['environment']['opencv'].startswith('5.')
    assert report['bundles'][0]['results']['gt']['summary']['pose_estimated'] == 3
    assert '| gt | 3/3 |' in (bucket / 'reports/smoke/summary.md').read_text()


def test_url_encoded_key(bucket):
    put_job(bucket, 'a b', dict(job_id='ab', bundles=['inputs/synthetic.npz']))
    result = handler(s3_event('b', 'jobs/a+b/job.json'), client=LocalBucket(bucket))
    assert result['status'] == 'ok' and (bucket / 'reports/ab/report.json').exists()


@pytest.mark.parametrize('job, reason', [
    (dict(job_id='x', bundles=['reports/other.npz']), 'outside the allowed prefixes'),
    (dict(job_id='x', bundles=['inputs/../jobs/x/job.json']), 'outside the allowed prefixes'),
    (dict(job_id='x', bundles=['inputs/synthetic.npz'], mask_sources=['dnn'], model='inputs/m.onnx'),
     'outside the allowed prefixes'),
    (dict(job_id='x', bundles=['inputs/missing.npz']), 'FileNotFoundError'),
])
def test_rejected_jobs_write_error_report(bucket, job, reason):
    key = put_job(bucket, 'x', job)
    result = handler(s3_event('b', key), client=LocalBucket(bucket))
    assert result['status'] == 'error' and reason in result['results'][0]['error']
    assert json.loads((bucket / 'reports/x/error.json').read_text())['status'] == 'error'
    assert not (bucket / 'reports/x/report.json').exists()


def test_recorded_trials_are_dropped_in_the_cloud(bucket):
    key = put_job(bucket, 'r', dict(job_id='r', bundles=['inputs/synthetic.npz'], recorded_trials=['/home/x']))
    assert handler(s3_event('b', key), client=LocalBucket(bucket))['status'] == 'ok'
    assert json.loads((bucket / 'reports/r/report.json').read_text())['recorded'] is None


def test_live_missions_are_evaluated_in_the_cloud(bucket):
    record = dict(trial='live/yaw+0-1', mission_succeeded=True, max_truth_gap_s=0.1,
                  pick=dict(t=1.0, p=[0.5, 0.2, 0.8], q=[0, 0, 0, 1]),
                  truth=dict(t=1.01, p=[0.501, 0.2, 0.8], q=[0, 0, 0, 1]),
                  detector=dict(backend='opencv5_dnn', opencv='5.0.0', frames=5, frames_with_cube=5, errors=0,
                                inference_ms=dict(n=4, median=30.0), forward_ms=dict(n=5, median=25.0)))
    (bucket / 'inputs/live.json').write_text(json.dumps(dict(format='galbot-cv5-live-missions/1',
                                                             missions=[record])))
    key = put_job(bucket, 'l', dict(job_id='l', live_missions=['inputs/live.json']))
    assert handler(s3_event('b', key), client=LocalBucket(bucket))['status'] == 'ok'
    live = json.loads((bucket / 'reports/l/report.json').read_text())['live']
    assert live['trials'][0]['position_xy_error_mm'] == pytest.approx(1.0)
    key = put_job(bucket, 'm', dict(job_id='m', live_missions=['models/live.json']))
    assert handler(s3_event('b', key), client=LocalBucket(bucket))['status'] == 'error'


def test_non_job_keys_and_empty_events(bucket):
    assert handler({}, client=LocalBucket(bucket))['status'] == 'ignored'
    result = handler(s3_event('b', 'inputs/synthetic.npz'), client=LocalBucket(bucket))
    assert result['status'] == 'error' and 'jobs/' in result['results'][0]['error']


def test_check_key_and_local_bucket_escape(tmp_path):
    assert check_key('inputs/a.npz') == 'inputs/a.npz'
    for bad in ('/inputs/a', 'inputs/../x', 'other/a', 5):
        with pytest.raises(ValueError):
            check_key(bad)
    with pytest.raises(ValueError):
        LocalBucket(tmp_path).head_object(Bucket='b', Key='../etc/passwd')


class S3ClientError(Exception):
    """Shape of botocore's ClientError: the error code lives in ``response``."""

    def __init__(self, code):
        super().__init__(f'An error occurred ({code}) when calling the HeadObject operation')
        self.response = {'Error': {'Code': code}}


class S3LikeBucket(LocalBucket):
    """Answers a missing key the way S3 does for a role without s3:ListBucket: 403."""

    def head_object(self, Bucket, Key):
        if not self._path(Key).exists():
            raise S3ClientError('403')
        return super().head_object(Bucket, Key)


def test_missing_object_on_s3_is_reported_as_missing_file(bucket):
    key = put_job(bucket, 'm', dict(job_id='m', bundles=['inputs/missing.npz']))
    result = handler(s3_event('b', key), client=S3LikeBucket(bucket))
    error = result['results'][0]['error']
    assert result['status'] == 'error' and error.startswith('FileNotFoundError') and 'inputs/missing.npz' in error


def test_missing_sidecar_on_s3_asks_for_input_shape(bucket):
    (bucket / 'models').mkdir()
    (bucket / 'models/m.onnx').write_bytes(b'not a model')
    key = put_job(bucket, 'd', dict(job_id='d', bundles=['inputs/synthetic.npz'], mask_sources=['dnn'],
                                    model='models/m.onnx'))
    result = handler(s3_event('b', key), client=S3LikeBucket(bucket))
    assert result['status'] == 'error' and 'input_shape' in result['results'][0]['error']


def test_other_s3_errors_are_not_masked(bucket):
    class Throttled(LocalBucket):
        def head_object(self, Bucket, Key):
            raise S3ClientError('SlowDown')
    key = put_job(bucket, 's', dict(job_id='s', bundles=['inputs/synthetic.npz']))
    result = handler(s3_event('b', key), client=Throttled(bucket))
    assert 'S3ClientError' in result['results'][0]['error'] and 'SlowDown' in result['results'][0]['error']


ONNX = Path(__file__).resolve().parents[1] / 'models/cube_yolo11n_seg.onnx'


def test_dnn_job_without_input_shape_fails_clearly(bucket):
    (bucket / 'models').mkdir()
    (bucket / 'models/m.onnx').write_bytes(b'not a model')
    key = put_job(bucket, 'd', dict(job_id='d', bundles=['inputs/synthetic.npz'], mask_sources=['dnn'],
                                    model='models/m.onnx'))
    result = handler(s3_event('b', key), client=LocalBucket(bucket))
    assert result['status'] == 'error' and 'input_shape' in result['results'][0]['error']


@pytest.mark.parametrize('shape', [[480], [480, 641], ['480', 640], [0, 640]])
def test_dnn_job_rejects_bad_input_shape(bucket, shape):
    (bucket / 'models').mkdir()
    (bucket / 'models/m.onnx').write_bytes(b'x')
    key = put_job(bucket, 'd', dict(job_id='d', bundles=['inputs/synthetic.npz'], mask_sources=['dnn'],
                                    model='models/m.onnx', input_shape=shape))
    result = handler(s3_event('b', key), client=LocalBucket(bucket))
    assert result['status'] == 'error' and 'input_shape must be' in result['results'][0]['error']


@pytest.mark.skipif(not ONNX.exists(), reason='local ONNX not available')
@pytest.mark.parametrize('sidecar', [True, False])
def test_dnn_job_with_sidecar_or_explicit_shape(bucket, sidecar):
    (bucket / 'models').mkdir()
    shutil.copyfile(ONNX, bucket / 'models/cube.onnx')
    job = dict(job_id='d', bundles=['inputs/synthetic.npz'], mask_sources=['dnn'], model='models/cube.onnx')
    if sidecar:
        shutil.copyfile(ONNX.with_suffix('.json'), bucket / 'models/cube.json')
    else:
        job['input_shape'] = [480, 640]
    result = handler(s3_event('b', put_job(bucket, 'd', job)), client=LocalBucket(bucket))
    assert result['status'] == 'ok', result
    report = json.loads((bucket / 'reports/d/report.json').read_text())
    assert report['input_shape'] == [480, 640] and 'dnn' in report['bundles'][0]['results']
