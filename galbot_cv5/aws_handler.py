"""AWS Lambda entry point: S3 ``jobs/<id>/job.json`` -> OpenCV 5 evaluation -> ``reports/<id>/``.

Trigger: S3 ObjectCreated on prefix ``jobs/`` with suffix ``job.json`` (template.yaml).
Reads are limited to ``inputs/``, ``models/`` and ``jobs/``; writes go only to
``reports/``, so a report can never re-trigger the function. Bundles are loaded
without pickle. Recorded-mission evaluation is refused in the cloud; sanitised
live-mission records (``live_missions``, poses and detector statistics only) are allowed.

``GALBOT_LOCAL_BUCKET_DIR`` replaces S3 with a local directory: used by the
container smoke test (Lambda Runtime Interface Emulator) and unit tests only.
"""

import json
import os
from pathlib import Path
import shutil
import tempfile
import urllib.parse

from galbot_cv5.report import render_markdown, run_job

READ_PREFIXES = ('inputs/', 'models/', 'jobs/')
REPORT_PREFIX = 'reports/'
MAX_OBJECT_BYTES = 256 * 1024 * 1024


class LocalBucket:
    """Directory-backed stand-in for the few S3 calls the handler makes."""

    def __init__(self, root):
        self.root = Path(root)

    def _path(self, key):
        path = (self.root / key).resolve()
        if self.root.resolve() not in path.parents:
            raise ValueError(f'key escapes the bucket: {key}')
        return path

    def head_object(self, Bucket, Key):
        return dict(ContentLength=self._path(Key).stat().st_size)

    def download_file(self, Bucket, Key, Filename):
        shutil.copyfile(self._path(Key), Filename)

    def put_object(self, Bucket, Key, Body, ContentType=None):
        path = self._path(Key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(Body if isinstance(Body, bytes) else Body.encode())


def default_client():
    local = os.environ.get('GALBOT_LOCAL_BUCKET_DIR')
    if local:
        return LocalBucket(local)
    import boto3
    return boto3.client('s3')


def check_key(key, prefixes=READ_PREFIXES):
    if not isinstance(key, str) or '..' in key.split('/') or key.startswith('/') or not key.startswith(prefixes):
        raise ValueError(f'key {key!r} is outside the allowed prefixes {prefixes}')
    return key


def process_job(client, bucket, job_key, workdir):
    check_key(job_key, ('jobs/',))
    job_path = Path(workdir) / 'job.json'
    client.download_file(Bucket=bucket, Key=job_key, Filename=str(job_path))
    job = json.loads(job_path.read_text())
    # A job may only name inputs/ and models/ objects; recorded FYP trials are local-only.
    job.pop('recorded_trials', None)
    for key in job.get('bundles', []) + job.get('live_missions', []):
        check_key(key, ('inputs/',))
    if job.get('model'):
        check_key(job['model'], ('models/',))

    def fetch(key):
        check_key(key)
        try:
            size = client.head_object(Bucket=bucket, Key=key)['ContentLength']
        except Exception as error:
            # Without s3:ListBucket a missing key comes back as 403, not 404. Report both as a
            # missing file so callers (e.g. the optional model sidecar) can handle it as one.
            code = str(getattr(error, 'response', {}).get('Error', {}).get('Code', ''))
            if code in ('403', '404', 'Forbidden', 'NotFound', 'NoSuchKey'):
                raise FileNotFoundError(f'{key}: not found or not readable (S3 {code})') from error
            raise
        if size > MAX_OBJECT_BYTES:
            raise ValueError(f'{key} is {size} bytes; limit {MAX_OBJECT_BYTES}')
        local = Path(workdir) / 'objects' / key
        local.parent.mkdir(parents=True, exist_ok=True)
        client.download_file(Bucket=bucket, Key=key, Filename=str(local))
        return local

    report = run_job(job, fetch, allow_recorded=False)
    prefix = f"{REPORT_PREFIX}{report['job_id']}/"
    client.put_object(Bucket=bucket, Key=prefix + 'report.json', Body=json.dumps(report, indent=2).encode(),
                      ContentType='application/json')
    client.put_object(Bucket=bucket, Key=prefix + 'summary.md', Body=render_markdown(report).encode(),
                      ContentType='text/markdown')
    return dict(job_key=job_key, report=prefix + 'report.json', status='ok')


def job_records(event):
    for record in event.get('Records', []):
        s3 = record.get('s3', {})
        bucket = s3.get('bucket', {}).get('name')
        key = urllib.parse.unquote_plus(s3.get('object', {}).get('key', ''))
        if bucket and key:
            yield bucket, key


def handler(event, context=None, client=None):
    client = client or default_client()
    results = []
    for bucket, key in job_records(event):
        with tempfile.TemporaryDirectory(dir=os.environ.get('GALBOT_TMP', '/tmp')) as workdir:
            try:
                results.append(process_job(client, bucket, key, workdir))
            except Exception as error:      # report the failure next to where the report would be
                job_id = key.split('/')[1] if key.count('/') >= 2 else 'unknown'
                body = json.dumps(dict(job_key=key, status='error', error=f'{type(error).__name__}: {error}'))
                try:
                    client.put_object(Bucket=bucket, Key=f'{REPORT_PREFIX}{check_key_component(job_id)}/error.json',
                                      Body=body.encode(), ContentType='application/json')
                except Exception:
                    pass
                results.append(json.loads(body))
    if not results:
        return dict(status='ignored', reason='no S3 records')
    return dict(status='ok' if all(r['status'] == 'ok' for r in results) else 'error', results=results)


def check_key_component(text):
    return ''.join(c for c in text if c.isalnum() or c in '-_.')[:64] or 'unknown'
