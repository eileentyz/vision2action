"""Evaluation job runner and report rendering, shared by the local CLI and the Lambda handler.

A job is a small JSON document::

    {"job_id": "smoke-001",
     "bundles": ["inputs/synthetic.npz"],          # bundle paths/keys
     "mask_sources": ["hsv", "gt"],                # any of dnn, hsv, gt
     "model": "models/cube.onnx",                  # optional; needed for dnn
     "input_shape": [480, 640],                    # optional; else read from models/cube.json
     "conf_threshold": 0.35,
     "live_missions": ["inputs/live_missions.json"]}  # optional; tools/export_live_missions.py

``run_job`` takes a ``fetch(key) -> local path`` callable, so the same code reads
local files or S3 objects. Recorded-mission evaluation (``recorded_trials``) is
local only: the recordings are FYP data and never go to the cloud by default.
``live_missions`` files hold only the pose the pick consumed, the simulator truth
next to it, the mission outcome and detector statistics, so they may be evaluated
in the cloud.
"""

import datetime as dt
import json
from pathlib import Path
import platform

import cv2
import numpy as np

from galbot_cv5.bundle import load_bundle
from galbot_cv5.evaluation import (MaskSource, evaluate_bundle, evaluate_live_record, evaluate_recorded_trial,
                                   load_live_missions, summarise_recorded)

MAX_BUNDLES = 20
VALID_SOURCES = ('dnn', 'hsv', 'gt')


def validate_job(job):
    if not isinstance(job, dict):
        raise ValueError('job must be a JSON object')
    job_id = str(job.get('job_id', ''))
    if not job_id or len(job_id) > 64 or not all(c.isalnum() or c in '-_.' for c in job_id):
        raise ValueError('job_id must be 1-64 characters of [A-Za-z0-9-_.]')
    bundles = job.get('bundles', [])
    if not isinstance(bundles, list) or len(bundles) > MAX_BUNDLES or not all(isinstance(b, str) for b in bundles):
        raise ValueError(f'bundles must be a list of at most {MAX_BUNDLES} keys')
    sources = job.get('mask_sources', ['hsv', 'gt'])
    if not sources or any(s not in VALID_SOURCES for s in sources):
        raise ValueError(f'mask_sources must be a non-empty subset of {VALID_SOURCES}')
    live = job.get('live_missions', [])
    if not isinstance(live, list) or len(live) > MAX_BUNDLES or not all(isinstance(k, str) for k in live):
        raise ValueError(f'live_missions must be a list of at most {MAX_BUNDLES} keys')
    if 'dnn' in sources and bundles and not job.get('model'):
        raise ValueError('mask source dnn needs "model" (an ONNX key)')
    if not bundles and not job.get('recorded_trials') and not live:
        raise ValueError('job has nothing to evaluate')
    return dict(job, job_id=job_id, bundles=bundles, mask_sources=list(sources), live_missions=live)


def environment():
    return dict(opencv=cv2.__version__, numpy=np.__version__, python=platform.python_version(),
                machine=platform.machine(), processor=platform.processor() or None,
                cv2_threads=cv2.getNumThreads())


def run_job(job, fetch, allow_recorded=False):
    job = validate_job(job)
    started = dt.datetime.now(dt.timezone.utc)
    runner = None
    if 'dnn' in job['mask_sources'] and job['bundles']:
        from galbot_cv5.yolo_seg_dnn import YoloSegDnn
        model_path = fetch(job['model'])
        input_shape = job.get('input_shape')
        if input_shape is None:
            # The static input size lives in the export sidecar (<model>.json); a graph run
            # at the wrong size fails inside cv2.dnn, so require one or the other.
            try:
                sidecar = fetch(str(Path(job['model']).with_suffix('.json')))
                input_shape = json.loads(Path(sidecar).read_text())['imgsz']
            except (OSError, KeyError, ValueError) as error:
                raise ValueError('dnn needs "input_shape" [h, w] in the job or the model\'s .json '
                                 f'sidecar next to it ({error})') from error
        if (not isinstance(input_shape, (list, tuple)) or len(input_shape) != 2
                or not all(isinstance(v, int) and 32 <= v <= 4096 and v % 32 == 0 for v in input_shape)):
            raise ValueError('input_shape must be [h, w], multiples of 32 in [32, 4096]')
        runner = YoloSegDnn(model_path, input_shape=input_shape,
                            conf_threshold=float(job.get('conf_threshold', 0.35)), engine=job.get('engine', 'new'))
        job['input_shape'] = list(input_shape)
    sources = [MaskSource(kind, runner if kind == 'dnn' else None) for kind in job['mask_sources']] if job['bundles'] else []
    report = dict(job_id=job['job_id'], started_utc=started.isoformat(), environment=environment(),
                  mask_sources=job['mask_sources'], model=job.get('model'), input_shape=job.get('input_shape'),
                  bundles=[], recorded=None, live=None)
    for key in job['bundles']:
        samples, meta = load_bundle(fetch(key))
        report['bundles'].append(dict(key=key, source=meta.get('source'), description=meta.get('description'),
                                      samples=len(samples), results=evaluate_bundle(samples, sources)))
    if job.get('recorded_trials'):
        if not allow_recorded:
            raise ValueError('recorded_trials is only allowed in local runs')
        trials = [evaluate_recorded_trial(path) for path in job['recorded_trials']]
        report['recorded'] = dict(summary=summarise_recorded(trials), trials=trials)
    if job['live_missions']:
        missions = [evaluate_live_record(record) for key in job['live_missions']
                    for record in load_live_missions(fetch(key))['missions']]
        report['live'] = dict(keys=job['live_missions'], summary=summarise_recorded(missions), trials=missions)
    report['finished_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
    return report


def _fmt(stats, unit='', digits=2):
    if not stats or not stats.get('n'):
        return 'n/a'
    return f"{stats['median']:.{digits}f} / {stats['p95']:.{digits}f} / {stats['max']:.{digits}f}{unit}"


def render_markdown(report):
    env = report['environment']
    lines = [f"# Perception evaluation: {report['job_id']}", '',
             f"OpenCV {env['opencv']}, NumPy {env['numpy']}, Python {env['python']} ({env['machine']}); "
             f"run {report['started_utc']} to {report['finished_utc']}.", '',
             'Statistics are median / p95 / max. Yaw errors are modulo 90 deg (square footprint).', '']
    for bundle in report['bundles']:
        lines += [f"## Bundle `{bundle['key']}` ({bundle['source']}, {bundle['samples']} stereo samples)", '']
        if bundle.get('description'):
            lines += [bundle['description'], '']
        lines += ['| Mask source | Pose estimated | Mask IoU | Centroid error (px) | Stereo XY error (mm) '
                  '| Fit XY error (mm) | Yaw error (deg) | Segmentation ms |',
                  '|---|---|---|---|---|---|---|---|']
        for kind, result in bundle['results'].items():
            s = result['summary']
            lines.append(f"| {kind} | {s['pose_estimated']}/{s['samples']} | {_fmt(s.get('mask_iou'), '', 3)} "
                         f"| {_fmt(s.get('centroid_error_px'))} | {_fmt(s['triangulation_xy_error_mm'])} "
                         f"| {_fmt(s['fit_xy_error_mm'])} | {_fmt(s['abs_yaw_error_deg'])} "
                         f"| {_fmt(s['segmentation_ms'], '', 1)} |")
        failures = {k: f for r in bundle['results'].values() for k, f in r['summary']['failures'].items()}
        if failures:
            lines += ['', 'Pose failures: ' + '; '.join(f'{k}: {v}' for k, v in failures.items())]
        lines.append('')
    if report.get('live'):
        lines += _missions_section('Live missions: the robot\'s own detector during the mission',
                                   report['live'], uploaded=True)
    if report.get('recorded'):
        lines += _missions_section('Live FYP perception in recorded missions', report['recorded'])
    return '\n'.join(lines)


def _missions_section(title, missions, uploaded=False):
    s = missions['summary']
    lines = [f'## {title}', '',
             f"{s['evaluated']}/{s['trials']} trials evaluated (first pose consumed by the pick vs simulator "
             f"truth at the same time); {s['mission_succeeded']} missions succeeded.", '']
    if uploaded:
        lines += [f"Input: {', '.join(f'`{k}`' for k in missions['keys'])} (sanitised pose records; no images).", '']
    if s.get('detector_backends'):
        lines += [f"Detector on the robot: {', '.join(s['detector_backends'])}; median per-frame inference "
                  f"{_fmt(s.get('detector_inference_ms'), ' ms', 1)} (median / p95 / max over trials).", '']
    lines += ['| Metric | median / p95 / max |', '|---|---|',
              f"| XY position error (mm) | {_fmt(s['position_xy_error_mm'])} |",
              f"| abs Z error (mm) | {_fmt(s['position_z_error_mm'])} |",
              f"| abs yaw error (deg) | {_fmt(s['abs_yaw_error_deg'])} |", '']
    rows = [t for t in missions['trials'] if t.get('detector')]
    if rows:
        lines += ['| Trial | Mission | Detector | Frames (cube) | Inference ms | XY error (mm) | Yaw error (deg) |',
                  '|---|---|---|---|---|---|---|']
        for t in rows:
            d = t['detector']
            ms = d['inference_ms'].get('median')
            lines.append(f"| {t['trial']} | {'ok' if t.get('mission_succeeded') else 'failed'} "
                         f"| {d['backend']}{' ' + d['opencv'] if d.get('opencv') else ''} "
                         f"| {d['frames']} ({d['frames_with_cube']}) | {'n/a' if ms is None else f'{ms:.1f}'} "
                         f"| {t.get('position_xy_error_mm', float('nan')):.2f} "
                         f"| {t.get('yaw_error_deg', float('nan')):+.2f} |")
        lines.append('')
    if s['not_evaluated']:
        lines += ['Not evaluated: ' + '; '.join(f'{k}: {v}' for k, v in s['not_evaluated'].items()), '']
    return lines
