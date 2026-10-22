"""Evaluation framework: bundle I/O, synthetic data, stereo evaluation, recorded missions, reports."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from galbot_cv5.bundle import FORMAT, load_bundle, save_bundle
from galbot_cv5.evaluation import (LIVE_FORMAT, MaskSource, evaluate_bundle, evaluate_live_record,
                                   evaluate_recorded_trial, evaluate_sample, live_record_from_trial,
                                   load_live_missions, summarise_recorded, wrap90, yaw_deg_from_quaternion)
from galbot_cv5.report import render_markdown, run_job, validate_job
from galbot_cv5.synthetic import make_samples, project, rig

HERE = Path(__file__).resolve().parents[1]
FYP_BUNDLE = HERE / 'build/bundles/headless_sweep.npz'
ONNX = HERE / 'models/cube_yolo11n_seg.onnx'


@pytest.fixture(scope='module')
def synthetic():
    return make_samples()                 # default 12 samples, seed 20261022 (the committed run)


def test_wrap90_and_quaternion_yaw():
    assert wrap90(0) == 0 and wrap90(90) == 0 and wrap90(-45) == -45 and wrap90(45) == -45
    assert wrap90(88) == pytest.approx(-2) and wrap90(-46) == pytest.approx(44)
    s = np.sin(np.radians(15))
    c = np.cos(np.radians(15))
    assert yaw_deg_from_quaternion([0, 0, s, c]) == pytest.approx(30)


def test_bundle_roundtrip_and_validation(tmp_path, synthetic):
    path = tmp_path / 'b.npz'
    save_bundle(path, synthetic, dict(source='synthetic'))
    samples, meta = load_bundle(path)
    assert meta['format'] == FORMAT and meta['samples'] == len(synthetic)
    for a, b in zip(samples, synthetic):
        assert a['id'] == b['id'] and a['has_gt'] is True
        assert np.array_equal(a['left'], b['left']) and np.array_equal(a['gt_right'], b['gt_right'])
        assert a['truth_yaw_deg'] == pytest.approx(b['truth_yaw_deg'])
    with pytest.raises(ValueError):
        save_bundle(tmp_path / 'empty.npz', [], {})


def test_bundle_rejects_wrong_format_and_pickles(tmp_path, synthetic):
    path = tmp_path / 'bad.npz'
    np.savez(path, meta=np.array(json.dumps({'format': 'other'})))
    with pytest.raises(ValueError, match='Unsupported'):
        load_bundle(path)
    np.savez(path, meta=np.array([{'format': FORMAT}], dtype=object))
    with pytest.raises(ValueError):          # object arrays need pickle, which is disabled
        load_bundle(path)


def test_synthetic_is_deterministic_and_geometrically_consistent(synthetic):
    again = make_samples()
    digest = lambda s: hashlib.sha256(s['left'].tobytes() + s['gt_right'].tobytes()).hexdigest()
    assert [digest(s) for s in synthetic] == [digest(s) for s in again]
    K, R, o = rig()
    for s in synthetic:
        assert s['gt_left'].any() and s['gt_right'].any()
        (ul, _), (ur, _) = project([s['truth_p']], K, R[0], o[0])[0], project([s['truth_p']], K, R[1], o[1])[0]
        assert ul - ur > 2                                   # positive disparity: right camera at +X
        assert -45 <= s['truth_yaw_deg'] < 45


def test_gt_masks_recover_synthetic_pose(synthetic):
    rows = [evaluate_sample(s, MaskSource('gt')) for s in synthetic]
    assert all(r['pose_ok'] for r in rows)
    # Regression bounds just above the measured 12-sample run (yaw max 0.84 deg, fit XY max 0.51 mm).
    assert max(abs(r['yaw_error_deg']) for r in rows) < 1.0
    assert max(r['fit_xy_error_mm'] for r in rows) < 1.0
    assert max(r['triangulation_xy_error_mm'] for r in rows) < 2.0


def test_hsv_segments_synthetic_scenes(synthetic):
    rows = [evaluate_sample(s, MaskSource('hsv')) for s in synthetic]
    assert all(r['pose_ok'] for r in rows)
    assert min(min(r['iou_left'], r['iou_right']) for r in rows) > 0.94   # measured min 0.949
    assert max(abs(r['yaw_error_deg']) for r in rows) < 3.0


def test_missing_object_is_a_reported_failure(synthetic):
    sample = dict(synthetic[0])
    sample['gt_left'] = np.zeros_like(sample['gt_left'])
    row = evaluate_sample(sample, MaskSource('gt'))
    assert row['pose_ok'] is False and 'both head images' in row['pose_failure']
    assert row['detected'] == [False, True]


def test_gt_source_skipped_without_ground_truth(synthetic):
    samples = [dict(s, has_gt=False) for s in synthetic[:2]]
    assert set(evaluate_bundle(samples, [MaskSource('gt'), MaskSource('hsv')])) == {'hsv'}
    with pytest.raises(ValueError):
        MaskSource('dnn')
    with pytest.raises(ValueError):
        MaskSource('other')


def write_recording(trial, rows):
    trial.mkdir(parents=True)
    (trial / 'recording.jsonl').write_text('\n'.join(json.dumps(r) for r in rows) + '\n')


def quat(yaw_deg):
    h = np.radians(yaw_deg) / 2
    return [0.0, 0.0, float(np.sin(h)), float(np.cos(h))]


def test_recorded_trial_errors(tmp_path):
    trial = tmp_path / 'sweep' / 'yaw+30-1'
    write_recording(trial, [
        dict(kind='truth', t=9.0, p=[0.5, 0.2, 0.8], q=quat(0)),
        dict(kind='truth', t=10.02, p=[0.5, 0.2, 0.8], q=quat(30)),
        dict(kind='pick_pose', t=10.0, p=[0.503, 0.204, 0.8], q=quat(-58)),     # 32 deg mod 90 -> +2
        dict(kind='pick_pose', t=11.0, p=[0.6, 0.2, 0.8], q=quat(0))])          # later poses ignored
    (trial.parent / 'results.json').write_text(json.dumps(
        [dict(case='yaw+30', repeat=1, mission_succeeded=True, passed=True, place_error_mm=3.0)]))
    out = evaluate_recorded_trial(trial)
    assert out['status'] == 'ok' and out['pick_poses'] == 2
    assert out['position_xy_error_mm'] == pytest.approx(5.0)
    assert out['yaw_error_deg'] == pytest.approx(2.0)
    assert out['mission_succeeded'] is True and out['place_error_mm'] == 3.0


def test_recorded_trial_without_pose_or_truth(tmp_path):
    a = tmp_path / 'a'
    write_recording(a, [dict(kind='truth', t=1.0, p=[0, 0, 0], q=quat(0))])
    b = tmp_path / 'b'
    write_recording(b, [dict(kind='pick_pose', t=1.0, p=[0, 0, 0], q=quat(0)),
                        dict(kind='truth', t=5.0, p=[0, 0, 0], q=quat(0))])
    results = [evaluate_recorded_trial(a), evaluate_recorded_trial(b)]
    assert results[0]['status'] == 'no pick pose recorded'
    assert results[1]['status'].startswith('no ground truth')
    summary = summarise_recorded(results)
    assert summary['evaluated'] == 0 and len(summary['not_evaluated']) == 2


def live_trial(tmp_path, backend_frames=True):
    trial = tmp_path / 'live' / 'yaw+30-1'
    write_recording(trial, [
        dict(kind='truth', t=10.02, p=[0.5, 0.2, 0.8], q=quat(30), joints=[0.1] * 20),
        dict(kind='pick_pose', t=10.0, p=[0.503, 0.204, 0.8], q=quat(-58))])
    (trial.parent / 'results.json').write_text(json.dumps(
        [dict(case='yaw+30', repeat=1, mission_succeeded=True, passed=True, place_error_mm=3.0)]))
    if backend_frames:
        (trial / 'events').mkdir()
        frames = [dict(event_type='YOLO_FRAME', metadata=dict(
            backend='opencv5_dnn', opencv='5.0.0', accepted_mask_produced=i != 3, inference_total_ms=ms,
            forward_ms=ms - 5)) for i, ms in enumerate([900.0, 30.0, 32.0, 28.0])]
        (trial / 'events/events-1.jsonl').write_text('\n'.join(json.dumps(f) for f in frames) + '\n')
    return trial


def test_live_record_is_sanitised_and_evaluates_like_the_recording(tmp_path):
    trial = live_trial(tmp_path)
    record = live_record_from_trial(trial)
    assert set(record['pick']) == {'t', 'p', 'q'} and 'joints' not in record['truth']
    assert str(tmp_path) not in json.dumps(record)
    d = record['detector']
    assert d['backend'] == 'opencv5_dnn' and d['opencv'] == '5.0.0'
    assert d['frames'] == 4 and d['frames_with_cube'] == 3
    assert d['inference_ms']['median'] == pytest.approx(30.0)          # warm-up frame excluded
    assert evaluate_live_record(json.loads(json.dumps(record))) == evaluate_recorded_trial(trial)
    out = evaluate_live_record(record)
    assert out['yaw_error_deg'] == pytest.approx(2.0) and out['position_xy_error_mm'] == pytest.approx(5.0)
    assert live_record_from_trial(live_trial(tmp_path / 'b', backend_frames=False)).get('detector') is None


def test_live_missions_file_validation(tmp_path):
    path = tmp_path / 'live.json'
    path.write_text(json.dumps(dict(format=LIVE_FORMAT, missions=[live_record_from_trial(live_trial(tmp_path))])))
    assert len(load_live_missions(path)['missions']) == 1
    for bad in [dict(format='other', missions=[]), dict(format=LIVE_FORMAT, missions=[dict(trial='x', pick=dict(
            t=1.0, p=[0, 0], q=[0, 0, 0, 1]))]), dict(format=LIVE_FORMAT, missions=[dict(pick=None)])]:
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            load_live_missions(path)


def test_run_job_with_live_missions_only(tmp_path):
    (tmp_path / 'live.json').write_text(json.dumps(dict(
        format=LIVE_FORMAT, missions=[live_record_from_trial(live_trial(tmp_path))])))
    report = run_job(dict(job_id='live-1', live_missions=['live.json'], mask_sources=['dnn']),
                     lambda key: tmp_path / key)
    assert report['bundles'] == [] and report['live']['summary']['evaluated'] == 1
    assert report['live']['summary']['detector_backends'] == ['opencv5_dnn']
    text = render_markdown(report)
    assert 'Live missions' in text and 'opencv5_dnn 5.0.0' in text and '| live/yaw+30-1 | ok |' in text


@pytest.mark.parametrize('job, message', [
    (dict(job_id='a b', bundles=['x']), 'job_id'),
    (dict(job_id='a', bundles=['x'] * 21), 'bundles'),
    (dict(job_id='a', bundles=['x'], mask_sources=['magic']), 'mask_sources'),
    (dict(job_id='a', bundles=['x'], mask_sources=['dnn']), 'model'),
    (dict(job_id='a', bundles=[]), 'nothing'),
])
def test_validate_job_rejects(job, message):
    with pytest.raises(ValueError, match=message):
        validate_job(job)


def test_run_job_and_markdown(tmp_path, synthetic):
    save_bundle(tmp_path / 's.npz', synthetic, dict(source='synthetic', description='test'))
    fetched = []

    def fetch(key):
        fetched.append(key)
        return tmp_path / key

    report = run_job(dict(job_id='t-1', bundles=['s.npz'], mask_sources=['hsv', 'gt']), fetch)
    assert fetched == ['s.npz']
    assert report['environment']['opencv'].startswith('5.')
    results = report['bundles'][0]['results']
    assert set(results) == {'hsv', 'gt'} and results['gt']['summary']['pose_estimated'] == len(synthetic)
    json.dumps(report)                                         # JSON-serialisable
    text = render_markdown(report)
    assert "| gt | 12/12 |" in text and 'Perception evaluation: t-1' in text
    with pytest.raises(ValueError, match='local runs'):
        run_job(dict(job_id='t-2', recorded_trials=['/x'], bundles=[]), fetch)


@pytest.mark.skipif(not (FYP_BUNDLE.exists() and ONNX.exists()), reason='local FYP bundle/ONNX not available')
def test_fyp_bundle_dnn_regression():
    from galbot_cv5.yolo_seg_dnn import YoloSegDnn
    samples, meta = load_bundle(FYP_BUNDLE)
    summary = evaluate_bundle(samples, [MaskSource('dnn', YoloSegDnn(ONNX))])['dnn']['summary']
    assert meta['source'] == 'fyp_replay'
    assert summary['pose_estimated'] == summary['samples'] == 21
    # Measured 2026-10-22: yaw median 1.02 deg, fit XY median 1.53 mm, mask IoU median 0.923.
    assert summary['abs_yaw_error_deg']['median'] < 1.1
    assert summary['fit_xy_error_mm']['median'] < 1.6
    assert summary['mask_iou']['median'] > 0.92
