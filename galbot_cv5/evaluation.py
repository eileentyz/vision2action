"""Perception evaluation: segmentation, centroid, position and yaw error against ground truth.

Two evaluations:

* ``evaluate_bundle``: OpenCV 5 head-stereo pipeline on a bundle (re-rendered FYP
  trials or synthetic). Masks come from cv2.dnn YOLO (``dnn``), the model-free
  HSV segmenter (``hsv``) or the ground-truth masks (``gt``, an oracle that
  isolates geometric error). Position and yaw reuse the FYP algorithms unchanged:
  head_localisation.mask_centroid/triangulate (coarse stereo point) and
  head_yaw.fit_head_yaw (silhouette fit of an upright box, yaw modulo 90 deg).
* ``evaluate_recorded_trial``: the live FYP perception as recorded during a
  mission (the pose the pick consumed, /manipulation/cube/pose) against
  simulator truth at the same time. No images; pure log analysis.
"""

import json
import math
from pathlib import Path
import time

import numpy as np

from galbot_cv5.classical import red_mask
from galbot_cv5.geometry import fyp_modules, mask_geometry
from galbot_cv5.metrics import centroid_error, mask_iou, summarise


def wrap90(deg):
    """Signed difference modulo 90 deg in [-45, 45): a square footprint's yaw is only defined mod 90."""
    return (deg + 45.0) % 90.0 - 45.0


def yaw_deg_from_quaternion(q):
    x, y, z, w = q
    return math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


class MaskSource:
    """Callable rgb -> 0/255 mask plus a name; wraps the three mask sources."""

    def __init__(self, kind, runner=None):
        if kind not in ('dnn', 'hsv', 'gt'):
            raise ValueError('mask source must be dnn, hsv or gt')
        if kind == 'dnn' and runner is None:
            raise ValueError('dnn mask source needs a YoloSegDnn runner')
        self.kind, self.runner = kind, runner

    def __call__(self, sample, side):
        if self.kind == 'gt':
            if not sample['has_gt']:
                raise ValueError('sample has no ground-truth mask')
            return sample[f'gt_{side}']
        image = sample[side]
        if self.kind == 'hsv':
            return red_mask(image)
        mask, _, _ = self.runner.best_mask(image)
        return mask


def evaluate_sample(sample, source):
    head_localisation, head_yaw = fyp_modules()
    row = dict(id=sample['id'], group=sample['group'], mask_source=source.kind,
               truth_yaw_deg=float(sample['truth_yaw_deg']),
               truth_xy_m=[float(v) for v in sample['truth_p'][:2]])
    started = time.perf_counter()
    masks = [np.asarray(source(sample, side)) for side in ('left', 'right')]
    row['segmentation_ms'] = 1000 * (time.perf_counter() - started)
    row['detected'] = [bool(m.any()) for m in masks]
    if sample['has_gt']:
        for side, mask in zip(('left', 'right'), masks):
            row[f'iou_{side}'] = mask_iou(mask, sample[f'gt_{side}'])
            row[f'centroid_px_{side}'] = centroid_error(mask, sample[f'gt_{side}'])
    geometry = mask_geometry(masks[0])
    row['image_angle_deg'] = None if geometry['angle'] is None else math.degrees(geometry['angle'])

    started = time.perf_counter()
    try:
        centroids = [head_localisation.mask_centroid(m) for m in masks]
        if any(c is None for c in centroids):
            raise ValueError('object not segmented in both head images')
        R, o, K = sample['R'], sample['o'], sample['K']
        baseline = float(np.linalg.norm(o[1] - o[0]))     # as the FYP replay tool
        point = head_localisation.triangulate(centroids[0], centroids[1], K.flatten(), baseline)
        base = R[0] @ point + o[0]
        row['triangulated_xyz_m'] = [float(v) for v in base]
        row['triangulation_xy_error_mm'] = 1000 * float(np.linalg.norm(base[:2] - sample['truth_p'][:2]))
        row['triangulation_z_error_mm'] = 1000 * float(base[2] - sample['truth_p'][2])
        binary = [(m > 0).astype(np.uint8) for m in masks]
        fit = head_yaw.fit_head_yaw(binary, K, [(R[0], o[0]), (R[1], o[1])], base[:2],
                                    float(sample['centre_z']), tuple(float(v) for v in sample['half']))
        row['fit_yaw_deg'] = float(fit.yaw_deg)
        row['fit_iou'] = float(fit.iou)
        row['yaw_error_deg'] = float(wrap90(fit.yaw_deg - float(sample['truth_yaw_deg'])))
        row['fit_xy_error_mm'] = 1000 * float(np.linalg.norm(np.asarray(fit.xy) - sample['truth_p'][:2]))
        row['pose_ok'] = True
    except ValueError as error:
        row['pose_ok'] = False
        row['pose_failure'] = str(error)
    row['pose_ms'] = 1000 * (time.perf_counter() - started)
    return row


def summarise_rows(rows):
    ok = [r for r in rows if r['pose_ok']]
    summary = dict(
        samples=len(rows), pose_estimated=len(ok),
        detected_both=sum(all(r['detected']) for r in rows),
        failures={r['id']: r['pose_failure'] for r in rows if not r['pose_ok']},
        segmentation_ms=summarise([r['segmentation_ms'] for r in rows]),
        pose_ms=summarise([r['pose_ms'] for r in rows]),
        triangulation_xy_error_mm=summarise([r['triangulation_xy_error_mm'] for r in ok]),
        triangulation_z_error_mm=summarise([abs(r['triangulation_z_error_mm']) for r in ok]),
        fit_xy_error_mm=summarise([r['fit_xy_error_mm'] for r in ok]),
        abs_yaw_error_deg=summarise([abs(r['yaw_error_deg']) for r in ok]),
        fit_iou=summarise([r['fit_iou'] for r in ok]))
    if any('iou_left' in r for r in rows):
        summary['mask_iou'] = summarise([r.get(f'iou_{s}') for r in rows for s in ('left', 'right')])
        summary['centroid_error_px'] = summarise([r.get(f'centroid_px_{s}') for r in rows
                                                  for s in ('left', 'right')])
    return summary


def evaluate_bundle(samples, sources):
    """{source kind: {'summary': ..., 'samples': [...]}} for each MaskSource."""
    results = {}
    for source in sources:
        if source.kind == 'gt' and not all(s['has_gt'] for s in samples):
            continue
        rows = [evaluate_sample(s, source) for s in samples]
        results[source.kind] = dict(summary=summarise_rows(rows), samples=rows)
    return results


# ---------------------------------------------------------------- recorded missions


def _nearest(rows, t):
    return min(rows, key=lambda r: abs(r['t'] - t)) if rows else None


LIVE_FORMAT = 'galbot-cv5-live-missions/1'


def _detector_frames(trial_dir):
    """Per-frame detector records (YOLO_FRAME events) the live object_pointcloud node logged."""
    frames = []
    for path in sorted(Path(trial_dir).glob('events/*.jsonl')):
        for line in open(path):
            if '"YOLO_FRAME"' in line:
                event = json.loads(line)
                if event.get('event_type') == 'YOLO_FRAME':
                    frames.append(event.get('metadata') or {})
    return frames


def live_record_from_trial(trial_dir, max_truth_gap_s=0.1):
    """The few numbers a live-mission evaluation needs, without images, paths or the world file.

    The first pose the pick consumed (/manipulation/cube/pose), the simulator truth nearest
    to it, the mission outcome and the live detector's per-frame statistics. This is what
    may be uploaded; the recording itself stays local.
    """
    trial_dir = Path(trial_dir)
    rows = [json.loads(line) for line in open(trial_dir / 'recording.jsonl') if line.strip()]
    picks = sorted((r for r in rows if r['kind'] == 'pick_pose'), key=lambda r: r['t'])
    truths = sorted((r for r in rows if r['kind'] == 'truth'), key=lambda r: r['t'])
    record = dict(trial=f'{trial_dir.parent.name}/{trial_dir.name}', pick_poses=len(picks),
                  pick=None, truth=None, max_truth_gap_s=max_truth_gap_s)
    result_file = trial_dir.parent / 'results.json'
    if result_file.exists():
        for entry in json.loads(result_file.read_text()):
            if Path(str(entry.get('case', ''))).name == trial_dir.name or \
                    f"{entry.get('case')}-{entry.get('repeat')}" == trial_dir.name:
                record.update(mission_succeeded=entry.get('mission_succeeded'), passed=entry.get('passed'),
                              place_error_mm=entry.get('place_error_mm'))
    if picks:
        pick = picks[0]                   # the first pose the pick consumed
        truth = _nearest(truths, pick['t'])
        record['pick'] = {k: pick[k] for k in ('t', 'p', 'q')}
        record['truth'] = {k: truth[k] for k in ('t', 'p', 'q')} if truth else None
    frames = _detector_frames(trial_dir)
    if frames:
        timed = [f['inference_total_ms'] for f in frames if isinstance(f.get('inference_total_ms'), (int, float))]
        record['detector'] = dict(
            backend=next((f['backend'] for f in frames if f.get('backend')), 'ultralytics'),
            opencv=next((f['opencv'] for f in frames if f.get('opencv')), None),
            frames=len(frames), frames_with_cube=sum(bool(f.get('accepted_mask_produced')) for f in frames),
            errors=sum('inference_error' in f for f in frames),
            # The first frame includes model warm-up in the Ultralytics adapter; keep it out of latency.
            inference_ms=summarise(timed[1:]),
            forward_ms=summarise([f['forward_ms'] for f in frames if isinstance(f.get('forward_ms'), (int, float))]))
    return record


def evaluate_live_record(record):
    """Live perception error of one mission record (live_record_from_trial output)."""
    out = {k: record.get(k) for k in ('trial', 'pick_poses', 'mission_succeeded', 'passed', 'place_error_mm')
           if k in record}
    if record.get('detector'):
        out['detector'] = record['detector']
    pick, truth = record.get('pick'), record.get('truth')
    gap = float(record.get('max_truth_gap_s', 0.1))
    if not pick:
        out['status'] = 'no pick pose recorded'
        return out
    if truth is None or abs(truth['t'] - pick['t']) > gap:
        out['status'] = 'no ground truth within %.2f s of the pick pose' % gap
        return out
    p, q = np.asarray(pick['p'], float), np.asarray(truth['p'], float)
    out.update(status='ok', truth_gap_s=float(truth['t'] - pick['t']),
               position_xy_error_mm=1000 * float(np.linalg.norm(p[:2] - q[:2])),
               position_z_error_mm=1000 * float(p[2] - q[2]),
               truth_yaw_deg=yaw_deg_from_quaternion(truth['q']),
               measured_yaw_deg=yaw_deg_from_quaternion(pick['q']))
    out['yaw_error_deg'] = float(wrap90(out['measured_yaw_deg'] - out['truth_yaw_deg']))
    return out


def evaluate_recorded_trial(trial_dir, max_truth_gap_s=0.1):
    """Live FYP perception error from a recorded trial (recording.jsonl [+ results.json, events/])."""
    return evaluate_live_record(live_record_from_trial(trial_dir, max_truth_gap_s))


def load_live_missions(path):
    """Read an uploaded live-missions file; only plain JSON numbers and names are accepted."""
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict) or data.get('format') != LIVE_FORMAT or not isinstance(data.get('missions'), list):
        raise ValueError(f'{Path(path).name}: not a {LIVE_FORMAT} file')
    if len(data['missions']) > 200:
        raise ValueError('at most 200 missions per live-missions file')
    for record in data['missions']:
        if not isinstance(record, dict) or not isinstance(record.get('trial'), str):
            raise ValueError('each mission needs a "trial" name')
        for key in ('pick', 'truth'):
            pose = record.get(key)
            if pose is not None and (not isinstance(pose, dict) or len(pose.get('p', [])) != 3
                                     or len(pose.get('q', [])) != 4
                                     or not np.isfinite([pose.get('t'), *pose['p'], *pose['q']]).all()):
                raise ValueError(f"{record['trial']}: {key} needs finite t, p[3] and q[4]")
    return data


def summarise_recorded(trials):
    ok = [t for t in trials if t.get('status') == 'ok']
    return dict(trials=len(trials), evaluated=len(ok),
                not_evaluated={t['trial']: t['status'] for t in trials if t.get('status') != 'ok'},
                position_xy_error_mm=summarise([t['position_xy_error_mm'] for t in ok]),
                position_z_error_mm=summarise([abs(t['position_z_error_mm']) for t in ok]),
                abs_yaw_error_deg=summarise([abs(t['yaw_error_deg']) for t in ok]),
                mission_succeeded=sum(bool(t.get('mission_succeeded')) for t in ok),
                missions_succeeded_total=sum(bool(t.get('mission_succeeded')) for t in trials),
                detector_backends=sorted({t['detector']['backend'] for t in trials if t.get('detector')}),
                detector_inference_ms=summarise([t['detector']['inference_ms']['median'] for t in trials
                                                 if t.get('detector') and t['detector']['inference_ms'].get('n')]))
