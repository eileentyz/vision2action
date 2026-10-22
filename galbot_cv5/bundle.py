"""Evaluation bundle: head-stereo samples with calibration and ground truth in one .npz.

Written by tools/export_trial_bundles.py (re-rendered FYP trials, NumPy 1.26 env)
and tools/make_synthetic_bundle.py (procedural, no project data); read by the
OpenCV 5 evaluator locally or in Lambda. NumPy only, so both environments can
import it, and no pickled objects, so a bundle from S3 cannot execute code.

Per sample:
  left, right      uint8 HxWx3 RGB head images (rectified pair, shared K)
  gt_left/right    uint8 HxW ground-truth object masks (0/255); has_gt False if unknown
  K                3x3 intrinsics
  R, o             (2, 3, 3), (2, 3): optical -> base rotation and origin per camera
  truth_p          object centre in base (m); truth_yaw_deg its yaw about base +Z
  centre_z, half   upright box model for the silhouette fit (centre height, half sizes)
"""
import json

import numpy as np

FORMAT = 'galbot-cv5-stereo-bundle/1'
ARRAY_FIELDS = ('left', 'right', 'gt_left', 'gt_right', 'K', 'R', 'o', 'truth_p', 'truth_yaw_deg',
                'centre_z', 'half', 'has_gt')
TEXT_FIELDS = ('id', 'group')


def save_bundle(path, samples, meta):
    if not samples:
        raise ValueError('A bundle needs at least one sample')
    arrays = {}
    for field in ARRAY_FIELDS:
        arrays[field] = np.stack([np.asarray(s[field]) for s in samples])
    for field in TEXT_FIELDS:
        arrays[field] = np.array([str(s[field]) for s in samples])
    arrays['meta'] = np.array(json.dumps(dict(meta, format=FORMAT, samples=len(samples))))
    np.savez_compressed(path, **arrays)


def load_bundle(path):
    """(samples, meta). Rejects unknown formats and inconsistent shapes."""
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data['meta']))
        if meta.get('format') != FORMAT:
            raise ValueError(f'Unsupported bundle format {meta.get("format")!r}')
        missing = [f for f in ARRAY_FIELDS + TEXT_FIELDS if f not in data.files]
        if missing:
            raise ValueError(f'Bundle is missing {missing}')
        columns = {f: data[f] for f in ARRAY_FIELDS + TEXT_FIELDS}
    count = len(columns['id'])
    if any(len(v) != count for v in columns.values()):
        raise ValueError('Bundle fields have different sample counts')
    shapes = dict(left=3, right=3, gt_left=2, gt_right=2)
    for field, ndim in shapes.items():
        if columns[field].ndim != ndim + 1 or columns[field].dtype != np.uint8:
            raise ValueError(f'{field} must be uint8 with {ndim} dims per sample')
    if columns['K'].shape[1:] != (3, 3) or columns['R'].shape[1:] != (2, 3, 3) or columns['o'].shape[1:] != (2, 3):
        raise ValueError('Calibration arrays have the wrong shape')
    samples = [{f: (columns[f][i].item() if columns[f][i].ndim == 0 else columns[f][i]) for f in columns}
               for i in range(count)]
    for s in samples:
        s['id'], s['group'] = str(s['id']), str(s['group'])
        s['has_gt'] = bool(s['has_gt'])
    return samples, meta
