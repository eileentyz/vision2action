"""Parity with the FYP's Ultralytics masks on real validation images.

Needs local, uncommitted inputs, so it is skipped unless all of them exist:
models/cube_yolo11n_seg.onnx, the reference .npz in data/reference/
(tools/reference_ultralytics.py) and the dataset (GALBOT_CUBE_DATASET, default
data/cube_seg). All three ship with the repository.
"""
import hashlib
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from galbot_cv5.yolo_seg_dnn import YoloSegDnn

HERE = Path(__file__).resolve().parents[1]
ONNX = HERE / 'models/cube_yolo11n_seg.onnx'
DATA = Path(os.environ.get('GALBOT_CUBE_DATASET', Path(__file__).resolve().parents[1] / 'data/cube_seg'))
REFERENCE = HERE / f'data/reference/{DATA.name}_val_ultralytics.npz'

pytestmark = pytest.mark.skipif(not (ONNX.exists() and REFERENCE.exists() and DATA.exists()),
                                reason='local ONNX, Ultralytics reference or dataset not available')


@pytest.mark.parametrize('engine', ['new', 'classic'])
def test_masks_match_ultralytics_reference(engine):
    reference = np.load(REFERENCE, allow_pickle=True)
    runner = YoloSegDnn(ONNX, conf_threshold=0.35, engine=engine)
    step = max(1, len(reference['names']) // 20)   # 20 images spread over head and wrist views
    for i in range(0, len(reference['names']), step):
        name, (h, w) = reference['names'][i], reference['shapes'][i]
        rgb = cv2.cvtColor(cv2.imread(str(DATA / 'images/val' / name)), cv2.COLOR_BGR2RGB)
        assert hashlib.sha256(rgb.tobytes()).hexdigest() == reference['rgb_sha256'][i]
        expected = np.unpackbits(np.asarray(reference['masks'][i], np.uint8), count=int(h * w)).reshape(h, w)
        mask, score, _ = runner.best_mask(rgb)
        assert np.count_nonzero((mask != 0) != (expected != 0)) == 0, name
        ref_score = reference['confidences'][i]
        assert (score is None) == bool(np.isnan(ref_score)), name
        if score is not None:
            assert score == pytest.approx(ref_score, abs=1e-5), name
