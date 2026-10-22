"""The OpenCV 5 mask server end to end, as the robot uses it: frames in over the socket,
masks out, compared with the Ultralytics reference on the shipped validation images."""
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import cv2
import numpy as np
import pytest

from galbot_cv5 import opencv5_protocol as wire

HERE = Path(__file__).resolve().parents[1]
MODEL = HERE / 'models/cube_yolo11n_seg.onnx'
DATA = HERE / 'data/cube_seg/images/val'
REFERENCE = HERE / 'data/reference/cube_seg_val_ultralytics.npz'
FRAMES = ['val_head_0000.png', 'val_head_0001.png', 'val_head_0002.png', 'val_wrist_0060.png']


def test_protocol_matches_the_ros2_copy():
    ros2 = HERE / 'integration/ros2/galbot_mobile_manipulation/opencv5_protocol.py'
    assert ros2.read_bytes() == (HERE / 'galbot_cv5/opencv5_protocol.py').read_bytes()


@pytest.fixture(scope='module')
def server(tmp_path_factory):
    if not (MODEL.exists() and REFERENCE.exists() and DATA.exists()):
        pytest.skip('model, reference or validation images not available')
    path = str(tmp_path_factory.mktemp('sock') / 'cv5.sock')
    env = {k: v for k, v in os.environ.items() if k != 'PYTHONPATH'}
    process = subprocess.Popen([sys.executable, str(HERE / 'tools/opencv5_mask_server.py'), '--model', str(MODEL),
                                '--socket', path, '--threads', '2', '--log-every', '0'], env=env)
    deadline = time.monotonic() + 60
    while not os.path.exists(path):
        assert process.poll() is None, 'server exited during start-up'
        assert time.monotonic() < deadline, 'server did not start'
        time.sleep(0.2)
    yield path
    process.terminate()
    process.wait(10)


def request(connection, kind, height=0, width=0, confidence=0.35, payload=b''):
    connection.sendall(wire.REQUEST.pack(wire.MAGIC, kind, height, width, confidence, 0) + payload)
    magic, status, score, forward_ms, total_ms, size = wire.RESPONSE.unpack(
        wire.receive_exactly(connection, wire.RESPONSE.size))
    assert magic == wire.MAGIC and status == wire.OK
    return score, forward_ms, wire.receive_exactly(connection, size)


def test_server_masks_match_ultralytics(server):
    reference = np.load(REFERENCE, allow_pickle=True)
    index = {name: i for i, name in enumerate(reference['names'])}
    with socket.socket(socket.AF_UNIX) as connection:
        connection.connect(server)
        _, _, hello = request(connection, wire.HELLO)
        hello = json.loads(hello)
        assert hello['opencv'].startswith('5.') and hello['names'] == {'0': 'cube'}
        for name in FRAMES:
            rgb = np.ascontiguousarray(cv2.cvtColor(cv2.imread(str(DATA / name)), cv2.COLOR_BGR2RGB))
            height, width = rgb.shape[:2]
            score, forward_ms, payload = request(connection, wire.PREDICT, height, width, payload=rgb.tobytes())
            mask = wire.unpack_mask(payload, height, width) != 0
            i = index[name]
            expected = np.unpackbits(np.asarray(reference['masks'][i], np.uint8),
                                     count=height * width).reshape(height, width) != 0
            assert np.array_equal(mask, expected), name
            assert math.isnan(score) == math.isnan(reference['confidences'][i]), name
            if not math.isnan(score):
                assert abs(score - reference['confidences'][i]) < 1e-5
            assert forward_ms > 0
