"""Wire format between the OpenCV 5 mask client (ROS env) and server (OpenCV 5 env).

OpenCV 5 needs NumPy 2 and ROS 2 Jazzy uses NumPy 1, so cv2.dnn runs in a separate
process reached over a local Unix socket. Only struct and NumPy are imported here, so
both interpreters can load this module.

Request:  header ``<4sBIIfI`` (magic, kind, height, width, confidence, class id), then
          for a PREDICT the HxWx3 rgb8 pixels.
Response: header ``<4sBfffI`` (magic, status, score, forward_ms, total_ms, length), then
          ``length`` payload bytes: the np.packbits mask for PREDICT, UTF-8 JSON for
          HELLO, or a UTF-8 error message when status is ERROR.
"""

import os
import struct

import numpy as np

MAGIC = b'CV5M'
REQUEST = struct.Struct('<4sBIIfI')
RESPONSE = struct.Struct('<4sBfffI')
HELLO, PREDICT = 1, 2
OK, ERROR = 0, 1
MAX_PIXELS = 4096 * 4096


def default_socket_path():
    return os.path.join(os.environ.get('XDG_RUNTIME_DIR') or '/tmp', 'galbot_cv5_mask.sock')


def receive_exactly(connection, size):
    chunks, remaining = [], size
    while remaining:
        chunk = connection.recv(min(remaining, 1 << 20))
        if not chunk:
            raise ConnectionError('OpenCV 5 mask socket closed mid-message')
        chunks.append(chunk)
        remaining -= len(chunk)
    return b''.join(chunks)


def pack_mask(mask):
    return np.packbits(np.asarray(mask) != 0).tobytes()


def unpack_mask(payload, height, width):
    bits = np.unpackbits(np.frombuffer(payload, np.uint8), count=height * width)
    return bits.reshape(height, width).astype(np.uint8) * 255
