"""YOLO11 instance segmentation with OpenCV 5 cv2.dnn only (no torch, no Ultralytics).

Reproduces the Ultralytics 8.4 predict path used by the FYP's YoloMaskProvider
(``retina_masks=True``):

1. Letterbox to the static ONNX input (centre padding, value 114, INTER_LINEAR).
   Ultralytics predicts .pt checkpoints with ``rect=True`` (stride-aligned minimum
   rectangle), so a 640x480 camera image reaches the network unpadded at 480x640;
   the graph is exported at that size and the letterbox is then a no-op.
2. ``output0`` (1, 4 + nc + nm, N): cx, cy, w, h in letterbox pixels, nc class
   scores (already sigmoid), nm mask coefficients.
3. Confidence filter (strictly greater than the threshold), optional class
   filter, class-aware NMS (cv2.dnn.NMSBoxesBatched, IoU 0.7), max_det.
4. Boxes mapped back to the original image (remove padding, divide by gain, clip).
5. ``output1`` (1, nm, mh, mw) prototypes: logits = coefficients @ prototypes,
   remove the letterbox padding at prototype resolution, bilinear resize to the
   original image, threshold logits > 0 (sigmoid > 0.5), crop to the box.

Images are RGB uint8 HxWx3, as delivered by the FYP's rgb8 camera topics.
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
import time

import cv2
import numpy as np

ENGINES = {'auto': 'ENGINE_AUTO', 'new': 'ENGINE_NEW', 'classic': 'ENGINE_CLASSIC'}


@dataclass(frozen=True)
class Letterbox:
    gain: float
    left: int
    top: int
    content_width: int
    content_height: int


@dataclass
class Detection:
    box: np.ndarray          # x1, y1, x2, y2 in original image pixels (float)
    score: float
    class_id: int
    mask: np.ndarray         # uint8 {0, 1}, original image shape


@dataclass
class Prediction:
    detections: list
    timings_ms: dict = field(default_factory=dict)


def letterbox(image, new_shape=(640, 640), color=114):
    """Resize keeping aspect ratio and pad to new_shape (h, w), as Ultralytics LetterBox."""
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError('letterbox expects an HxWx3 uint8 image')
    h, w = image.shape[:2]
    gain = min(new_shape[0] / h, new_shape[1] / w)
    new_w, new_h = round(w * gain), round(h * gain)
    dw, dh = (new_shape[1] - new_w) / 2, (new_shape[0] - new_h) / 2
    if (w, h) != (new_w, new_h):
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    padded = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                value=(color, color, color))
    return padded, Letterbox(gain, left, top, new_w, new_h)


def scale_boxes_to_original(boxes_xyxy, input_shape, original_shape):
    """Map letterbox-space xyxy boxes to the original image (Ultralytics scale_boxes + clip)."""
    h0, w0 = original_shape[:2]
    gain = min(input_shape[0] / h0, input_shape[1] / w0)
    pad_x = round((input_shape[1] - round(w0 * gain)) / 2 - 0.1)
    pad_y = round((input_shape[0] - round(h0 * gain)) / 2 - 0.1)
    boxes = np.asarray(boxes_xyxy, dtype=np.float32).copy().reshape(-1, 4)
    boxes[:, [0, 2]] = ((boxes[:, [0, 2]] - pad_x) / gain).clip(0, w0)
    boxes[:, [1, 3]] = ((boxes[:, [1, 3]] - pad_y) / gain).clip(0, h0)
    return boxes


def decode_detections(output0, conf_threshold, iou_threshold=0.7, classes=None, max_det=300,
                      num_masks=32, max_nms=30000):
    """Decode one image's raw head output (4 + nc + nm, N).

    Returns (xyxy boxes in letterbox pixels, scores, class ids, mask coefficients),
    sorted by descending score after class-aware NMS.
    """
    pred = np.asarray(output0, dtype=np.float32)
    if pred.ndim == 3:
        if pred.shape[0] != 1:
            raise ValueError('decode_detections handles one image at a time')
        pred = pred[0]
    num_classes = pred.shape[0] - 4 - num_masks
    if num_classes < 1:
        raise ValueError(f'Output has {pred.shape[0]} channels; too few for {num_masks} mask coefficients')
    pred = pred.T                                        # (N, 4 + nc + nm)
    class_scores = pred[:, 4:4 + num_classes]
    class_ids = class_scores.argmax(axis=1)
    scores = class_scores[np.arange(len(pred)), class_ids]
    keep = scores > conf_threshold
    if classes is not None:
        keep &= np.isin(class_ids, list(classes))
    pred, scores, class_ids = pred[keep], scores[keep], class_ids[keep]
    empty = (np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, int),
             np.zeros((0, num_masks), np.float32))
    if not len(pred):
        return empty
    if len(pred) > max_nms:
        top = np.argsort(-scores, kind='stable')[:max_nms]
        pred, scores, class_ids = pred[top], scores[top], class_ids[top]
    cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    xywh = np.stack([xyxy[:, 0], xyxy[:, 1], w, h], axis=1)
    indices = cv2.dnn.NMSBoxesBatched(xywh.tolist(), scores.tolist(), class_ids.tolist(),
                                      float(conf_threshold), float(iou_threshold))
    indices = np.asarray(indices, dtype=int).reshape(-1)
    indices = indices[np.argsort(-scores[indices], kind='stable')][:max_det]
    return (xyxy[indices], scores[indices], class_ids[indices],
            pred[indices, 4 + num_classes:])


def reconstruct_masks(prototypes, coefficients, boxes_original, input_shape, original_shape):
    """Binary {0,1} masks at original resolution (Ultralytics process_mask_native)."""
    protos = np.asarray(prototypes, dtype=np.float32)
    if protos.ndim == 4:
        protos = protos[0]
    nm, mh, mw = protos.shape
    h0, w0 = original_shape[:2]
    if not len(coefficients):
        return np.zeros((0, h0, w0), np.uint8)
    # cv2.gemm, not NumPy @: NumPy's OpenBLAS pool keeps spinning after the call
    # and starves the next cv2.dnn forward (measured 20 ms -> 70 ms on 8 cores).
    logits = cv2.gemm(np.ascontiguousarray(coefficients, np.float32),
                      np.ascontiguousarray(protos.reshape(nm, -1)), 1.0, None, 0.0).reshape(-1, mh, mw)
    gain = min(mh / h0, mw / w0)
    pad_w, pad_h = (mw - round(w0 * gain)) / 2, (mh - round(h0 * gain)) / 2
    top, left = round(pad_h - 0.1), round(pad_w - 0.1)
    bottom, right = top + round(h0 * gain), left + round(w0 * gain)
    rows = np.arange(h0, dtype=np.float32)[:, None]
    cols = np.arange(w0, dtype=np.float32)[None, :]
    masks = np.empty((len(logits), h0, w0), np.uint8)
    for i, (logit, box) in enumerate(zip(logits, np.asarray(boxes_original, np.float32))):
        upsampled = cv2.resize(np.ascontiguousarray(logit[top:bottom, left:right]), (w0, h0),
                               interpolation=cv2.INTER_LINEAR)
        inside = (cols >= box[0]) & (cols < box[2]) & (rows >= box[1]) & (rows < box[3])
        masks[i] = (upsampled > 0) & inside
    return masks


class YoloSegDnn:
    """OpenCV 5 cv2.dnn YOLO11-seg runner with the FYP adapter's selection rule."""

    def __init__(self, onnx_path, input_shape=None, conf_threshold=0.35, iou_threshold=0.7,
                 max_det=300, num_masks=32, engine='new', threads=None):
        if not 0 < conf_threshold <= 1:
            raise ValueError('conf_threshold must be in (0, 1]')
        if engine not in ENGINES:
            raise ValueError(f'engine must be one of {sorted(ENGINES)}')
        if threads is not None:
            cv2.setNumThreads(int(threads))
        self.net = cv2.dnn.readNetFromONNX(str(onnx_path), engine=getattr(cv2.dnn, ENGINES[engine]))
        self.output_names = list(self.net.getUnconnectedOutLayersNames())
        if input_shape is None:
            # The export script writes the static input size next to the graph.
            sidecar = Path(onnx_path).with_suffix('.json')
            input_shape = json.loads(sidecar.read_text())['imgsz'] if sidecar.exists() else (640, 640)
        self.input_shape = tuple(int(v) for v in input_shape)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_det = max_det
        self.num_masks = num_masks
        self.engine = engine

    def forward(self, rgb):
        padded, _ = letterbox(rgb, self.input_shape)
        # The ONNX graph expects RGB in [0, 1], NCHW; the input is already RGB.
        blob = cv2.dnn.blobFromImage(padded, scalefactor=1 / 255.0, swapRB=False, crop=False)
        self.net.setInput(blob)
        outputs = dict(zip(self.output_names, self.net.forward(self.output_names)))
        detection = next(o for o in outputs.values() if o.ndim == 3)
        prototypes = next(o for o in outputs.values() if o.ndim == 4)
        return detection, prototypes

    def predict(self, rgb, classes=None):
        timings = {}
        started = time.perf_counter()
        detection, prototypes = self.forward(rgb)
        timings['forward_ms'] = 1000 * (time.perf_counter() - started)
        post = time.perf_counter()
        boxes, scores, class_ids, coefficients = decode_detections(
            detection, self.conf_threshold, self.iou_threshold, classes, self.max_det, self.num_masks)
        boxes = scale_boxes_to_original(boxes, self.input_shape, rgb.shape)
        masks = reconstruct_masks(prototypes, coefficients, boxes, self.input_shape, rgb.shape)
        # Ultralytics drops detections whose cropped mask is empty.
        detections = [Detection(b, float(s), int(c), m)
                      for b, s, c, m in zip(boxes, scores, class_ids, masks) if m.any()]
        timings['postprocess_ms'] = 1000 * (time.perf_counter() - post)
        timings['total_ms'] = 1000 * (time.perf_counter() - started)
        return Prediction(detections, timings)

    def best_mask(self, rgb, class_id=0):
        """Highest-scoring instance of class_id as a 0/255 mask, like YoloMaskProvider."""
        prediction = self.predict(rgb, classes=[class_id])
        mask = np.zeros(rgb.shape[:2], np.uint8)
        candidates = [d for d in prediction.detections
                      if d.class_id == class_id and d.score >= self.conf_threshold]
        best = max(candidates, key=lambda d: d.score, default=None)
        if best is not None:
            mask[best.mask != 0] = 255
        return mask, (best.score if best is not None else None), prediction
