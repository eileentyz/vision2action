"""Procedural head-stereo scenes of a red upright box, drawn with OpenCV only.

No project data: a generic rectified pinhole pair looks down at a table, and a
30 x 30 x 60 mm box with known pose is projected with cv2.projectPoints and
rasterised face by face (back-face culled, Lambertian shading) over a noisy
table texture. Ground truth (masks, centre, yaw) is exact by construction, so
the bundle exercises the full evaluation (segmentation, stereo triangulation,
silhouette yaw fit, reports) for integration tests and the AWS smoke test.
"""

import math

import cv2
import numpy as np

WIDTH, HEIGHT = 640, 480
FOVY_DEG = 60.0
BASELINE_M = 0.07
HALF = (0.015, 0.015, 0.03)
TABLE_Z = 0.80                       # table top in the synthetic base frame (m)
CAMERA_ORIGIN = np.array([0.20, 0.0, 1.30])
CAMERA_PITCH_DEG = 50.0              # optical axis below horizontal, looking along base +X


def intrinsics(width=WIDTH, height=HEIGHT, fovy_deg=FOVY_DEG):
    f = 0.5 * height / math.tan(math.radians(fovy_deg) / 2)
    return np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]], float)


def optical_to_base(pitch_deg=CAMERA_PITCH_DEG):
    """Rotation whose columns are the optical x (right), y (down), z (forward) axes in base."""
    p = math.radians(pitch_deg)
    z = np.array([math.cos(p), 0.0, -math.sin(p)])     # forward and down
    x = np.array([0.0, -1.0, 0.0])                      # image right = base -Y
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def rig(pitch_deg=CAMERA_PITCH_DEG, origin=CAMERA_ORIGIN, baseline=BASELINE_M):
    """(K, R (2,3,3), o (2,3)) for a rectified pair; the right camera sits at +X optical."""
    R = optical_to_base(pitch_deg)
    left = np.asarray(origin, float)
    right = left + baseline * R[:, 0]
    return intrinsics(), np.stack([R, R]), np.stack([left, right])


def box_faces(centre, yaw_deg, half=HALF):
    """Eight corners and six faces (corner indices, outward normal) of an upright box in base."""
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
    corners = (signs * half) @ rotation.T + centre
    faces = []
    for axis in range(3):
        for sign in (-1, 1):
            index = [i for i in range(8) if signs[i, axis] == sign]
            quad = [index[0], index[1], index[3], index[2]]      # perimeter order
            normal = rotation[:, axis] * sign
            faces.append((quad, normal))
    return corners, faces


def project(points_base, K, R, o):
    """Pixels of base-frame points with cv2.projectPoints (world = base, camera = optical)."""
    rotation = R.T                                   # base -> optical
    rvec, _ = cv2.Rodrigues(rotation)
    tvec = -rotation @ o
    pixels, _ = cv2.projectPoints(np.asarray(points_base, np.float64).reshape(-1, 1, 3), rvec, tvec, K, None)
    return pixels.reshape(-1, 2)


def table_texture(rng, shape=(HEIGHT, WIDTH)):
    base = rng.integers(150, 190)
    noise = cv2.GaussianBlur(rng.normal(0, 18, shape).astype(np.float32), (0, 0), 3)
    gradient = np.linspace(-12, 12, shape[1], dtype=np.float32)[None, :]
    grey = np.clip(base + noise + gradient, 0, 255).astype(np.uint8)
    return cv2.cvtColor(grey, cv2.COLOR_GRAY2RGB)


def render(centre, yaw_deg, K, R, o, rng, colour=(200, 25, 25), half=HALF):
    """(RGB image, 0/255 silhouette mask) of the box seen by one camera."""
    image = table_texture(rng)
    mask = np.zeros(image.shape[:2], np.uint8)
    corners, faces = box_faces(centre, yaw_deg, half)
    pixels = project(corners, K, R, o)
    light = np.array([0.3, 0.2, 1.0]) / np.linalg.norm([0.3, 0.2, 1.0])
    visible = []
    for quad, normal in faces:
        centroid = corners[quad].mean(axis=0)
        if np.dot(normal, o - centroid) <= 0:          # back face
            continue
        depth = float(np.linalg.norm(centroid - o))
        visible.append((depth, quad, normal))
    for _, quad, normal in sorted(visible, key=lambda v: -v[0]):  # far to near
        shade = 0.45 + 0.55 * max(0.0, float(np.dot(normal, light)))
        polygon = np.round(pixels[quad] * 16).astype(np.int32)
        cv2.fillConvexPoly(image, polygon, tuple(int(c * shade) for c in colour), cv2.LINE_AA, shift=4)
        cv2.fillConvexPoly(mask, polygon, 255, cv2.LINE_8, shift=4)
    noisy = image.astype(np.float32) + rng.normal(0, 3.0, image.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8), mask


def make_samples(count=12, seed=20261022):
    """Deterministic list of bundle samples with yaw in [-45, 45) and varied table position."""
    rng = np.random.default_rng(seed)
    K, R, o = rig()
    samples = []
    for i in range(count):
        yaw = float(rng.uniform(-45, 45))
        centre = np.array([rng.uniform(0.62, 0.74), rng.uniform(-0.06, 0.06), TABLE_Z + HALF[2]])
        left, gt_left = render(centre, yaw, K, R[0], o[0], rng)
        right, gt_right = render(centre, yaw, K, R[1], o[1], rng)
        samples.append(dict(id=f'synthetic#{i:03d}', group='synthetic', left=left, right=right,
                            gt_left=gt_left, gt_right=gt_right, has_gt=True, K=K, R=R, o=o,
                            truth_p=centre, truth_yaw_deg=yaw, centre_z=float(centre[2]),
                            half=np.asarray(HALF, float)))
    return samples
