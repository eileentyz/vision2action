"""Make upload copies of bundles and the ONNX model with local paths removed (export env).

Bundles: absolute trial paths in the metadata become run/trial names; pixel and
ground-truth arrays are copied unchanged. ONNX: the Ultralytics ``description``
(which embeds the local dataset path) is replaced; weights, graph and the other
metadata (including the AGPL-3.0 ``license`` entry) are kept. A report of what
changed and the SHA-256 of every output is written to build/upload/manifest.json.

    .venv-export/bin/python tools/sanitize_for_upload.py \\
        --bundle build/bundles/headless_sweep.npz --onnx build/cube_yolo11n_seg.onnx
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from galbot_cv5.bundle import load_bundle, save_bundle  # noqa: E402

HOME_MARKERS = ('/home/', '/Users/', 'C:\\Users\\')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def strip_paths(value):
    if isinstance(value, str) and any(m in value for m in HOME_MARKERS):
        parts = Path(value).parts
        return '/'.join(parts[-2:])
    if isinstance(value, list):
        return [strip_paths(v) for v in value]
    if isinstance(value, dict):
        return {k: strip_paths(v) for k, v in value.items()}
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--bundle', type=Path, action='append', default=[])
    parser.add_argument('--onnx', type=Path)
    parser.add_argument('--output', type=Path, default=HERE / 'build/upload')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = dict(bundles=[], model=None)

    for path in args.bundle:
        samples, meta = load_bundle(path)
        clean = strip_paths({k: v for k, v in meta.items() if k not in ('format', 'samples')})
        out = args.output / path.name
        save_bundle(out, samples, clean)
        reloaded, _ = load_bundle(out)
        for a, b in zip(samples, reloaded):
            for field in ('left', 'right', 'gt_left', 'gt_right', 'K', 'R', 'o', 'truth_p'):
                if not np.array_equal(a[field], b[field]):
                    raise SystemExit(f'{out}: {field} changed while sanitising')
        manifest['bundles'].append(dict(source=path.name, output=out.name, sha256=sha256(out),
                                        samples=len(samples), meta=clean))

    if args.onnx:
        import onnx
        model = onnx.load(str(args.onnx))
        changed = []
        for prop in model.metadata_props:
            if any(m in prop.value for m in HOME_MARKERS):
                changed.append(prop.key)
                prop.value = 'Ultralytics YOLO11n-seg fine-tuned on synthetic MuJoCo cube images (Galbot FYP)'
        out = args.output / args.onnx.name
        onnx.save(model, str(out))
        weights_before = hashlib.sha256(b''.join(t.raw_data for t in onnx.load(str(args.onnx)).graph.initializer))
        weights_after = hashlib.sha256(b''.join(t.raw_data for t in onnx.load(str(out)).graph.initializer))
        if weights_before.hexdigest() != weights_after.hexdigest():
            raise SystemExit('ONNX weights changed while sanitising')
        sidecar = json.loads(args.onnx.with_suffix('.json').read_text())
        sidecar = strip_paths(dict(sidecar, onnx_sha256=sha256(out), sanitised_from_sha256=sha256(args.onnx)))
        out.with_suffix('.json').write_text(json.dumps(sidecar, indent=2) + '\n')
        manifest['model'] = dict(output=out.name, sha256=sha256(out), metadata_replaced=changed,
                                 weights_sha256=weights_after.hexdigest(),
                                 license=next((p.value for p in model.metadata_props if p.key == 'license'), None))
        shutil.copymode(args.onnx, out)

    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    leftovers = [p.name for p in args.output.iterdir()
                 if p.is_file() and any(m.encode() in p.read_bytes() for m in HOME_MARKERS)]
    print(json.dumps(dict(manifest, files_still_containing_home_paths=leftovers), indent=2))
    if leftovers:
        raise SystemExit(f'Home paths remain in {leftovers}')


if __name__ == '__main__':
    main()
