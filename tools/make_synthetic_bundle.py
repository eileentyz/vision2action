"""Write the procedural synthetic stereo bundle (no project data) and a matching job file.

    .venv-cv5/bin/python tools/make_synthetic_bundle.py --output samples/synthetic
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from galbot_cv5.bundle import save_bundle  # noqa: E402
from galbot_cv5.synthetic import make_samples  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--output', type=Path, default=HERE / 'samples/synthetic')
    parser.add_argument('--count', type=int, default=12)
    parser.add_argument('--seed', type=int, default=20261022)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    bundle = args.output / 'synthetic_stereo.npz'
    save_bundle(bundle, make_samples(args.count, args.seed), dict(
        source='synthetic', seed=args.seed,
        description='Procedural OpenCV-drawn stereo views of a red 30x30x60 mm box (no project data)'))
    job = dict(job_id='synthetic-smoke', bundles=[bundle.name], mask_sources=['hsv', 'gt'])
    (args.output / 'job.json').write_text(json.dumps(job, indent=2) + '\n')
    print(f'wrote {bundle} ({bundle.stat().st_size / 1e6:.2f} MB) and {args.output / "job.json"}')


if __name__ == '__main__':
    main()
