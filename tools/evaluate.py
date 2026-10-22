"""Run an evaluation job locally (OpenCV 5 env) and write report.json + summary.md.

    .venv-cv5/bin/python tools/evaluate.py --job-id fyp-headless-sweep \\
        --bundle build/bundles/headless_sweep.npz --mask-sources dnn,gt,hsv \\
        --model build/cube_yolo11n_seg.onnx \\
        --recorded $GALBOT_FYP_WS/validation_runs/live_timing_fix2_20261015_002735/*/yaw* \\
        --output results/eval_fyp_headless_sweep

or ``--job path/to/job.json`` with the same schema the Lambda handler reads.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from galbot_cv5.report import render_markdown, run_job  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--job', type=Path)
    parser.add_argument('--job-id')
    parser.add_argument('--bundle', action='append', default=[])
    parser.add_argument('--mask-sources', default='hsv,gt')
    parser.add_argument('--model')
    parser.add_argument('--recorded', nargs='*', default=[])
    parser.add_argument('--threads', type=int)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    if args.threads:
        import cv2
        cv2.setNumThreads(args.threads)
    if args.job:
        job = json.loads(args.job.read_text())
    else:
        job = dict(job_id=args.job_id, bundles=args.bundle, mask_sources=args.mask_sources.split(','),
                   recorded_trials=[str(p) for p in args.recorded])
        if args.model:
            job['model'] = args.model
    base = args.job.parent if args.job else Path.cwd()

    def fetch(key):
        path = Path(key).expanduser()
        return path if path.is_absolute() else base / path

    report = run_job(job, fetch, allow_recorded=True)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    (args.output / 'summary.md').write_text(render_markdown(report))
    print(render_markdown(report))


if __name__ == '__main__':
    main()
