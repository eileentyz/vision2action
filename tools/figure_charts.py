"""Publication charts and the architecture diagram, from the committed reports only.

Reads results/eval_fyp_headless_sweep (local) and results/cloud/* (AWS Lambda) and writes
PNG + SVG into docs/figures. The cloud evidence panel is rendered from CLI and
CloudWatch output with the account ID replaced by a placeholder. Palette: categorical
slots 1-3 of the dataviz reference palette, validated all-pairs (light).

    .venv-export/bin/python tools/figure_charts.py
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
SURFACE, INK, INK2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#8a8984', '#e4e3df'
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
SOURCES = (('dnn', 'OpenCV 5 cv2.dnn', BLUE), ('gt', 'ground-truth mask', ORANGE), ('hsv', 'HSV threshold', AQUA))

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11, 'text.color': INK, 'axes.labelcolor': INK2,
    'axes.edgecolor': GRID, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.facecolor': SURFACE,
    'figure.facecolor': SURFACE, 'savefig.facecolor': SURFACE, 'axes.spines.top': False,
    'axes.spines.right': False, 'svg.fonttype': 'none'})


def save(fig, out, name):
    for ext in ('png', 'svg'):
        fig.savefig(out / f'{name}.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)


def results(path):
    return json.loads(Path(path).read_text())['bundles'][0]['results']


def accuracy(cloud, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw=dict(wspace=0.35))
    panels = (('yaw_error_deg', 'abs yaw error (deg)', lambda v: abs(v)),
              ('fit_xy_error_mm', 'fitted XY position error (mm)', lambda v: v))
    for ax, (field, label, f) in zip(axes, panels):
        for row, (key, name, color) in enumerate(SOURCES):
            values = sorted(f(s[field]) for s in cloud[key]['samples'] if s['pose_ok'])
            jitter = [(i % 5 - 2) * 0.06 for i in range(len(values))]
            ax.scatter(values, [row + j for j in jitter], s=34, color=color, edgecolor=SURFACE,
                       linewidth=1.2, zorder=3)
            median = values[len(values) // 2]
            ax.plot([median, median], [row - 0.3, row + 0.3], color=INK, linewidth=2, zorder=4)
            ax.annotate(f'median {median:.2f}', (median, row + 0.32), color=INK2, fontsize=9,
                        ha='left', va='bottom', xytext=(3, 0), textcoords='offset points')
        ax.set_yticks(range(len(SOURCES)), [n for _, n, _ in SOURCES])
        ax.set_ylim(len(SOURCES) - 0.4, -0.6)
        ax.set_xlabel(label)
        ax.set_xlim(left=-0.03 * max(abs(s[field]) for k, _, _ in SOURCES
                                      for s in cloud[k]['samples'] if s['pose_ok']))
        ax.grid(axis='x', color=GRID, linewidth=0.8)
        ax.tick_params(axis='y', length=0)
        ax.spines['left'].set_visible(False)
    axes[1].set_yticklabels([])
    fig.suptitle('Pose accuracy by mask source: 21 re-rendered FYP head stereo pairs, measured on AWS Lambda',
                 x=0.06, ha='left', fontsize=12)
    fig.text(0.06, -0.07, 'Each dot is one stereo pair; black tick = median. Identical values were measured '
             'locally (differences below 1e-12).', color=MUTED, fontsize=9)
    save(fig, out, 'fig_accuracy_by_mask_source')


def latency(local, cloud, out):
    rows = (('segmentation_ms', 'cv2.dnn segmentation\n(one stereo pair = 2 images)'),
            ('pose_ms', 'stereo triangulation +\nsilhouette yaw fit (one pair)'))
    series = (('local workstation (8 OpenCV threads)', local, BLUE),
              ('AWS Lambda, 3008 MB (2 OpenCV threads)', cloud, ORANGE))
    fig, ax = plt.subplots(figsize=(9, 3.2))
    height = 0.34
    for i, (field, _) in enumerate(rows):
        for j, (name, data, color) in enumerate(series):
            value = data['dnn']['summary'][field]['median']
            y = i + (j - 0.5) * (height + 0.04)
            ax.barh(y, value, height=height, color=color, label=name if i == 0 else None, zorder=3)
            ax.annotate(f'{value:.0f} ms', (value, y), xytext=(4, 0), textcoords='offset points',
                        va='center', fontsize=10, color=INK)
    ax.set_yticks(range(len(rows)), [label for _, label in rows])
    ax.invert_yaxis()
    ax.set_xlabel('median time per stereo pair (ms)')
    ax.grid(axis='x', color=GRID, linewidth=0.8)
    ax.tick_params(axis='y', length=0)
    ax.spines['left'].set_visible(False)
    ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=9)
    ax.set_title('Same code, same 21 stereo pairs: local vs AWS Lambda latency', loc='left', fontsize=12)
    fig.text(0.0, -0.2, 'Local: AMD Ryzen 7 8845HS laptop. Lambda: warm medians; the first sample of a cold '
             'start took 1.4 s. Accuracy was identical in both.', color=MUTED, fontsize=9)
    save(fig, out, 'fig_latency_local_vs_lambda')


def box(ax, xy, w, h, title, lines=(), edge=INK2, fill='#ffffff', title_size=10.5):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle='round,pad=0.02,rounding_size=0.12', linewidth=1.3,
                                edgecolor=edge, facecolor=fill, zorder=2))
    ax.text(xy[0] + 0.15, xy[1] + h - 0.22, title, fontsize=title_size, weight='bold', va='top', zorder=3)
    for k, line in enumerate(lines):
        ax.text(xy[0] + 0.15, xy[1] + h - 0.62 - 0.33 * k, line, fontsize=9, color=INK2, va='top', zorder=3)


def zone(ax, xy, w, h, label, color):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle='round,pad=0.02,rounding_size=0.2', linewidth=1.6,
                                edgecolor=color, facecolor='none', linestyle=(0, (5, 3)), zorder=1))
    ax.text(xy[0] + 0.2, xy[1] + h + 0.12, label, fontsize=11.5, weight='bold', color=color, va='bottom')


def arrow(ax, a, b, text=None, offset=(0, 0.14), rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=14, linewidth=1.4, color=INK2,
                                 connectionstyle=f'arc3,rad={rad}', zorder=4))
    if text:
        ax.text((a[0] + b[0]) / 2 + offset[0], (a[1] + b[1]) / 2 + offset[1], text, fontsize=8.5,
                color=INK2, ha='center', va='bottom', zorder=5,
                bbox=dict(boxstyle='round,pad=0.15', facecolor=SURFACE, edgecolor='none'))


def architecture(out):
    fig, ax = plt.subplots(figsize=(16, 9.4))
    ax.set_xlim(0, 16)
    ax.set_ylim(-0.1, 9.3)
    ax.axis('off')

    zone(ax, (0.3, 4.75), 6.9, 3.95, 'FYP robotic system (ROS 2 Jazzy + MuJoCo; unchanged)', INK2)
    box(ax, (0.6, 6.85), 3.0, 1.5, 'Galbot One Golf (sim)', ('MuJoCo, head stereo +', 'wrist cameras, Nav2,', 'MoveIt 2'))
    box(ax, (4.0, 6.85), 2.9, 1.5, 'YoloMaskProvider', ('YOLO11n-seg via', 'Ultralytics + PyTorch', '(live pipeline)'))
    box(ax, (0.6, 5.05), 3.0, 1.45, 'Recorded trials', ('recording.jsonl, world.xml', '(joints, base pose,', 'cube pose)'))
    box(ax, (4.0, 5.05), 2.9, 1.45, 'FYP geometry', ('head_localisation: stereo', 'head_yaw: silhouette fit', '-> pick pose'))
    arrow(ax, (3.6, 7.6), (4.0, 7.6))
    arrow(ax, (5.45, 6.85), (5.45, 6.5))
    arrow(ax, (2.1, 6.85), (2.1, 6.5), 'logged', offset=(0.55, -0.13))

    zone(ax, (7.6, 4.75), 8.1, 3.95, 'Competition: OpenCV 5 on the local workstation', BLUE)
    box(ax, (7.9, 6.85), 2.3, 1.5, 'ONNX export', ('same checkpoint,', 'static 480x640,', 'opset 17'), edge=BLUE)
    box(ax, (10.5, 6.85), 2.7, 1.5, 'OpenCV 5 cv2.dnn', ('letterbox, NMSBoxesBatched,', 'cv2.gemm masks,', 'no torch'), edge=BLUE)
    box(ax, (13.5, 6.85), 2.0, 1.5, 'Parity check', ('150/150 decisions,', '0 mask pixels', 'differ'), edge=BLUE)
    box(ax, (7.9, 5.05), 3.4, 1.45, 'Bundle export', ('MuJoCo re-render of states:', '21 stereo pairs + GT masks,', 'synthetic set (12 pairs)'), edge=BLUE)
    box(ax, (11.7, 5.05), 3.8, 1.45, 'evaluate.py (OpenCV 5)', ('mask IoU, centroid, stereo XY,', 'yaw; FYP geometry reused', 'unchanged'), edge=BLUE)
    arrow(ax, (10.2, 7.6), (10.5, 7.6))
    arrow(ax, (13.2, 7.6), (13.5, 7.6))
    arrow(ax, (11.3, 5.8), (11.7, 5.8))
    arrow(ax, (12.4, 6.85), (12.9, 6.5))
    arrow(ax, (6.9, 7.95), (7.9, 7.95), 'checkpoint', offset=(0, 0.08))
    arrow(ax, (2.1, 5.05), (7.9, 5.45), 'recorded states', offset=(-0.3, -0.75), rad=0.22)

    zone(ax, (0.3, 0.3), 15.4, 3.6, 'AWS (private, Sydney, Free Plan)', ORANGE)
    box(ax, (0.6, 0.8), 3.3, 2.6, 'Upload (owner-approved)', ('sanitize_for_upload.py:', 'no local paths, SHA-256', 'manifest; submit_job.sh:', 'synthetic by default'), edge=ORANGE)
    box(ax, (4.3, 0.8), 2.9, 2.6, 'S3 bucket', ('inputs/  models/  jobs/', 'SSE-S3, no public', 'access, TLS only,', '30-day expiry'), edge=ORANGE)
    box(ax, (8.2, 0.8), 3.2, 2.6, 'Lambda (container)', ('OpenCV 5 + NumPy 2,', '3008 MB, 600 s;', 'reads inputs/ models/,', 'writes reports/ only'), edge=ORANGE)
    box(ax, (11.8, 2.3), 3.6, 1.1, 'S3 reports/<job>/', ('report.json, summary.md',), edge=ORANGE)
    box(ax, (11.8, 0.8), 1.75, 1.25, 'ECR', ('image, private,', 'scan on push'), edge=ORANGE, title_size=10)
    box(ax, (13.65, 0.8), 1.75, 1.25, 'CloudWatch', ('logs + alarm,', 'Budgets USD 5'), edge=ORANGE, title_size=10)
    arrow(ax, (3.9, 2.1), (4.3, 2.1))
    arrow(ax, (7.2, 2.1), (8.2, 2.1), 'trigger', offset=(0, 0.08))
    arrow(ax, (11.4, 2.85), (11.8, 2.85))
    arrow(ax, (11.8, 1.4), (11.4, 1.4))
    arrow(ax, (9.9, 5.05), (3.0, 3.4), 'bundles + ONNX', offset=(1.9, 0.0), rad=0.04)
    arrow(ax, (13.6, 3.4), (13.6, 5.05), 'compare: identical accuracy', offset=(-0.2, -0.05))
    save(fig, out, 'fig_architecture')


def live_architecture(out):
    """Live branch: the robot's masks come from OpenCV 5, the report is recomputed on AWS."""
    fig, ax = plt.subplots(figsize=(16, 8.9))
    ax.set_xlim(0, 16)
    ax.set_ylim(-0.1, 8.8)
    ax.axis('off')

    zone(ax, (0.3, 4.35), 7.1, 3.95, 'Robot: FYP mission (ROS 2 Jazzy + MuJoCo, NumPy 1)', INK2)
    box(ax, (0.6, 6.55), 3.1, 1.45, '1  Galbot One Golf (sim)', ('head stereo + wrist cameras,', 'Nav2, MoveIt 2'))
    box(ax, (4.0, 6.55), 3.1, 1.45, '2  object_pointcloud node', ('yolo_backend:=opencv5', '(OpenCV5MaskProvider)'))
    box(ax, (4.0, 4.65), 3.1, 1.45, '4  FYP geometry + mission', ('stereo point + head yaw fit', '-> pick -> carry -> place'))
    box(ax, (0.6, 4.65), 3.1, 1.45, '5  Trial record', ('recording.jsonl, events/', 'per frame: backend, ms'))
    arrow(ax, (3.7, 7.45), (4.0, 7.45))
    arrow(ax, (5.55, 6.55), (5.55, 6.1))
    arrow(ax, (4.0, 5.35), (3.7, 5.35))

    zone(ax, (7.8, 4.35), 7.9, 3.95, 'OpenCV 5 (.venv-cv5, NumPy 2) on the same laptop', BLUE)
    box(ax, (8.6, 6.55), 3.3, 1.45, '3  opencv5_mask_server.py', ('cv2.dnn YOLO11n-seg (ONNX),', 'best cube -> mask'), edge=BLUE)
    box(ax, (12.3, 6.55), 3.1, 1.45, 'Parity with Ultralytics', ('150/150 detections agree,', '0 mask pixels differ'), edge=BLUE)
    box(ax, (8.6, 4.65), 3.3, 1.45, '6  Export (local)', ('re-rendered stereo bundle,', 'live_missions.json (poses)'), edge=BLUE)
    box(ax, (12.3, 4.65), 3.1, 1.45, '7  Local report', ('evaluate.py (OpenCV 5)', 'results/live/<run>/local'), edge=BLUE)
    arrow(ax, (7.1, 7.55), (8.6, 7.55), 'rgb8 frame', offset=(0, 0.06))
    arrow(ax, (8.6, 7.0), (7.1, 7.0), 'mask + score', offset=(0, -0.42))
    arrow(ax, (11.9, 7.25), (12.3, 7.25))
    arrow(ax, (11.9, 5.35), (12.3, 5.35))
    arrow(ax, (2.15, 4.65), (8.6, 4.9), 'after the mission', offset=(-1.7, -0.55), rad=0.2)

    zone(ax, (0.3, 0.3), 15.4, 3.35, 'AWS (private project, ap-southeast-2)', ORANGE)
    box(ax, (0.6, 0.65), 3.3, 2.45, '8  submit_job.sh', ('uploads the sanitised', 'bundle, live_missions.json', 'and ONNX (owner-approved)'), edge=ORANGE)
    box(ax, (4.3, 0.65), 3.1, 2.45, 'S3 bucket', ('inputs/  models/  jobs/', 'private, SSE-S3,', 'TLS only, 30-day expiry'), edge=ORANGE)
    box(ax, (8.6, 0.65), 3.3, 2.45, '9  Lambda (OpenCV 5)', ('container image from ECR;', 're-runs cv2.dnn on the', 'bundle, scores the mission'), edge=ORANGE)
    box(ax, (12.3, 0.65), 3.1, 2.45, '10  S3 reports/<job>/', ('summary.md, report.json', '-> results/cloud/<job>/'), edge=ORANGE)
    arrow(ax, (3.9, 1.9), (4.3, 1.9))
    arrow(ax, (7.4, 1.9), (8.6, 1.9), 'trigger', offset=(0, 0.08))
    arrow(ax, (11.9, 1.9), (12.3, 1.9))
    arrow(ax, (9.2, 4.65), (2.6, 3.1), 'upload', offset=(1.75, 0.18), rad=0.03)
    arrow(ax, (13.85, 3.1), (13.85, 4.65), 'same accuracy', offset=(-0.85, -0.1))
    save(fig, out, 'fig_live_end_to_end')


def cloud_evidence(out):
    lines = [
        ('# Rendered from AWS CLI and CloudWatch Logs output, 2026-10-22. Account ID replaced with <account-id>.', MUTED),
        ('', INK),
        ('$ aws cloudformation describe-stacks --stack-name galbot-cv5-eval --query Stacks[0].StackStatus', INK2),
        ('CREATE_COMPLETE    (8 resources: S3 bucket + policy, Lambda + S3 permission, IAM role, logs, alarm, budget)', INK),
        ('', INK),
        ('$ aws s3 ls --recursive s3://galbot-cv5-eval-<account-id>-ap-southeast-2/reports/', INK2),
        ('2026-10-22 14:26:46      35991 reports/synthetic-smoke/report.json', INK),
        ('2026-10-22 14:26:46        874 reports/synthetic-smoke/summary.md', INK),
        ('2026-10-22 14:30:14      87778 reports/fyp-headless-sweep-dnn/report.json', INK),
        ('2026-10-22 14:30:14       1001 reports/fyp-headless-sweep-dnn/summary.md', INK),
        ('2026-10-22 14:28:48        164 reports/fyp-headless-sweep/error.json    # first attempt: model sidecar not uploaded', INK),
        ('', INK),
        ('# CloudWatch Logs /aws/lambda/galbot-cv5-eval-evaluator', INK2),
        ('REPORT  synthetic-smoke         Duration: 8365.56 ms   Billed: 11622 ms  Memory: 3008 MB  Max used: 190 MB  Init: 3256 ms', INK),
        ('REPORT  fyp-headless-sweep-dnn  Duration: 19716.13 ms  Billed: 19717 ms  Memory: 3008 MB  Max used: 298 MB', INK),
        ('', INK),
        ('# ECR image scan (opencv5-6af372c): CRITICAL 0  HIGH 0  MEDIUM 0  LOW 0  (1 informational)', INK2),
        ('# Cost Explorer, month to date: USD 0.00    Free Plan: ACTIVE', INK2),
    ]
    fig = plt.figure(figsize=(13, 0.22 * len(lines) + 0.3))
    for k, (text, color) in enumerate(lines):
        fig.text(0.015, 1 - (k + 0.9) / (len(lines) + 0.6), text, family='DejaVu Sans Mono', fontsize=9.5,
                 color=color, va='top')
    save(fig, out, 'fig_cloud_run_evidence')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--output', type=Path, default=HERE / 'docs/figures')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    local = results(HERE / 'results/eval_fyp_headless_sweep/report.json')
    cloud = results(HERE / 'results/cloud/fyp-headless-sweep-dnn/report.json')
    accuracy(cloud, args.output)
    latency(local, cloud, args.output)
    architecture(args.output)
    live_architecture(args.output)
    cloud_evidence(args.output)
    print('\n'.join(str(p.relative_to(HERE)) for p in sorted(args.output.glob('fig_*'))))


if __name__ == '__main__':
    main()
