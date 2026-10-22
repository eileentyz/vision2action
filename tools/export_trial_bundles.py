"""Re-render recorded FYP trials into head-stereo evaluation bundles (export env, ROS sourced).

For each trial (validate_pick_place_mission output: recording.jsonl, world.xml,
events/), the scene is rebuilt with the FYP replay tool's Scene class, from the
recorded joints, base pose and simulator cube pose, at evenly spaced times while
the live head detector ran (cube_handoff HEAD_SEARCH -> HANDED_OFF). Head left/right
RGB, MuJoCo geom-segmentation ground-truth masks, intrinsics, extrinsics (as TF
gives the live detector) and the true cube pose in base_link are stored.

The recording has no camera images: these are re-rendered views of the recorded
state, not the frames the live detector saw. Output stays local (build/).

    source /opt/ros/jazzy/setup.bash
    .venv-export/bin/python tools/export_trial_bundles.py --output build/bundles/headless_sweep.npz \\
        $GALBOT_FYP_WS/validation_runs/live_timing_fix2_20261015_002735/headless_sweep/yaw*
"""
import argparse
import math
import os
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parents[1]
# Needs the private FYP checkout (GALBOT_FYP_REPO): ROS 2 package, recordings, checkpoint.
REPO = Path(os.environ['GALBOT_FYP_REPO']) if 'GALBOT_FYP_REPO' in os.environ else HERE.parent / 'galbot_mobile_manipulation'
sys.path.insert(0, str(REPO / 'packages/galbot_mobile_manipulation'))
sys.path.insert(0, str(REPO / 'tools'))
sys.path.insert(0, str(HERE))

from galbot_cv5.bundle import save_bundle  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('trials', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--per-trial', type=int, default=3, help='samples per trial in the head window')
    args = parser.parse_args()

    import replay_mission_cameras as replay   # FYP tool: Scene, frames, head_yaw_window, CENTRE_Z, HALF

    samples, used, skipped = [], [], {}
    for trial in args.trials:
        if not (trial / 'recording.jsonl').exists() or not (trial / 'world.xml').exists():
            skipped[str(trial)] = 'no recording.jsonl/world.xml'
            continue
        search, handed = replay.head_yaw_window(trial)
        if search is None or handed is None:
            skipped[str(trial)] = 'no HEAD_SEARCH -> HANDED_OFF window in events'
            continue
        states = [(t, s) for t, s in replay.frames(trial) if search <= t <= handed]
        if not states:
            skipped[str(trial)] = 'no recorded state inside the head window'
            continue
        scene = replay.Scene(trial)
        cube_geom = mujoco.mj_name2id(scene.m, mujoco.mjtObj.mjOBJ_GEOM, 'pick_cube_geom')
        if cube_geom < 0:
            skipped[str(trial)] = 'world has no pick_cube_geom'
            continue
        segmenter = mujoco.Renderer(scene.m, replay.H, replay.W)
        segmenter.enable_segmentation_rendering()
        picks = np.unique(np.linspace(0, len(states) - 1, args.per_trial).round().astype(int))
        for k, index in enumerate(picks):
            t, state = states[index]
            scene.apply(state)
            images, masks = [], []
            for camera in ('head_left_camera', 'head_right_camera'):
                images.append(scene.render(camera))
                segmenter.update_scene(scene.d, camera=scene.cams[camera])
                seg = segmenter.render()
                masks.append(((seg[..., 0] == cube_geom) & (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM))
                             .astype(np.uint8) * 255)
            extrinsics = scene.head_extrinsics()
            truth = state['truth']
            samples.append(dict(
                id=f'{trial.parent.name}/{trial.name}#{k}', group=trial.name,
                left=images[0], right=images[1], gt_left=masks[0], gt_right=masks[1], has_gt=True,
                K=scene.K, R=np.stack([r for r, _ in extrinsics]), o=np.stack([o for _, o in extrinsics]),
                truth_p=np.asarray(truth['p'], float), truth_yaw_deg=math.degrees(replay.yaw_of(truth['q'])),
                centre_z=float(replay.CENTRE_Z), half=np.asarray(replay.HALF, float)))
        segmenter.close()
        scene.renderer.close()
        used.append(str(trial))
        print(f'{trial}: {len(picks)} samples', flush=True)

    if not samples:
        raise SystemExit(f'No usable trials: {skipped}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_bundle(args.output, samples, dict(
        source='fyp_replay', description='Re-rendered head stereo views of recorded FYP trials',
        trials=used, skipped=skipped, per_trial=args.per_trial, mujoco=mujoco.__version__))
    print(f'wrote {args.output}: {len(samples)} samples from {len(used)} trials; skipped {len(skipped)}')


if __name__ == '__main__':
    main()
