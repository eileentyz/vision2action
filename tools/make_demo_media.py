"""Demo video and GIFs of a live OpenCV 5 mission: the robot beside what OpenCV 5 saw (FYP env).

    source /opt/ros/jazzy/setup.bash && source install/setup.bash
    GALBOT_FYP_REPO=src/galbot_mobile_manipulation .venv/bin/python ~/vision2action/tools/make_demo_media.py \\
        validation_runs/opencv5_live_<run>/yaw+30-1 \\
        --frames ~/vision2action/build/live/opencv5_live_<run>/frames

The frames come from run_live_demo.sh --record: each one is the image the robot sent to
the OpenCV 5 server, with the mask the server returned drawn on it. The left panel is the
recorded mission (joints, base, cube) re-rendered by the FYP replay renderer; both panels
share the mission clock (frame file times vs recording.jsonl). Writes to --output:
    live_mission_opencv5.mp4   side by side, for the video
    live_mission_opencv5.gif   the same, smaller and faster, for the README
    opencv5_detections.gif     the OpenCV 5 frames alone
"""
import argparse
import importlib.util
import os
from pathlib import Path
import subprocess

import cv2
import mujoco
import numpy as np

HERE = Path(__file__).resolve().parents[1]
# The replay renderer is FYP code (tools/render_mission_mp4.py): needs GALBOT_FYP_REPO.
REPO = Path(os.environ['GALBOT_FYP_REPO']) if 'GALBOT_FYP_REPO' in os.environ else HERE.parent / 'galbot_mobile_manipulation'
PANEL_W, PANEL_H, BAR, FPS = 800, 600, 44, 20


def load_renderer():
    spec = importlib.util.spec_from_file_location('render_mission_mp4', REPO / 'tools/render_mission_mp4.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def detection_frames(directory):
    """(time, path) of the served frames, oldest first; the file time is when OpenCV 5 replied."""
    files = sorted(directory.glob('frame_*.jpg'))
    return sorted((f.stat().st_mtime, f) for f in files)


def fit(image, width, height):
    scale = min(width / image.shape[1], height / image.shape[0])
    small = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)),
                       interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width, 3), np.uint8)
    y, x = (height - small.shape[0]) // 2, (width - small.shape[1]) // 2
    canvas[y:y + small.shape[0], x:x + small.shape[1]] = small
    return canvas


def ffmpeg_writer(path, width, height):
    return subprocess.Popen(['ffmpeg', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                             '-s', f'{width}x{height}', '-r', str(FPS), '-i', '-', '-c:v', 'libx264',
                             '-preset', 'veryfast', '-crf', '23', '-pix_fmt', 'yuv420p',
                             '-movflags', '+faststart', str(path)], stdin=subprocess.PIPE)


def gif(source, target, fps, width, speed=1.0, pattern=False):
    vf = (f'setpts=PTS/{speed},fps={fps},scale={width}:-1:flags=lanczos,split[a][b];'
          '[a]palettegen=max_colors=64:stats_mode=diff[p];[b][p]paletteuse=dither=none:diff_mode=rectangle')
    source_args = ['-framerate', '20', '-pattern_type', 'glob', '-i', str(source)] if pattern else ['-i', str(source)]
    subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', *source_args, '-vf', vf, '-loop', '0', str(target)],
                   check=True)


def side_by_side(renderer, trial, detections, output, speed):
    model = renderer.load_composed_model(str(trial / 'world.xml'), renderer.resolve_galbot_model_path(''),
                                         enable_head_cameras=True)
    model.vis.global_.offwidth = max(PANEL_W, model.vis.global_.offwidth)
    model.vis.global_.offheight = max(PANEL_H, model.vis.global_.offheight)
    data = mujoco.MjData(model)
    joint_ids = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i): i for i in range(model.njnt)}
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'base_link')
    scene = mujoco.Renderer(model, PANEL_H, PANEL_W)
    camera = mujoco.MjvCamera()
    camera.distance, camera.elevation, camera.azimuth = 2.6, -25, 135
    states, captions = renderer.frames(trial), renderer.captions(trial)
    row = renderer.result_of(trial) or {}
    verdict = 'PASS' if row.get('passed') else 'FAIL'
    yaw = row.get('case_yaw_deg')

    width, height = 2 * PANEL_W, PANEL_H + BAR
    writer = ffmpeg_writer(output, width, height)
    t0, t_end = states[0][0], states[-1][0]
    i = ci = di = 0
    shown, latest, latest_t = {}, None, None
    count = int((t_end - t0) / speed * FPS) + 1
    for k in range(count):
        t = t0 + k * speed / FPS
        while i < len(states) - 1 and states[i + 1][0] <= t:
            i += 1
        while ci < len(captions) and captions[ci][0] <= t:
            shown[captions[ci][1]] = captions[ci][2]
            ci += 1
        while di < len(detections) and detections[di][0] <= t:
            latest_t, latest = detections[di]
            di += 1
        renderer.apply(model, data, joint_ids, states[i][1], base)
        camera.lookat[:] = data.xpos[base] + [0, 0, 0.6]
        scene.update_scene(data, camera=camera)
        left = scene.render().copy()
        renderer.text(left, f"t = {t - t0:5.1f} s  ({speed:g}x)", (14, 30), 0.6)
        renderer.text(left, f"state: {shown.get('state', '-')}", (14, 60), 0.65)
        for j, key in enumerate(('head', 'grasp')):
            if key in shown:
                renderer.text(left, shown[key], (14, 90 + 28 * j), 0.65, (255, 230, 120))
        if latest is None:
            right = np.zeros((PANEL_H, PANEL_W, 3), np.uint8)
            renderer.text(right, 'waiting for the first camera frame', (180, PANEL_H // 2), 0.7)
        else:
            right = fit(cv2.cvtColor(cv2.imread(str(latest)), cv2.COLOR_BGR2RGB), PANEL_W, PANEL_H)
            if t - latest_t > 1.0:      # the detector only runs while the robot looks for the cube
                right = (right * 0.6).astype(np.uint8)
                renderer.text(right, f'last OpenCV 5 frame, {t - latest_t:.0f} s ago', (14, PANEL_H - 20), 0.65)
        frame = np.zeros((height, width, 3), np.uint8)
        frame[BAR:, :PANEL_W], frame[BAR:, PANEL_W:] = left, right
        cv2.line(frame, (PANEL_W, BAR), (PANEL_W, height), (255, 255, 255), 2)
        header = (f"Live mission, cube yaw {yaw:+.0f} deg: {verdict}" if yaw is not None else verdict)
        renderer.text(frame, header, (14, 30), 0.75, (120, 255, 120) if verdict == 'PASS' else (255, 140, 120))
        renderer.text(frame, 'robot camera -> OpenCV 5 cv2.dnn (YOLO11n-seg) -> mask', (PANEL_W + 14, 30), 0.7)
        writer.stdin.write(frame.tobytes())
    writer.stdin.close()
    writer.wait()
    scene.close()
    print(f'{output} ({count / FPS:.0f} s, {verdict}, {len(detections)} OpenCV 5 frames)', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('trial', type=Path, help='validation_runs/opencv5_live_<run>/<case> with recording.jsonl')
    parser.add_argument('--frames', type=Path, required=True, help='run_live_demo.sh --record output')
    parser.add_argument('--output', type=Path, default=HERE / 'docs/media')
    parser.add_argument('--speed', type=float, default=2.0, help='mission time per video second (mp4)')
    args = parser.parse_args()

    detections = detection_frames(args.frames)
    if not detections:
        raise SystemExit(f'no frame_*.jpg in {args.frames}: run run_live_demo.sh --record')
    args.output.mkdir(parents=True, exist_ok=True)
    video = args.output / 'live_mission_opencv5.mp4'
    side_by_side(load_renderer(), args.trial.resolve(), detections, video, args.speed)
    gif(video, args.output / 'live_mission_opencv5.gif', fps=5, width=800, speed=3.0)
    gif(args.frames / 'frame_*.jpg', args.output / 'opencv5_detections.gif', fps=5, width=480, pattern=True)
    for name in ('live_mission_opencv5.mp4', 'live_mission_opencv5.gif', 'opencv5_detections.gif'):
        print(f'{args.output / name}: {(args.output / name).stat().st_size / 1e6:.1f} MB')


if __name__ == '__main__':
    main()
