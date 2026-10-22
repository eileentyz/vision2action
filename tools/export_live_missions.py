"""Extract upload-safe live-mission records from validate_pick_place_mission output (OpenCV 5 env).

Per trial: the first pose the pick consumed, the simulator truth nearest to it, the
mission outcome and the live detector's per-frame statistics (backend, OpenCV
version, frames, inference ms). No images, world files, joint logs or local paths.

    .venv-cv5/bin/python tools/export_live_missions.py --output build/live/live_missions.json \\
        $GALBOT_FYP_WS/validation_runs/<run>/yaw*
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from galbot_cv5.evaluation import LIVE_FORMAT, live_record_from_trial, load_live_missions  # noqa: E402

HOME_MARKERS = ('/home/', '/Users/', 'C:\\Users\\')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('trials', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    missions = [live_record_from_trial(t) for t in args.trials if (t / 'recording.jsonl').exists()]
    if not missions:
        raise SystemExit('no trial directory with recording.jsonl')
    text = json.dumps(dict(format=LIVE_FORMAT, missions=missions), indent=2) + '\n'
    if any(m in text for m in HOME_MARKERS):
        raise SystemExit('a local path ended up in the live-missions file; refusing to write it')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    load_live_missions(args.output)       # the same check the cloud applies
    for m in missions:
        d = m.get('detector') or {}
        print(f"{m['trial']}: mission {'ok' if m.get('mission_succeeded') else 'failed'}, "
              f"pick pose {'yes' if m['pick'] else 'no'}, detector {d.get('backend', 'n/a')} "
              f"{d.get('frames', 0)} frames")
    print(f'wrote {args.output}: {len(missions)} missions')


if __name__ == '__main__':
    main()
