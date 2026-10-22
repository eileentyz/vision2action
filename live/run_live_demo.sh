#!/usr/bin/env bash
# End-to-end live demo: robot mission with the OpenCV 5 detector -> local report -> AWS report.
#
#   live/run_live_demo.sh [--yaw DEG]... [--viewer] [--record] [--cloud] [--run-dir DIR]
#
# 1. Starts tools/opencv5_mask_server.py (OpenCV 5 cv2.dnn, .venv-cv5) unless one is running.
# 2. Runs the FYP mission (validate_pick_place_mission.py, pick_place_mission_yolo.launch.py with
#    yolo_backend:=opencv5): every head and wrist mask the robot uses comes from OpenCV 5.
#    --viewer opens the MuJoCo viewer. --run-dir DIR skips this step and reuses DIR.
#    --record keeps every frame OpenCV 5 served, with its mask drawn, in build/live/<run>/frames
#    (local only; tools/make_demo_media.py turns them into the demo GIF and video).
# 3. Exports the mission locally: re-rendered head stereo bundle and upload-safe live-mission
#    records (poses, outcome, detector statistics; no images or paths).
# 4. Local OpenCV 5 evaluation -> results/live/<run>/local/summary.md
# 5. --cloud: uploads the job (sanitised bundle, live-mission records, model) to the deployed
#    stack; the S3 trigger runs the Lambda (OpenCV 5) and the report comes back to
#    results/cloud/<job>/. Needs AWS_PROFILE=galbot-cv5 and a fresh `aws login`.
#
# Needs the FYP ROS 2 workspace (GALBOT_FYP_WS; private, not part of this repository) built with
# the OpenCV 5 backend from integration/ros2/. GALBOT_CV5_WS can point at a separate overlay.
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"           # this repository
# The robot side is the FYP ROS 2 workspace (not public), built with integration/ros2/.
fyp="${GALBOT_FYP_WS:?set GALBOT_FYP_WS to the FYP colcon workspace}"
repo="${GALBOT_FYP_REPO:-$fyp/src/galbot_mobile_manipulation}"
overlay="${GALBOT_CV5_WS:-$fyp}"
socket="${GALBOT_CV5_SOCKET:-${XDG_RUNTIME_DIR:-/tmp}/galbot_cv5_mask.sock}"
model="$here/models/cube_yolo11n_seg.onnx"
yaws=() viewer=() record=() cloud=false run_dir=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --yaw) yaws+=(--yaw "$2"); shift 2 ;;
        --viewer) viewer=(--viewer); shift ;;
        --record) record=(--record); shift ;;
        --cloud) cloud=true; shift ;;
        --run-dir) run_dir="$(cd "$2" && pwd)"; shift 2 ;;
        *) echo "unknown argument $1" >&2; exit 2 ;;
    esac
done
[[ ${#yaws[@]} -gt 0 ]] || yaws=(--yaw 30)
[[ -f "$model" ]] || { echo "missing $model: run tools/export_onnx.py (see README.md)" >&2; exit 2; }

ros_env() {
    # shellcheck disable=SC1091
    set +u; source /opt/ros/jazzy/setup.bash; source "$fyp/install/setup.bash"
    source "$overlay/install/setup.bash"; set -u
}
cv5() { env -u PYTHONPATH "$here/.venv-cv5/bin/python" "$@"; }
step() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

server_pid=""
cleanup() {
    if [[ -n "$server_pid" ]]; then
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT

if [[ -z "$run_dir" ]]; then
    run_dir="$overlay/validation_runs/opencv5_live_$(date +%Y%m%d_%H%M%S)"
    [[ ${#record[@]} -eq 0 ]] || record=(--record "$here/build/live/$(basename "$run_dir")/frames")
    if [[ -S "$socket" ]] && cv5 -c 'import socket,sys; s=socket.socket(socket.AF_UNIX); s.connect(sys.argv[1])' "$socket" 2>/dev/null; then
        step "OpenCV 5 mask server already running on $socket"
        [[ ${#record[@]} -eq 0 ]] || { echo "--record needs the server this script starts; stop that one" >&2; exit 2; }
    else
        step "1. OpenCV 5 mask server"
        # Not via cv5(): $! must be the server itself (env execs python), not a subshell.
        env -u PYTHONPATH "$here/.venv-cv5/bin/python" "$here/tools/opencv5_mask_server.py" \
            --model "$model" --socket "$socket" --threads 4 "${record[@]}" &
        server_pid=$!
        for _ in $(seq 60); do [[ -S "$socket" ]] && break; sleep 0.5; done
        [[ -S "$socket" ]] || { echo "server did not start" >&2; exit 1; }
    fi
    step "2. Live mission, detector = OpenCV 5 ($run_dir)"
    ( ros_env; "$fyp/.venv/bin/python" "$repo/tools/validate_pick_place_mission.py" --output "$run_dir" \
        "${yaws[@]}" "${viewer[@]}" --launch-file pick_place_mission_yolo.launch.py \
        --launch-arg yolo_backend:=opencv5 --launch-arg "opencv5_socket:=$socket" ) || true
    cleanup; server_pid=""
fi
[[ -f "$run_dir/results.json" ]] || { echo "no results.json in $run_dir" >&2; exit 1; }
name="$(basename "$run_dir")"
trials=("$run_dir"/*/)
work="$here/build/live/$name"
mkdir -p "$work/job"

step "3. Export the mission (local)"
( ros_env; "$here/.venv-export/bin/python" "$here/tools/export_trial_bundles.py" \
    --output "$work/bundle/$name.npz" "${trials[@]%/}" )
cv5 "$here/tools/export_live_missions.py" --output "$work/job/live_missions.json" "${trials[@]%/}"
"$here/.venv-export/bin/python" "$here/tools/sanitize_for_upload.py" --bundle "$work/bundle/$name.npz" \
    --output "$work/job" >/dev/null
cp "$model" "${model%.onnx}.json" "$work/job/"
job_id="live-${name#opencv5_live_}"
cat > "$work/job/job.json" <<EOF
{"job_id": "$job_id", "bundles": ["$name.npz"], "live_missions": ["live_missions.json"],
 "mask_sources": ["dnn", "gt", "hsv"], "model": "cube_yolo11n_seg.onnx"}
EOF

step "4. Local OpenCV 5 evaluation"
cv5 "$here/tools/evaluate.py" --job "$work/job/job.json" --threads 8 --output "$here/results/live/$name/local"

if $cloud; then
    step "5. AWS: S3 -> Lambda (OpenCV 5) -> report"
    export AWS_REGION="${AWS_REGION:-ap-southeast-2}"
    account=$(aws sts get-caller-identity --query Account --output text)
    bucket=$(aws cloudformation describe-stacks --stack-name galbot-cv5-eval \
        --query 'Stacks[0].Outputs[?OutputKey==`BucketName`].OutputValue' --output text)
    { "$here/aws/submit_job.sh" "$bucket" "$work/job" --allow-project-data 2>&1 | grep -v '^Completed '
      echo; cat "$here/results/cloud/$job_id/summary.md"; } | sed -u "s/$account/<account-id>/g"
fi
