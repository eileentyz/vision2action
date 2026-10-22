#!/usr/bin/env bash
# Build the Lambda image and run the synthetic job through the Lambda Runtime Interface
# Emulator, with a local directory standing in for S3. No AWS account, no network calls
# from the function, no charges. Optional: --with-model also runs the cv2.dnn path with
# the local ONNX export on the synthetic bundle (the model stays on this machine).
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
cd "$here"
"$here/aws/stage_context.sh" >/dev/null
docker build -q -t galbot-cv5-eval:local build/lambda-context >/dev/null
bucket="$here/build/smoke-bucket-$$"
mkdir -p "$bucket/inputs" "$bucket/jobs/synthetic-smoke"
.venv-cv5/bin/python tools/make_synthetic_bundle.py --output "$bucket/stage" >/dev/null
mv "$bucket/stage/synthetic_stereo.npz" "$bucket/inputs/" && rm -rf "$bucket/stage"
echo '{"job_id":"synthetic-smoke","bundles":["inputs/synthetic_stereo.npz"],"mask_sources":["hsv","gt"]}' \
    > "$bucket/jobs/synthetic-smoke/job.json"
jobs=(synthetic-smoke)
if [[ "${1:-}" == "--with-model" ]]; then
    mkdir -p "$bucket/models" "$bucket/jobs/synthetic-dnn"
    cp models/cube_yolo11n_seg.onnx models/cube_yolo11n_seg.json "$bucket/models/"
    echo '{"job_id":"synthetic-dnn","bundles":["inputs/synthetic_stereo.npz"],"mask_sources":["dnn"],"model":"models/cube_yolo11n_seg.onnx"}' \
        > "$bucket/jobs/synthetic-dnn/job.json"
    jobs+=(synthetic-dnn)
fi
name="galbot-cv5-smoke-$$"
docker run -d --rm --name "$name" --user "$(id -u):$(id -g)" -p 127.0.0.1:9000:8080 \
    --memory 3008m --cpus 2 -e GALBOT_LOCAL_BUCKET_DIR=/bucket -v "$bucket:/bucket" galbot-cv5-eval:local >/dev/null
trap 'docker stop "$name" >/dev/null 2>&1 || true' EXIT
sleep 2
status=0
for job in "${jobs[@]}"; do
    response=$(curl -s -XPOST localhost:9000/2015-03-31/functions/function/invocations \
        -d "{\"Records\":[{\"s3\":{\"bucket\":{\"name\":\"local\"},\"object\":{\"key\":\"jobs/$job/job.json\"}}}]}")
    echo "$job: $response"
    [[ "$response" == *'"status": "ok"'* ]] || status=1
done
docker logs "$name" 2>&1 | grep -E '^REPORT' || true
echo "reports: $bucket/reports"
exit $status
