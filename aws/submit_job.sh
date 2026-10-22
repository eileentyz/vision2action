#!/usr/bin/env bash
# Upload a local job directory (job.json + the bundles, live-mission files and model it names) to the deployed
# bucket, trigger the evaluation by writing jobs/<id>/job.json last, and download the report.
# A model's <name>.json export sidecar is uploaded too when it sits next to the model.
#
#   aws/submit_job.sh BUCKET samples/synthetic
#
# Data policy guard: by default only directories under samples/ (synthetic,
# no project data) may be uploaded. Anything else needs --allow-project-data, which must
# only be used after the project owner has approved sharing that data.
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
bucket="${1:?bucket name (stack output BucketName)}"
dir="$(cd "${2:?local job directory with job.json}" && pwd)"
allow="${3:-}"
if [[ "$dir" != "$here/samples/"* && "$allow" != "--allow-project-data" ]]; then
    echo "refusing to upload $dir: only samples/ is cleared for upload" >&2
    exit 2
fi
job_id=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["job_id"])' "$dir/job.json")
if [[ -n "$(aws s3 ls "s3://$bucket/reports/$job_id/" 2>/dev/null)" ]]; then
    echo "reports/$job_id/ already exists in the bucket; use a new job_id" >&2
    exit 2
fi
# Local bundle names map to inputs/<name>; the uploaded job names the S3 keys.
python3 - "$dir/job.json" > "$dir/.job.s3.json" <<'PY'
import json, sys
job = json.load(open(sys.argv[1]))
job['bundles'] = ['inputs/' + b.split('/')[-1] for b in job.get('bundles', [])]
if job.get('live_missions'):
    job['live_missions'] = ['inputs/' + b.split('/')[-1] for b in job['live_missions']]
if job.get('model'):
    job['model'] = 'models/' + job['model'].split('/')[-1]
job.pop('recorded_trials', None)
print(json.dumps(job))
PY
for bundle in $(python3 -c 'import json,sys; j=json.load(open(sys.argv[1])); print(" ".join(j.get("bundles", []) + j.get("live_missions", [])))' "$dir/job.json"); do
    aws s3 cp "$dir/$bundle" "s3://$bucket/inputs/$(basename "$bundle")"
done
model=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("model") or "")' "$dir/job.json")
if [[ -n "$model" ]]; then
    aws s3 cp "$dir/$model" "s3://$bucket/models/$(basename "$model")"
    sidecar="$dir/${model%.*}.json"
    if [[ -f "$sidecar" ]]; then
        aws s3 cp "$sidecar" "s3://$bucket/models/$(basename "$sidecar")"
    fi
fi
aws s3 cp "$dir/.job.s3.json" "s3://$bucket/jobs/$job_id/job.json"
echo "submitted jobs/$job_id/job.json; waiting for reports/$job_id/ ..."
for _ in $(seq 1 60); do
    # summary.md is written after report.json, so its presence means the report is complete.
    for name in summary.md error.json; do
        if aws s3 ls "s3://$bucket/reports/$job_id/$name" >/dev/null 2>&1; then
            mkdir -p "$here/results/cloud/$job_id"
            aws s3 cp --recursive "s3://$bucket/reports/$job_id/" "$here/results/cloud/$job_id/"
            if [[ "$name" == error.json ]]; then
                cat "$here/results/cloud/$job_id/error.json" >&2
                exit 1
            fi
            exit 0
        fi
    done
    sleep 10
done
echo "no report after 10 minutes; check the function's log group" >&2
exit 1
