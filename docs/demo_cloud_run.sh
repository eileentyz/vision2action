#!/usr/bin/env bash
# Screen-recording-safe cloud demo: run the synthetic job (no project data) on the deployed
# Lambda with a fresh job ID, then print the summary and the Lambda REPORT line. Every line
# of output has the AWS account ID replaced with <account-id>.
#
#   AWS_PROFILE=galbot-cv5 docs/demo_cloud_run.sh
#
# Needs a signed-in session (aws login --region ap-southeast-2 --profile galbot-cv5).
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
export AWS_REGION="${AWS_REGION:-ap-southeast-2}"
account=$(aws sts get-caller-identity --query Account --output text)
redact() { sed -u "s/$account/<account-id>/g"; }
bucket=$(aws cloudformation describe-stacks --stack-name galbot-cv5-eval \
    --query 'Stacks[0].Outputs[?OutputKey==`BucketName`].OutputValue' --output text)
job_id="synthetic-demo-$(date +%Y%m%d-%H%M%S)"
work=$(mktemp -d "$here/samples/demo.XXXX")
trap 'rm -rf "$work"' EXIT
cp "$here/samples/synthetic/synthetic_stereo.npz" "$work/"
printf '{"job_id": "%s", "bundles": ["synthetic_stereo.npz"], "mask_sources": ["hsv", "gt"]}\n' "$job_id" \
    > "$work/job.json"

{
    echo "== stack: $(aws cloudformation describe-stacks --stack-name galbot-cv5-eval --query 'Stacks[0].StackStatus' --output text)"
    echo "== submitting $job_id to s3://$bucket/jobs/$job_id/job.json"
    "$here/aws/submit_job.sh" "$bucket" "$work" 2>&1 | grep -v '^Completed '
    echo
    echo "== reports/$job_id/summary.md"
    cat "$here/results/cloud/$job_id/summary.md"
    echo
    echo "== Lambda REPORT (CloudWatch Logs)"
    sleep 5
    aws logs filter-log-events --log-group-name /aws/lambda/galbot-cv5-eval-evaluator \
        --start-time $(( ($(date +%s) - 600) * 1000 )) --filter-pattern REPORT \
        --output json | python3 -c 'import json, sys
events = [e["message"].strip() for e in json.load(sys.stdin)["events"]]
print(events[-1] if events else "no REPORT line yet")'
} 2>&1 | redact
