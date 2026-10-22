#!/usr/bin/env bash
# Stage the minimal Docker build context for the Lambda image into build/lambda-context.
# Only these files enter the image: galbot_cv5/*.py, the FYP head_localisation.py and
# head_yaw.py (ROS-free), and requirements-lambda.txt. No models, datasets or recordings.
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
fyp="$here/fyp/galbot_mobile_manipulation"
out="$here/build/lambda-context"
rm -rf "$out"
mkdir -p "$out/galbot_cv5" "$out/galbot_mobile_manipulation"
cp "$here"/galbot_cv5/*.py "$out/galbot_cv5/"
cp "$fyp/head_localisation.py" "$fyp/head_yaw.py" "$out/galbot_mobile_manipulation/"
echo '"""Vendored ROS-free FYP geometry modules (head_localisation, head_yaw)."""' \
    > "$out/galbot_mobile_manipulation/__init__.py"
cp "$here/aws/Dockerfile" "$here/aws/requirements-lambda.txt" "$out/"
find "$out" -type f | sort
