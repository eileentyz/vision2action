# OpenCV 5 evaluation Lambda (container image). Build context: build/lambda-context
# (created by aws/stage_context.sh), which holds only the files copied below.
FROM public.ecr.aws/lambda/python:3.12@sha256:25aee9bac65f131c5d090d3e3a36ccf232d65cf3e6c1dbd0bffd43ca7ce34406

# The base digest locks AL2023 release 2023.12.20260914; take the security fixes for the
# HIGH and MEDIUM ECR scan findings (openssl, curl, rpm, pcre2, libxml2) from a pinned
# later release.
RUN dnf -y --releasever=2023.12.20261021 upgrade openssl-snapsafe-libs openssl-fips-provider-latest \
        libcurl-minimal curl-minimal rpm rpm-libs pcre2 pcre2-syntax libxml2 \
    && dnf clean all && rm -rf /var/cache/dnf

COPY requirements-lambda.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements-lambda.txt

# Competition package and the two ROS-free FYP geometry modules it reuses unchanged.
COPY galbot_cv5/ ${LAMBDA_TASK_ROOT}/galbot_cv5/
COPY galbot_mobile_manipulation/ ${LAMBDA_TASK_ROOT}/galbot_mobile_manipulation/

# One OpenCV thread pool; keep BLAS from spawning a competing pool (see README).
ENV OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 GALBOT_TMP=/tmp

CMD ["galbot_cv5.aws_handler.handler"]
