#!/usr/bin/env bash
# Read-only pre-deployment checks. Creates nothing, uploads nothing, prints no credentials.
#   aws/preflight.sh galbot-cv5
set -uo pipefail
profile="${1:-galbot-cv5}"
region="ap-southeast-2"
here="$(cd "$(dirname "$0")" && pwd)"
aws() { command aws --profile "$profile" --region "$region" "$@"; }
section() { printf '\n== %s\n' "$1"; }

section "Identity (temporary session: aws login team-member role or IAM user, not root)"
arn=$(aws sts get-caller-identity --query Arn --output text) || { echo "not signed in: run aws login --profile $profile"; exit 1; }
account=$(aws sts get-caller-identity --query Account --output text)
echo "$arn"
[[ "$arn" == *":root" ]] && echo "WARNING: this is the root user; use the IAM user instead"

section "Free Tier plan and credits"
aws freetier get-account-plan-state --region us-east-1 \
    --query '{plan:accountPlanType,status:accountPlanStatus,expires:accountPlanExpirationDate,remaining:accountPlanRemainingCredits}' \
    --output table || echo "could not read plan state (needs freetier:GetAccountPlanState)"

section "Lambda account limits in $region"
aws lambda get-account-settings --query '{concurrency:AccountLimit.ConcurrentExecutions,unreserved:AccountLimit.UnreservedConcurrentExecutions,codeStorageUsed:AccountUsage.TotalCodeSize}' --output table

section "Existing stacks / bucket name"
aws cloudformation list-stacks --query "StackSummaries[?starts_with(StackName,'galbot-cv5-eval') && StackStatus!='DELETE_COMPLETE'].[StackName,StackStatus]" --output text
bucket="galbot-cv5-eval-$account-$region"
if aws s3api head-bucket --bucket "$bucket" 2>/dev/null; then echo "bucket $bucket already exists"; else echo "bucket $bucket: not present (will be created)"; fi

section "Deployer policy validation (IAM Access Analyzer)"
aws accessanalyzer validate-policy --policy-type IDENTITY_POLICY \
    --policy-document "file://$here/deployer-policy.json" \
    --query 'findings[].[findingType,issueCode,findingDetails]' --output text || true

section "Month-to-date cost (Cost Explorer can lag up to 24 h)"
aws ce get-cost-and-usage --region us-east-1 --granularity MONTHLY --metrics UnblendedCost \
    --time-period "Start=$(date +%Y-%m-01),End=$(date -d tomorrow +%Y-%m-%d)" \
    --query 'ResultsByTime[].Total.UnblendedCost' --output table || echo "Cost Explorer not available yet (it can take 24 h after sign-up)"
