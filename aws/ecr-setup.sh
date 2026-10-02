#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# ECR Setup — Create repositories for API and Webapp images
# ═══════════════════════════════════════════════════════════════════════
# Run this ONCE to set up ECR repositories.
#
# Prerequisites:
#   - AWS CLI installed and configured (aws configure)
#   - IAM user/role with ecr:CreateRepository permission
#
# Usage:
#   chmod +x aws/ecr-setup.sh
#   ./aws/ecr-setup.sh
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail

REGION="${AWS_DEFAULT_REGION:-ap-south-1}"
API_REPO="churn-intelligence-api"
WEBAPP_REPO="churn-intelligence-webapp"

echo "════════════════════════════════════════════"
echo "  Setting up ECR repositories"
echo "  Region: ${REGION}"
echo "════════════════════════════════════════════"

# Create API repository
echo ""
echo "Creating repository: ${API_REPO}..."
aws ecr create-repository \
    --repository-name "${API_REPO}" \
    --region "${REGION}" \
    --image-scanning-configuration scanOnPush=true \
    --image-tag-mutability MUTABLE \
    2>/dev/null && echo "  ✓ ${API_REPO} created" \
    || echo "  → ${API_REPO} already exists (skipping)"

# Create Webapp repository
echo ""
echo "Creating repository: ${WEBAPP_REPO}..."
aws ecr create-repository \
    --repository-name "${WEBAPP_REPO}" \
    --region "${REGION}" \
    --image-scanning-configuration scanOnPush=true \
    --image-tag-mutability MUTABLE \
    2>/dev/null && echo "  ✓ ${WEBAPP_REPO} created" \
    || echo "  → ${WEBAPP_REPO} already exists (skipping)"

# Set lifecycle policy (keep last 10 images, auto-delete older ones)
LIFECYCLE_POLICY='{
  "rules": [
    {
      "rulePriority": 1,
      "description": "Keep last 10 images",
      "selection": {
        "tagStatus": "any",
        "countType": "imageCountMoreThan",
        "countNumber": 10
      },
      "action": {
        "type": "expire"
      }
    }
  ]
}'

echo ""
echo "Applying lifecycle policy (keep last 10 images)..."
for REPO in "${API_REPO}" "${WEBAPP_REPO}"; do
    aws ecr put-lifecycle-policy \
        --repository-name "${REPO}" \
        --region "${REGION}" \
        --lifecycle-policy-text "${LIFECYCLE_POLICY}" \
        2>/dev/null && echo "  ✓ Policy applied to ${REPO}" \
        || echo "  → Could not apply policy to ${REPO}"
done

echo ""
echo "════════════════════════════════════════════"
echo "  ✓ ECR setup complete"
echo "════════════════════════════════════════════"

# Print registry URI for reference
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo ""
echo "Registry URI: ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
echo "API image:    ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${API_REPO}"
echo "Webapp image: ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${WEBAPP_REPO}"
