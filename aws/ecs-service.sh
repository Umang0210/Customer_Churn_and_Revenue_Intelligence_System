#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# ECS Service Setup — Create cluster, log group, and service
# ═══════════════════════════════════════════════════════════════════════
# Run this ONCE to create the ECS infrastructure.
# After this, Jenkins will handle updates via `aws ecs update-service`.
#
# Prerequisites:
#   - AWS CLI configured
#   - ECR repositories created (run ecr-setup.sh first)
#   - Docker images pushed to ECR at least once
#   - A VPC with subnets and a security group allowing ports 8000, 8001
#
# Usage:
#   chmod +x aws/ecs-service.sh
#   ./aws/ecs-service.sh
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail

REGION="${AWS_DEFAULT_REGION:-ap-south-1}"
CLUSTER_NAME="churn-cluster"
SERVICE_NAME="churn-service"
TASK_FAMILY="churn-intelligence-task"
LOG_GROUP="/ecs/churn-intelligence"

# ── AWS Network Config ───────────────────────────────────────────────
SUBNET_IDS="${SUBNET_IDS:-subnet-0ca7aed097336a0e1}"
SECURITY_GROUP="${SECURITY_GROUP:-sg-0e54add0a088df4c2}"

echo "════════════════════════════════════════════"
echo "  Setting up ECS infrastructure"
echo "  Region:  ${REGION}"
echo "  Cluster: ${CLUSTER_NAME}"
echo "════════════════════════════════════════════"

# 1. Create CloudWatch log group
echo ""
echo "Step 1: Creating CloudWatch log group..."
aws logs create-log-group \
    --log-group-name "${LOG_GROUP}" \
    --region "${REGION}" \
    2>/dev/null && echo "  ✓ Log group created" \
    || echo "  → Log group already exists (skipping)"

# 2. Create ECS cluster
echo ""
echo "Step 2: Creating ECS cluster..."
aws ecs create-cluster \
    --cluster-name "${CLUSTER_NAME}" \
    --region "${REGION}" \
    --capacity-providers FARGATE \
    --default-capacity-provider-strategy capacityProvider=FARGATE,weight=1 \
    2>/dev/null && echo "  ✓ Cluster created" \
    || echo "  → Cluster already exists (skipping)"

# 3. Register task definition
echo ""
echo "Step 3: Registering task definition..."
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# Replace placeholders in task definition
TASK_DEF=$(cat aws/ecs-task-definition.json \
    | sed "s/ACCOUNT_ID/${ACCOUNT_ID}/g" \
    | sed "s/REGION/${REGION}/g")

echo "${TASK_DEF}" > /tmp/task-def-resolved.json

TASK_ARN=$(aws ecs register-task-definition \
    --cli-input-json file:///tmp/task-def-resolved.json \
    --region "${REGION}" \
    --query 'taskDefinition.taskDefinitionArn' \
    --output text)

echo "  ✓ Task definition registered: ${TASK_ARN}"

# 4. Create ECS service
echo ""
echo "Step 4: Creating ECS service..."
aws ecs create-service \
    --cluster "${CLUSTER_NAME}" \
    --service-name "${SERVICE_NAME}" \
    --task-definition "${TASK_ARN}" \
    --desired-count 1 \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={subnets=[${SUBNET_IDS}],securityGroups=[${SECURITY_GROUP}],assignPublicIp=ENABLED}" \
    --region "${REGION}" \
    2>/dev/null && echo "  ✓ Service created" \
    || echo "  → Service may already exist. Use 'aws ecs update-service' to update."

echo ""
echo "════════════════════════════════════════════"
echo "  ✓ ECS setup complete"
echo "════════════════════════════════════════════"
echo ""
echo "  Cluster : ${CLUSTER_NAME}"
echo "  Service : ${SERVICE_NAME}"
echo "  Task    : ${TASK_ARN}"
echo ""
echo "  Jenkins will now auto-deploy on each successful pipeline run."
echo "  To check service status:"
echo "    aws ecs describe-services --cluster ${CLUSTER_NAME} --services ${SERVICE_NAME} --region ${REGION}"
