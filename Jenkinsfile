/*
 * Jenkinsfile — Customer Churn & Revenue Intelligence System
 * ===========================================================
 * Declarative pipeline: ML pipeline → metrics gate → Docker → ECR → ECS
 *
 * Triggers:
 *   - Push to main branch
 *   - Weekly auto-retrain (Sunday 2 AM)
 *   - Manual trigger via Jenkins UI
 *
 * Required Jenkins Credentials (configure in Jenkins > Manage Credentials):
 *   - AWS_ACCOUNT_ID       : Your 12-digit AWS account ID
 *   - AWS_DEFAULT_REGION   : e.g. us-east-1
 *   - aws-credentials      : AWS Access Key ID + Secret (type: AWS Credentials)
 */

pipeline {
    agent any

    // ── Auto-retrain: every Sunday at 2:00 AM ────────────────────────────────
    triggers {
        cron('H 2 * * 0')
    }

    environment {
        AWS_ACCOUNT_ID    = credentials('AWS_ACCOUNT_ID')
        AWS_DEFAULT_REGION = credentials('AWS_DEFAULT_REGION')
        ECR_REGISTRY      = "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_DEFAULT_REGION}.amazonaws.com"
        API_IMAGE         = 'churn-intelligence-api'
        WEBAPP_IMAGE      = 'churn-intelligence-webapp'
        IMAGE_TAG         = "${BUILD_NUMBER}"
        K8S_NAMESPACE     = 'churn-intelligence'
    }

    options {
        timeout(time: 45, unit: 'MINUTES')
        timestamps()
        buildDiscarder(logRotator(numToKeepStr: '10'))
    }

    stages {

        // ════════════════════════════════════════════════════════════════════
        // STAGE 1: Checkout
        // ════════════════════════════════════════════════════════════════════
        stage('Checkout') {
            steps {
                checkout scm
                echo "Checked out branch: ${env.BRANCH_NAME ?: 'main'}"
            }
        }

        // ════════════════════════════════════════════════════════════════════
        // STAGE 2: Install Dependencies
        // ════════════════════════════════════════════════════════════════════
        stage('Install Dependencies') {
            steps {
                sh '''
                    python3 -m pip install --upgrade pip
                    pip3 install -r requirements.txt
                '''
            }
        }

        // ════════════════════════════════════════════════════════════════════
        // STAGE 3: Run ML Pipeline (Ingest → Train → Evaluate)
        // ════════════════════════════════════════════════════════════════════
        stage('Run ML Pipeline') {
            steps {
                sh '''
                    echo "Starting full ML pipeline..."
                    python3 run_pipeline.py
                '''
            }
            post {
                always {
                    archiveArtifacts artifacts: 'logs/*.log', allowEmptyArchive: true
                    archiveArtifacts artifacts: 'reports/**/*', allowEmptyArchive: true
                }
            }
        }

        // ════════════════════════════════════════════════════════════════════
        // STAGE 4: Metrics Gate (fail build if model is not good enough)
        // ════════════════════════════════════════════════════════════════════
        stage('Metrics Gate') {
            steps {
                sh '''
                    python3 -c "
import json, sys
from pathlib import Path

report = json.loads(Path('reports/evaluation_report.json').read_text())
auc    = report.get('roc_auc', 0)
passed = report.get('passes_gate', False)
threshold = report.get('min_auc_threshold', 0.70)

print(f'ROC-AUC : {auc}')
print(f'Threshold: {threshold}')
print(f'Gate     : {\"PASSED\" if passed else \"FAILED\"}')

if not passed:
    print(f'FAIL: ROC-AUC {auc} < minimum {threshold}')
    sys.exit(1)
"
                '''
            }
        }

        // ════════════════════════════════════════════════════════════════════
        // STAGE 5: Docker Build
        // ════════════════════════════════════════════════════════════════════
        stage('Docker Build') {
            steps {
                sh """
                    echo "Building API image..."
                    docker build -t ${API_IMAGE}:${IMAGE_TAG} -t ${API_IMAGE}:latest -f Dockerfile .

                    echo "Building Webapp image..."
                    docker build -t ${WEBAPP_IMAGE}:${IMAGE_TAG} -t ${WEBAPP_IMAGE}:latest -f Dockerfile.webapp .
                """
            }
        }

        // ════════════════════════════════════════════════════════════════════
        // STAGE 6: Push to AWS ECR
        // ════════════════════════════════════════════════════════════════════
        stage('Push to ECR') {
            steps {
                withCredentials([[$class: 'AmazonWebServicesCredentialsBinding',
                                  credentialsId: 'aws-credentials']]) {
                    sh """
                        # Authenticate Docker with ECR
                        aws ecr get-login-password --region ${AWS_DEFAULT_REGION} \
                            | docker login --username AWS --password-stdin ${ECR_REGISTRY}

                        # Tag and push API image
                        docker tag ${API_IMAGE}:${IMAGE_TAG} ${ECR_REGISTRY}/${API_IMAGE}:${IMAGE_TAG}
                        docker tag ${API_IMAGE}:latest       ${ECR_REGISTRY}/${API_IMAGE}:latest
                        docker push ${ECR_REGISTRY}/${API_IMAGE}:${IMAGE_TAG}
                        docker push ${ECR_REGISTRY}/${API_IMAGE}:latest

                        # Tag and push Webapp image
                        docker tag ${WEBAPP_IMAGE}:${IMAGE_TAG} ${ECR_REGISTRY}/${WEBAPP_IMAGE}:${IMAGE_TAG}
                        docker tag ${WEBAPP_IMAGE}:latest       ${ECR_REGISTRY}/${WEBAPP_IMAGE}:latest
                        docker push ${ECR_REGISTRY}/${WEBAPP_IMAGE}:${IMAGE_TAG}
                        docker push ${ECR_REGISTRY}/${WEBAPP_IMAGE}:latest
                    """
                }
            }
        }

        // ════════════════════════════════════════════════════════════════════
        // STAGE 7: Deploy to Kubernetes
        // ════════════════════════════════════════════════════════════════════
        stage('Deploy to Kubernetes') {
            steps {
                withCredentials([[$class: 'AmazonWebServicesCredentialsBinding',
                                  credentialsId: 'aws-credentials']]) {
                    sh """
                        echo "Deploying to Kubernetes cluster..."
                        
                        # Configure kubectl (assuming EKS)
                        # aws eks update-kubeconfig --name my-cluster --region ${AWS_DEFAULT_REGION}

                        # Substitute image placeholders with actual ECR images
                        sed -i "s|IMAGE_PLACEHOLDER|${ECR_REGISTRY}/${API_IMAGE}:${IMAGE_TAG}|g" k8s/api-deployment.yaml
                        sed -i "s|WEBAPP_IMAGE_PLACEHOLDER|${ECR_REGISTRY}/${WEBAPP_IMAGE}:${IMAGE_TAG}|g" k8s/webapp-deployment.yaml

                        # Apply manifests
                        kubectl apply -f k8s/ --namespace=${K8S_NAMESPACE}

                        echo "Kubernetes deployment applied."
                    """
                }
            }
        }
    }

    // ── Post-Pipeline Notifications ──────────────────────────────────────────
    post {
        success {
            echo """
            ════════════════════════════════════════════
            ✅  PIPELINE SUCCEEDED
            ────────────────────────────────────────────
            Build   : #${BUILD_NUMBER}
            Images  : ${ECR_REGISTRY}/${API_IMAGE}:${IMAGE_TAG}
                      ${ECR_REGISTRY}/${WEBAPP_IMAGE}:${IMAGE_TAG}
            K8s NS  : ${K8S_NAMESPACE}
            ════════════════════════════════════════════
            """
        }
        failure {
            echo """
            ════════════════════════════════════════════
            ❌  PIPELINE FAILED
            ────────────────────────────────────────────
            Build : #${BUILD_NUMBER}
            Check the console output for details.
            ════════════════════════════════════════════
            """
        }
        always {
            // Clean up local Docker images to save disk
            sh '''
                docker image prune -f || true
            '''
        }
    }
}
