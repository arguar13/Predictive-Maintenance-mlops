#!/usr/bin/env bash
# Bootstrap de LocalStack ("la nube falsa"): se ejecuta automaticamente una
# sola vez, cuando LocalStack termina de arrancar (hook nativo
# /etc/localstack/init/ready.d/*.sh), y crea los mismos recursos que existen
# en AWS real (ver terraform/s3.tf) para que la app y sus tests corran contra
# infraestructura simulada localmente, sin tocar recursos ni credenciales
# reales de AWS.
set -euo pipefail

BUCKET_NAME="${MLOPS_S3_BUCKET:-predictive-maintenance-mlops-artifacts-040175285118}"
DLQ_NAME="${MLOPS_SQS_DLQ:-engine-alerts-dlq}"
DB_SECRET_NAME="${MLOPS_DB_SECRET:-predictive-maintenance/db-credentials}"

echo "[localstack-init] Creando bucket S3: ${BUCKET_NAME}"
awslocal s3 mb "s3://${BUCKET_NAME}"
awslocal s3api put-bucket-versioning \
  --bucket "${BUCKET_NAME}" \
  --versioning-configuration Status=Enabled

echo "[localstack-init] Creando cola SQS: ${DLQ_NAME}"
awslocal sqs create-queue --queue-name "${DLQ_NAME}"

echo "[localstack-init] Creando secreto en Secrets Manager: ${DB_SECRET_NAME}"
awslocal secretsmanager create-secret \
  --name "${DB_SECRET_NAME}" \
  --secret-string '{"username":"mlflow_user","password":"mlflow_password","dbname":"mlflow_db","host":"postgres","port":5432}'

echo "[localstack-init] Listo: bucket=${BUCKET_NAME} cola=${DLQ_NAME} secreto=${DB_SECRET_NAME}"
