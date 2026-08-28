"""Integración End-to-End: la aplicación habla con "AWS" (S3, SQS,
Secrets Manager) contra LocalStack — infraestructura simulada, sin tocar
recursos reales ni consumir cuota de AWS.

Fase 3 ("la falsa nube"). Requiere Docker.

Nota: testcontainers/boto3 se importan de forma perezosa (dentro de los
fixtures, no a nivel de módulo) para que la recolección de pytest siga
siendo rápida en `make test` cuando estos tests se deseleccionan por marker.
"""

import pytest

pytestmark = pytest.mark.integration

# Mismo nombre que el bucket real (ver terraform/s3.tf, core_ml/feature_store/
# feature_store.yaml, core_ml/.dvc/config): LocalStack lo simula sin tocar AWS.
BUCKET_NAME = "predictive-maintenance-mlops-artifacts-040175285118"


@pytest.fixture(scope="module")
def localstack_container():
    from testcontainers.community.localstack import LocalStackContainer

    with LocalStackContainer(image="localstack/localstack:3.8").with_services(
        "s3", "sqs", "secretsmanager"
    ) as container:
        yield container


@pytest.fixture(scope="module")
def aws_clients(localstack_container):
    import boto3

    endpoint_url = localstack_container.get_url()
    session = boto3.session.Session(
        aws_access_key_id="test",
        aws_secret_access_key="test",
        region_name="us-east-1",
    )
    return {
        "s3": session.client("s3", endpoint_url=endpoint_url),
        "sqs": session.client("sqs", endpoint_url=endpoint_url),
        "secretsmanager": session.client("secretsmanager", endpoint_url=endpoint_url),
    }


def test_s3_bucket_roundtrip_mirrors_the_feast_offline_store_path(aws_clients):
    """El mismo patrón de ruta que usa Feast (features.py) y DVC (S3 remote)."""
    s3 = aws_clients["s3"]
    s3.create_bucket(Bucket=BUCKET_NAME)

    key = "feast/data/engine_features.parquet"
    body = b"parquet-bytes-placeholder"
    s3.put_object(Bucket=BUCKET_NAME, Key=key, Body=body)

    obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
    assert obj["Body"].read() == body

    listing = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix="feast/")
    assert any(item["Key"] == key for item in listing.get("Contents", []))


def test_sqs_queue_roundtrip_for_alert_dead_lettering(aws_clients):
    sqs = aws_clients["sqs"]
    queue_url = sqs.create_queue(QueueName="engine-alerts-dlq")["QueueUrl"]

    message_body = '{"engine_id": "ENG_001", "status": "Critical"}'
    sqs.send_message(QueueUrl=queue_url, MessageBody=message_body)

    messages = sqs.receive_message(QueueUrl=queue_url, MaxNumberOfMessages=1, WaitTimeSeconds=5)
    assert len(messages.get("Messages", [])) == 1
    assert "ENG_001" in messages["Messages"][0]["Body"]


def test_secrets_manager_stores_and_retrieves_db_credentials(aws_clients):
    secretsmanager = aws_clients["secretsmanager"]
    secret_name = "predictive-maintenance/db-credentials"
    secret_value = (
        '{"username": "mlflow_user", "password": "mlflow_password", "dbname": "mlflow_db"}'
    )

    secretsmanager.create_secret(Name=secret_name, SecretString=secret_value)

    fetched = secretsmanager.get_secret_value(SecretId=secret_name)
    assert fetched["SecretString"] == secret_value
