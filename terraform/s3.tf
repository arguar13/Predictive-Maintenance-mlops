resource "aws_s3_bucket" "mlflow_artifacts" {
  bucket = "${var.project_name}-artifacts-${data.aws_caller_identity.current.account_id}"
}

resource "aws_s3_bucket_versioning" "mlflow_artifacts_versioning" {
  bucket = aws_s3_bucket.mlflow_artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "mlflow_artifacts_encryption" {
  bucket = aws_s3_bucket.mlflow_artifacts.id
  rule {
    apply_server_side_encryption_by_default {
      # CMK del proyecto (terraform/kms.tf) en vez de la clave gestionada por
      # AWS: habilita rotacion, key policy propia y atribucion en CloudTrail.
      sse_algorithm     = "aws:kms"
      kms_master_key_id = aws_kms_key.mlops.arn
    }
    # Clave de bucket: una sola llamada a KMS por lote de objetos en vez de
    # una por objeto. MLflow escribe muchos artefactos pequeños por run.
    bucket_key_enabled = true
  }
}

# El bucket guarda artefactos de modelos, datasets versionados con DVC y el
# registro de Feast: nada de eso debe ser jamas accesible publicamente. El
# bucket de estado de Terraform (terraform/bootstrap/main.tf) ya tenia este
# bloqueo; el de artefactos no -- trivy lo reporta como AWS-0094.
resource "aws_s3_bucket_public_access_block" "mlflow_artifacts" {
  bucket                  = aws_s3_bucket.mlflow_artifacts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Utilizado para agregar la cuenta al nombre del bucket
data "aws_caller_identity" "current" {}

output "s3_bucket_name" {
  value = aws_s3_bucket.mlflow_artifacts.bucket
}