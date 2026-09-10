resource "aws_s3_bucket" "mlflow_artifacts" {
  bucket = "${var.project_name}-artifacts-${data.aws_caller_identity.current.account_id}"
  # force_destroy: mismo criterio que los repos ECR (terraform/ecr.tf) -- este
  # proyecto pasa por ciclos destroy/apply completos, no un entorno
  # productivo de largo plazo. Sin esto, `terraform destroy` falla en cuanto
  # el bucket tiene objetos (datos de DVC, artefactos de MLflow) o versiones
  # antiguas (versioning esta habilitado mas abajo); force_destroy vacia el
  # bucket, incluidas todas las versiones, antes de eliminarlo.
  force_destroy = true
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
      # SSE-S3 (clave gestionada por AWS, AES256) en vez de una CMK propia:
      # sigue cifrado en reposo; solo se pierde rotacion/key policy propias y
      # la atribucion de uso de clave en CloudTrail -- una CMK dedicada es la
      # mejora obvia si eso se vuelve un requisito.
      sse_algorithm = "AES256"
    }
  }
}

# El bucket guarda artefactos de modelos y datasets versionados con DVC:
# nada de eso debe ser jamas accesible publicamente. El bucket de estado de
# Terraform (terraform/bootstrap/main.tf) ya tenia este bloqueo; el de
# artefactos no -- trivy lo reporta como AWS-0094.
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
