# ============================================================
# Customer Managed Key (CMK) del proyecto
#
# Motivo: `make security` (trivy, mismo comando que corre el stage "quality"
# de .gitlab-ci.yml) fallaba con dos hallazgos de la misma familia:
#
#   AWS-0132 (HIGH) terraform/s3.tf   -- el bucket de artefactos cifraba con
#                                        SSE-S3 (AES256, clave de AWS)
#   AWS-0098 (LOW)  terraform/secrets.tf -- los secretos usaban la clave por
#                                        defecto de Secrets Manager
#
# Con una clave gestionada por AWS no se controla la rotacion, no se puede
# acotar quien descifra mediante una key policy, y el uso de la clave no
# queda atribuido en CloudTrail. Una CMK unica de proyecto resuelve las tres
# cosas y cuesta ~1 USD/mes.
#
# NOTA sobre el bucket de artefactos: `bucket_key_enabled = true` en s3.tf
# hace que S3 use una clave de bucket intermedia, reduciendo las llamadas a
# KMS (y su coste) en varios ordenes de magnitud para cargas con muchos
# objetos pequeños -- que es exactamente el patron de MLflow.
# ============================================================

resource "aws_kms_key" "mlops" {
  description             = "CMK de ${var.project_name}: artefactos S3 (MLflow/DVC/Feast) y secretos de Secrets Manager"
  deletion_window_in_days = var.environment == "dev" ? 7 : 30
  enable_key_rotation     = true
}

resource "aws_kms_alias" "mlops" {
  name          = "alias/${var.project_name}"
  target_key_id = aws_kms_key.mlops.key_id
}

output "kms_key_arn" {
  value = aws_kms_key.mlops.arn
}
