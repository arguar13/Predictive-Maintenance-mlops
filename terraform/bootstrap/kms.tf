# ============================================================
# CMK del backend de estado de Terraform
#
# El bucket de estado guarda terraform.tfstate en claro: endpoints, ARNs,
# la contraseña generada de RDS (random_password) y --  en este mismo
# directorio -- la secret access key del usuario de despliegue
# (iam_deployer.tf). Cifrarlo con SSE-S3 (clave de AWS) dejaba el hallazgo
# AWS-0132 (HIGH) de trivy y, mas importante, sin control sobre quien puede
# descifrarlo ni rotacion propia.
#
# Esta clave es independiente de la CMK del stack principal (terraform/kms.tf)
# a proposito: el bootstrap tiene que poder aplicarse en una cuenta vacia,
# antes de que exista ningun otro recurso.
# ============================================================

resource "aws_kms_key" "tfstate" {
  description             = "CMK del backend de estado de Terraform de ${var.project_name}"
  deletion_window_in_days = 30
  enable_key_rotation     = true
}

resource "aws_kms_alias" "tfstate" {
  name          = "alias/${var.project_name}-tfstate"
  target_key_id = aws_kms_key.tfstate.key_id
}
