# ============================================================
# Identidad de despliegue (IaC), para dejar de usar la cuenta root
#
# AWS documenta explicitamente que la cuenta root NO debe tener access keys
# y no debe usarse para trabajo operativo: no se puede acotar con politicas,
# no se puede revocar sin cambiar la contraseña de la cuenta, y toda su
# actividad en CloudTrail aparece como "root" sin atribucion util.
#
# Este bootstrap es el UNICO punto del proyecto que se aplica con
# credenciales root, y su proposito es precisamente eliminar esa necesidad:
# crea un usuario IAM con su propia access key, que es la identidad usada
# por `terraform/` (el stack real) y por la CLI a partir de ese momento.
#
# Se modela como grupo + usuario (no politica inline en el usuario) para
# que dar de alta a un segundo operador sea una linea, no una copia de la
# politica.
# ============================================================

resource "aws_iam_group" "deployers" {
  name = "${var.project_name}-deployers"
  path = "/mlops/"
}

# AdministratorAccess: el stack de terraform/ provisiona VPC, EKS (+ IRSA),
# RDS, S3, ECR, Secrets Manager, CloudWatch Logs y crea roles/politicas IAM
# y un OIDC provider. Acotar eso a mano produce una
# politica gigante, fragil y que hay que reeditar con cada `terraform apply`.
# El limite real de blast radius aqui no es la politica, es que esta
# identidad SI se puede rotar, auditar por nombre en CloudTrail y revocar en
# segundos -- ninguna de las tres cosas es cierta para root.
resource "aws_iam_group_policy_attachment" "deployers_admin" {
  group      = aws_iam_group.deployers.name
  policy_arn = "arn:aws:iam::aws:policy/AdministratorAccess"
}

resource "aws_iam_user" "deployer" {
  name = "${var.project_name}-deployer"
  path = "/mlops/"

  tags = {
    Project   = var.project_name
    ManagedBy = "Terraform"
    # Sin ";": los valores de tag de IAM solo admiten
    # [\p{L}\p{Z}\p{N}_.:/=+\-@] -- un punto y coma hace fallar CreateUser
    # con ValidationError.
    Purpose = "Identidad de despliegue IaC. Sustituye el uso de la cuenta root"
  }
}

resource "aws_iam_user_group_membership" "deployer" {
  user   = aws_iam_user.deployer.name
  groups = [aws_iam_group.deployers.name]
}

resource "aws_iam_access_key" "deployer" {
  user = aws_iam_user.deployer.name
}

# ------------------------------------------------------------------
# Outputs
#
# NOTA DE SEGURIDAD: el secret queda en el estado de Terraform de ESTE
# directorio (bootstrap), que es local y esta gitignoreado (*.tfstate).
# Es la contrapartida inevitable de crear una credencial estatica por IaC.
# Rotacion: `terraform taint aws_iam_access_key.deployer && terraform apply`.
# ------------------------------------------------------------------
output "deployer_user_name" {
  description = "Usuario IAM a usar en lugar de root para terraform/ y la AWS CLI"
  value       = aws_iam_user.deployer.name
}

output "deployer_access_key_id" {
  description = "AWS_ACCESS_KEY_ID del usuario de despliegue"
  value       = aws_iam_access_key.deployer.id
}

output "deployer_secret_access_key" {
  description = "AWS_SECRET_ACCESS_KEY del usuario de despliegue"
  value       = aws_iam_access_key.deployer.secret
  sensitive   = true
}
