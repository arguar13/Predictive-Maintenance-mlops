# ============================================================
# Secretos de terceros gestionados por Terraform
#
# ANTES: este archivo era un `data "aws_secretsmanager_secret"` que asumia
# que el secreto ya existia, creado a mano con
# `aws secretsmanager create-secret`. Eso rompia el principio de
# Infraestructura como Codigo (un `terraform apply` sobre una cuenta limpia
# fallaba con "Secrets Manager can't find the specified secret") y dejaba un
# recurso de produccion sin declarar en ningun sitio.
#
# AHORA: Terraform es dueño del secreto. El valor del token NO se commitea:
# se pasa como variable sensible via terraform.tfvars (gitignoreado, ver
# .gitignore: *.tfvars) o via la variable de entorno TF_VAR_git_repo_token.
#
# El nombre pasa de "github-token" a "git-token": el repositorio de este
# proyecto esta en GitLab (ver gitops/argocd/application.yaml y
# .gitlab-ci.yml), no en GitHub. El nombre anterior mandaba a cualquiera que
# operase esto a buscar un token del sitio equivocado. Como todavia no hay
# nada desplegado, el renombrado no tiene coste de migracion.
# ============================================================

variable "git_repo_token" {
  type        = string
  description = "Personal Access Token del repositorio Git (GitLab), sincronizado al cluster via ExternalSecret. Pasar por terraform.tfvars o TF_VAR_git_repo_token; nunca commitear."
  sensitive   = true
}

resource "aws_secretsmanager_secret" "git_token" {
  name        = "${var.project_name}/git-token"
  description = "Personal Access Token del repositorio Git, consumido por el ExternalSecret 'mlops-secrets' (clave GIT_REPO_TOKEN)"

  # En dev, borrado inmediato: con el default de 30 dias, un destroy+apply
  # falla con "You can't create this secret because a secret with this name
  # is already scheduled for deletion".
  recovery_window_in_days = var.environment == "dev" ? 0 : 30
  # CMK del proyecto (terraform/kms.tf), no la clave por defecto de
  # Secrets Manager.
  kms_key_id = aws_kms_key.mlops.arn
}

resource "aws_secretsmanager_secret_version" "git_token" {
  secret_id     = aws_secretsmanager_secret.git_token.id
  secret_string = jsonencode({ token = var.git_repo_token })
}
