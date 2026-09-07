variable "aws_region" {
  type    = string
  default = "us-east-1"
}

variable "environment" {
  type    = string
  default = "dev"
}

variable "project_name" {
  type    = string
  default = "predictive-maintenance-mlops"
}

variable "cluster_version" {
  type    = string
  default = "1.36"
}

# CIDR desde las que se puede alcanzar el endpoint publico del api-server
# de EKS. Por defecto, la IP publica del operador que provisiono el cluster.
# Si tu IP cambia (ISP dinamico, VPN), actualizala aqui y vuelve a aplicar,
# o `kubectl` empezara a dar timeout.
variable "cluster_public_access_cidrs" {
  type        = list(string)
  description = "CIDR permitidas para el endpoint publico del api-server de EKS. NUNCA 0.0.0.0/0."
  default     = ["181.199.150.181/32"]
}

variable "vpc_cidr" {
  type    = string
  default = "10.0.0.0/16"
}

variable "db_username" {
  type        = string
  description = "Usuario administrador de RDS"
  default     = "mlflow_admin"
}

# ------------------------------------------------------------------
# OIDC de GitLab CI/CD (ver terraform/iam.tf: aws_iam_openid_connect_provider.gitlab)
# ------------------------------------------------------------------
variable "gitlab_oidc_host" {
  type        = string
  description = "Host del issuer OIDC de GitLab (gitlab.com, o el dominio de tu instancia self-managed)"
  default     = "gitlab.com"
}

variable "gitlab_project_path" {
  type        = string
  description = "namespace/proyecto de GitLab que puede asumir GitLabCIRole (claim 'sub' del id_token). Ajustar al proyecto real antes de aplicar."
  default     = "personal-group7745334/predictive-maintenance-mlops"
}
