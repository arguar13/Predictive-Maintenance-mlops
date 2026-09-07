# ============================================================
# API key de /predict (api/main.py::require_api_key)
#
# Credencial de primer partido: no hay nada que un humano deba "saber" de
# antemano (a diferencia de un PAT de un servicio externo), asi que se
# genera con random_password, el mismo patron ya usado para la contraseña
# de RDS (ver rds.tf) - nadie elige ni commitea un valor, Terraform es la
# unica fuente de verdad.
# ============================================================

resource "random_password" "api_key" {
  length  = 40
  special = false # va en un header HTTP (X-API-Key); evita tener que citar/escapar
}

resource "aws_secretsmanager_secret" "api_key" {
  name        = "${var.project_name}/api-key"
  description = "API key de autenticacion para /predict, consumida por el ExternalSecret 'mlops-secrets' (clave API_KEY)"

  recovery_window_in_days = var.environment == "dev" ? 0 : 30
  kms_key_id              = aws_kms_key.mlops.arn
}

resource "aws_secretsmanager_secret_version" "api_key" {
  secret_id     = aws_secretsmanager_secret.api_key.id
  secret_string = jsonencode({ api_key = random_password.api_key.result })
}
