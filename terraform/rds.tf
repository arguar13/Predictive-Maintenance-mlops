# Security Group para RDS
resource "aws_security_group" "rds_sg" {
  name        = "${var.project_name}-rds-sg"
  description = "Permitir trafico PostgreSQL exclusivamente desde EKS"
  vpc_id      = module.vpc.vpc_id

  ingress {
    description     = "PostgreSQL desde los nodos de EKS (MLflow backend store)"
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [module.eks.node_security_group_id]
  }
}

# ------------------------------------------------------------------
# Credenciales de RDS generadas y versionadas por Terraform (antes: un
# `data "aws_secretsmanager_secret"` que leia un secreto pre-creado a mano
# con un nombre mal escrito -- "proeycto-mlops11/db-password", que no
# coincidia con ningun secreto real). Ahora Terraform es la unica fuente de
# verdad: genera la contraseña, crea el secreto, y kubernetes/base/
# externalsecret.yaml lo sincroniza al cluster (mismo nombre que aqui).
# ------------------------------------------------------------------
resource "random_password" "db_password" {
  length  = 32
  special = false # simplifica el parseo en connection strings (postgresql://user:pass@host)
}

resource "aws_secretsmanager_secret" "db_credentials" {
  name        = "${var.project_name}/db-credentials"
  description = "Credenciales de RDS PostgreSQL (backend store de MLflow), sincronizadas a K8s via ExternalSecret"

  # Misma CMK de proyecto que el resto de secretos y artefactos (kms.tf).
  kms_key_id = aws_kms_key.mlops.arn

  # En dev, borrado inmediato: con el default de 30 dias un destroy+apply
  # falla con "secret ... is already scheduled for deletion".
  recovery_window_in_days = var.environment == "dev" ? 0 : 30
}

resource "aws_secretsmanager_secret_version" "db_credentials" {
  secret_id = aws_secretsmanager_secret.db_credentials.id
  secret_string = jsonencode({
    username = var.db_username
    password = random_password.db_password.result
    dbname   = "mlflow_db"
    host     = aws_db_instance.mlflow_db.address
    port     = 5432
  })
}

# Instancia RDS
resource "aws_db_instance" "mlflow_db" {
  identifier             = "${var.project_name}-backend-db"
  instance_class         = "db.t3.micro"
  allocated_storage      = 20
  engine                 = "postgres"
  engine_version         = "18.3"
  username               = var.db_username
  password               = random_password.db_password.result
  db_name                = "mlflow_db"
  db_subnet_group_name   = module.vpc.database_subnet_group_name
  vpc_security_group_ids = [aws_security_group.rds_sg.id]
  skip_final_snapshot    = var.environment == "dev" ? true : false
  multi_az               = var.environment == "prod" ? true : false

  # AWS-0080 (HIGH): sin esto el volumen de la instancia queda SIN cifrar.
  # Aqui vive el backend store de MLflow: nombres de modelos, metricas,
  # parametros y rutas a los artefactos de cada run.
  storage_encrypted = true
  kms_key_id        = aws_kms_key.mlops.arn

  # AWS-0077: sin retencion de backups no hay point-in-time recovery. 7 dias
  # en dev, 30 en prod.
  backup_retention_period = var.environment == "prod" ? 30 : 7

  # AWS-0176: permite autenticarse contra la BD con credenciales IAM
  # temporales (IRSA) en vez de solo con usuario/contraseña estaticos.
  iam_database_authentication_enabled = true

  # AWS-0177: en prod, un `terraform destroy` accidental no puede borrar la
  # base de datos. En dev queda desactivado para poder reciclar el entorno.
  deletion_protection = var.environment == "prod" ? true : false

  # AWS-0133: Performance Insights para diagnosticar consultas lentas.
  performance_insights_enabled    = true
  performance_insights_kms_key_id = aws_kms_key.mlops.arn
}

output "rds_endpoint" {
  value = aws_db_instance.mlflow_db.endpoint
}

output "db_credentials_secret_name" {
  value = aws_secretsmanager_secret.db_credentials.name
}
