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
# Credenciales de RDS generadas por Terraform (nunca elegidas ni commiteadas
# a mano). El valor sale directo de `terraform output` (ver mas abajo,
# marcado sensitive) y se pega a mano en un Secret de Kubernetes plano
# (kubernetes/base/secret.yaml) -- documentado en el README, seccion
# "Desplegar". Un secret manager dedicado con rotacion automatica es la
# mejora obvia si ese flujo manual deja de ser aceptable.
# ------------------------------------------------------------------
resource "random_password" "db_password" {
  length  = 32
  special = false # simplifica el parseo en connection strings (postgresql://user:pass@host)
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

  # Instancia unica, sin Multi-AZ ni replicas de lectura: coherente con el
  # alcance actual del proyecto, no un despliegue de alta disponibilidad.
  # Multi-AZ es la mejora obvia si la disponibilidad se vuelve un requisito.
  multi_az = false

  # Cifrado en reposo con la clave por defecto de AWS (AES256 gestionada por
  # AWS), no una CMK propia: sigue cifrado, solo que sin key policy ni
  # rotacion propias que gestionar.
  storage_encrypted = true

  backup_retention_period = var.environment == "prod" ? 30 : 7
  deletion_protection     = var.environment == "prod" ? true : false
}

output "rds_endpoint" {
  # .address (solo hostname), no .endpoint ("host:port"): el unico
  # consumidor es kubernetes/base/configmap.yaml -> DB_HOST, y
  # kubernetes/base/mlflow.yaml ya concatena ":5432" al construir el
  # connection string. Con .endpoint el resultado habria sido
  # "host:5432:5432", una URI invalida.
  value = aws_db_instance.mlflow_db.address
}

output "db_username" {
  value = var.db_username
}

output "db_password" {
  description = "Password de RDS generado por Terraform. Copialo al Secret de Kubernetes (ver kubernetes/base/secret.yaml y README.md)."
  value       = random_password.db_password.result
  sensitive   = true
}
