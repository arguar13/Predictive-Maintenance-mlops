# Security Group para RDS
resource "aws_security_group" "rds_sg" {
  name        = "${var.project_name}-rds-sg"
  description = "Permitir trafico PostgreSQL exclusivamente desde EKS"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [module.eks.node_security_group_id]
  }
}

# Lectura del secreto en Secrets Manager
data "aws_secretsmanager_secret" "db_password" {
  name = "proeycto-mlops11/db-password" # nombre exacto de tu secreto
}

data "aws_secretsmanager_secret_version" "db_password" {
  secret_id = data.aws_secretsmanager_secret.db_password.id
}

# Inyección del secreto en un local
locals {
  db_password = jsondecode(data.aws_secretsmanager_secret_version.db_password.secret_string)["db_password"]
}

# Instancia RDS
resource "aws_db_instance" "mlflow_db" {
  identifier             = "${var.project_name}-backend-db"
  instance_class         = "db.t3.micro"
  allocated_storage      = 20
  engine                 = "postgres"
  engine_version         = "18.3" 
  username               = var.db_username
  password               = local.db_password
  db_subnet_group_name   = module.vpc.database_subnet_group_name
  vpc_security_group_ids = [aws_security_group.rds_sg.id]
  skip_final_snapshot    = var.environment == "dev" ? true : false
  multi_az               = var.environment == "prod" ? true : false
}

output "rds_endpoint" {
  value = aws_db_instance.mlflow_db.endpoint
}