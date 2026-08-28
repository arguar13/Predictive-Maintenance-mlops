terraform {
  required_version = ">= 1.5.0"

  # Backend remoto, creado por terraform/bootstrap/ (bucket S3 versionado y
  # cifrado con CMK + tabla DynamoDB de locking). Valores tomados
  # literalmente del output "backend_config" de ese modulo.
  #
  # El estado deja de vivir en el portatil: se acabaron los "a mi me
  # funciona" por tener un tfstate desincronizado, y el lock de DynamoDB
  # impide que dos applies simultaneos corrompan el estado.
  backend "s3" {
    bucket         = "predictive-maintenance-mlops-tfstate-040175285118"
    key            = "predictive-maintenance-mlops/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "predictive-maintenance-mlops-tfstate-locks"
    encrypt        = true
  }

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.0"
    }
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Environment = var.environment
      Project     = var.project_name
      ManagedBy   = "Terraform"
    }
  }
}