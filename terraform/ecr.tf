# NOTA: el nombre "-streaming" es historico (ya existe en AWS con este
# nombre); en la practica esta imagen es la de la API (ver Dockerfile en la
# raiz y kubernetes/base/api.yaml). No se renombra para no forzar un
# destroy+recreate de un recurso ya provisionado -- el consumer real de
# streaming ahora tiene su propio repositorio, ver mas abajo.
resource "aws_ecr_repository" "streaming_repo" {
  name = "${var.project_name}-streaming"
  # AWS-0031 (HIGH): tags inmutables. Con tags mutables, cualquiera con
  # permiso de push puede reemplazar el contenido de un tag ya desplegado
  # (incluido el commit SHA que ArgoCD tiene fijado en el overlay) sin dejar
  # rastro. Con IMMUTABLE, un tag publicado es una referencia permanente:
  # re-publicar el mismo SHA falla en voz alta en vez de sobrescribir en
  # silencio. Encaja con el esquema de tags del proyecto, que ya usa el
  # commit SHA (Makefile: IMAGE_TAG), nunca "latest", en un release real.
  image_tag_mutability = "IMMUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  # AWS-0033: cifrado con la CMK del proyecto en vez de la clave gestionada
  # por AWS, igual que S3 y Secrets Manager.
  encryption_configuration {
    encryption_type = "KMS"
    kms_key         = aws_kms_key.mlops.arn
  }
}

resource "aws_ecr_lifecycle_policy" "repo_cleanup" {
  repository = aws_ecr_repository.streaming_repo.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Mantener solo las ultimas 10 imagenes"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 10
      }
      action = {
        type = "expire"
      }
    }]
  })
}

# Imagen del streaming-consumer (core_ml/Dockerfile): antes no existia y el
# Deployment de streaming-consumer apuntaba a la imagen de la API, que
# nunca incluyo core_ml/streaming/ ni sus dependencias -> crashloop.
resource "aws_ecr_repository" "consumer_repo" {
  name = "${var.project_name}-consumer"
  # AWS-0031 (HIGH): tags inmutables. Con tags mutables, cualquiera con
  # permiso de push puede reemplazar el contenido de un tag ya desplegado
  # (incluido el commit SHA que ArgoCD tiene fijado en el overlay) sin dejar
  # rastro. Con IMMUTABLE, un tag publicado es una referencia permanente:
  # re-publicar el mismo SHA falla en voz alta en vez de sobrescribir en
  # silencio. Encaja con el esquema de tags del proyecto, que ya usa el
  # commit SHA (Makefile: IMAGE_TAG), nunca "latest", en un release real.
  image_tag_mutability = "IMMUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  # AWS-0033: cifrado con la CMK del proyecto en vez de la clave gestionada
  # por AWS, igual que S3 y Secrets Manager.
  encryption_configuration {
    encryption_type = "KMS"
    kms_key         = aws_kms_key.mlops.arn
  }
}

resource "aws_ecr_lifecycle_policy" "consumer_repo_cleanup" {
  repository = aws_ecr_repository.consumer_repo.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Mantener solo las ultimas 10 imagenes"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 10
      }
      action = {
        type = "expire"
      }
    }]
  })
}

# Imagen del servicio de deteccion de drift (monitoring/Dockerfile): antes
# no tenia imagen propia ni Deployment -- kubernetes/base/prometheus.yml
# scrapeaba un target ("evidently_service:8000") que nunca llegaba a existir.
resource "aws_ecr_repository" "monitoring_repo" {
  name                 = "${var.project_name}-monitoring"
  image_tag_mutability = "IMMUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "KMS"
    kms_key         = aws_kms_key.mlops.arn
  }
}

resource "aws_ecr_lifecycle_policy" "monitoring_repo_cleanup" {
  repository = aws_ecr_repository.monitoring_repo.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Mantener solo las ultimas 10 imagenes"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 10
      }
      action = {
        type = "expire"
      }
    }]
  })
}

output "ecr_api_repository_url" {
  value = aws_ecr_repository.streaming_repo.repository_url
}

output "ecr_consumer_repository_url" {
  value = aws_ecr_repository.consumer_repo.repository_url
}

output "ecr_monitoring_repository_url" {
  value = aws_ecr_repository.monitoring_repo.repository_url
}
