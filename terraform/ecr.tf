# Unico repo ECR de este proyecto: MLflow corre desde la imagen publica
# ghcr.io/mlflow/mlflow (ver kubernetes/base/mlflow.yaml y docker-compose.yml)
# en vez de una imagen propia, asi que no hace falta un repo dedicado para
# ella. Tampoco hay un repo de cache remoto de BuildKit: el pipeline de CI
# (.gitlab-ci.yml) usa un `docker build` + `docker push` simple.
resource "aws_ecr_repository" "api_repo" {
  name = "${var.project_name}-api"
  # AWS-0031 (HIGH): tags inmutables. Con tags mutables, cualquiera con
  # permiso de push puede reemplazar el contenido de un tag ya desplegado
  # (incluido el commit SHA que el overlay de Kustomize tiene fijado) sin
  # dejar rastro. Con IMMUTABLE, un tag publicado es una referencia
  # permanente: re-publicar el mismo SHA falla en voz alta en vez de
  # sobrescribir en silencio.
  image_tag_mutability = "IMMUTABLE"
  # force_delete: este proyecto pasa por ciclos destroy/apply frecuentes
  # (entorno de aprendizaje, no produccion real) -- sin esto, "terraform
  # destroy" falla si el repo tiene alguna imagen publicada, y hay que
  # vaciarlo a mano con la AWS CLI antes de poder destruir.
  force_delete = true

  image_scanning_configuration {
    scan_on_push = true
  }

  # Cifrado con la clave por defecto de ECR (AES256 gestionada por AWS).
}

resource "aws_ecr_lifecycle_policy" "repo_cleanup" {
  repository = aws_ecr_repository.api_repo.name

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
  value = aws_ecr_repository.api_repo.repository_url
}
