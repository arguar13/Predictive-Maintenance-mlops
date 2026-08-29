# ============================================================
# Rol para acceso desde EKS a S3 + Secrets Manager (mlops-env:mlops-sa)
# ============================================================
module "iam_eks_role" {
  source    = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version   = "~> 5.30"
  role_name = "${var.project_name}-s3-access-role-v2"

  role_policy_arns = {
    policy = aws_iam_policy.s3_access_policy_v2.arn
  }

  oidc_providers = {
    main = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["mlops-env:mlops-sa"]
    }
  }
}

resource "aws_iam_policy" "s3_access_policy_v2" {
  name        = "${var.project_name}-s3-policy-v2"
  description = "Permisos granulares para que MLflow/API/consumer en EKS accedan a S3 y Secrets Manager"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid = "S3ArtifactsAccess"
        Action = [
          "s3:ListBucket",
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject"
        ]
        Effect = "Allow"
        Resource = [
          aws_s3_bucket.mlflow_artifacts.arn,
          "${aws_s3_bucket.mlflow_artifacts.arn}/*"
        ]
      },
      {
        # Consumido por el ExternalSecret de kubernetes/base/externalsecret.yaml
        # (ClusterSecretStore autenticado via este mismo rol IRSA).
        Sid = "SecretsManagerRead"
        Action = [
          "secretsmanager:GetSecretValue",
          "secretsmanager:DescribeSecret"
        ]
        Effect = "Allow"
        Resource = [
          aws_secretsmanager_secret.db_credentials.arn,
          aws_secretsmanager_secret.git_token.arn
        ]
      },
      {
        # Sin esto, con el bucket y los secretos cifrados con la CMK del
        # proyecto (terraform/kms.tf), cada s3:GetObject/PutObject y cada
        # secretsmanager:GetSecretValue falla con AccessDenied: KMS exige
        # permiso explicito sobre la clave ADEMAS del permiso sobre el
        # recurso.
        Sid = "KmsProjectKey"
        Action = [
          "kms:Decrypt",
          "kms:GenerateDataKey",
          "kms:DescribeKey"
        ]
        Effect   = "Allow"
        Resource = [aws_kms_key.mlops.arn]
      }
    ]
  })
}

# ============================================================
# OIDC federado para GitLab CI/CD (sin credenciales estaticas de larga
# duracion): GitLab emite un id_token JWT (ver .gitlab-ci.yml, id_tokens:
# AWS_OIDC_TOKEN) que este proveedor permite canjear via
# sts:AssumeRoleWithWebIdentity.
# ============================================================
data "tls_certificate" "gitlab" {
  url = "https://${var.gitlab_oidc_host}"
}

resource "aws_iam_openid_connect_provider" "gitlab" {
  url             = "https://${var.gitlab_oidc_host}"
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = [data.tls_certificate.gitlab.certificates[0].sha1_fingerprint]
}

# ============================================================
# Rol para GitLab CI/CD (GitLabCIRole)
#
# ANTES: assume_role_policy confiaba en "Service: ec2.amazonaws.com", pero
# .gitlab-ci.yml usa `aws sts assume-role-with-web-identity` con un token
# OIDC de GitLab -- esa combinacion NUNCA pudo autenticar (un trust policy
# de servicio EC2 no acepta AssumeRoleWithWebIdentity). Ahora el trust
# policy es el OIDC provider de arriba, con el "sub" del token limitado a
# este proyecto de GitLab.
# ============================================================
resource "aws_iam_role" "gitlab_ci_role" {
  name = "GitLabCIRole"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Federated = aws_iam_openid_connect_provider.gitlab.arn
        }
        Action = "sts:AssumeRoleWithWebIdentity"
        Condition = {
          StringEquals = {
            "${var.gitlab_oidc_host}:aud" = "sts.amazonaws.com"
          }
          StringLike = {
            # Ajusta a ":ref:main" para restringir a la rama por defecto en
            # produccion; con "*" cualquier rama/MR de este proyecto puede
            # asumir el rol (razonable mientras se valida el pipeline).
            "${var.gitlab_oidc_host}:sub" = "project_path:${var.gitlab_project_path}:*"
          }
        }
      }
    ]
  })
}

# Politica propia y acotada: SOLO push/pull a los repositorios ECR de este
# proyecto y acceso al bucket de DVC/MLflow. Ya NO incluye permisos de EKS:
# con GitOps (ArgoCD), CI deja de tocar el cluster directamente (ver
# gitops/argocd/application.yaml), asi que ya no necesita
# AmazonEKSClusterPolicy -- reduce la superficie de lo que un pipeline
# comprometido podria hacer.
resource "aws_iam_policy" "gitlab_ci_policy" {
  name        = "${var.project_name}-gitlab-ci-policy"
  description = "Permisos minimos para el pipeline de CI: push/pull de ECR y acceso al bucket de artefactos"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "EcrAuth"
        Effect   = "Allow"
        Action   = ["ecr:GetAuthorizationToken"]
        Resource = "*"
      },
      {
        # ANTES solo listaba streaming_repo/consumer_repo: build_image hace
        # push de las CUATRO imagenes (api/streaming, consumer, monitoring,
        # mlflow), pero nunca habia llegado a intentar el push de
        # monitoring/mlflow via este rol (se quedaba sin ancho de banda en
        # el push de api, mucho antes) -- hubiera fallado con AccessDenied
        # en cuanto el pipeline llegara ahi. build_cache: repo de cache
        # remoto de BuildKit (docker buildx --cache-from/--cache-to), no una
        # imagen de release.
        Sid    = "EcrPushPull"
        Effect = "Allow"
        Action = [
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
          "ecr:PutImage",
          "ecr:InitiateLayerUpload",
          "ecr:UploadLayerPart",
          "ecr:CompleteLayerUpload",
          # Sin esto, el chequeo skip-if-exists de `make docker-buildx-push-*`
          # (Makefile: `aws ecr describe-images` antes de cada push, necesario
          # porque los repos son IMMUTABLE y un retry de build_image no puede
          # re-publicar una imagen ya subida en un intento previo) fallaba con
          # AccessDenied -- indistinguible en el `if` de un "tag no existe",
          # asi que el reintento SIEMPRE volvia a intentar el push y reventaba
          # contra el mismo error de tag inmutable que se queria evitar.
          "ecr:DescribeImages"
        ]
        Resource = [
          aws_ecr_repository.streaming_repo.arn,
          aws_ecr_repository.consumer_repo.arn,
          aws_ecr_repository.monitoring_repo.arn,
          aws_ecr_repository.mlflow_repo.arn,
          aws_ecr_repository.build_cache.arn
        ]
      },
      {
        Sid    = "S3ArtifactsAccess"
        Effect = "Allow"
        Action = [
          "s3:ListBucket",
          "s3:GetObject",
          "s3:PutObject"
        ]
        Resource = [
          aws_s3_bucket.mlflow_artifacts.arn,
          "${aws_s3_bucket.mlflow_artifacts.arn}/*"
        ]
      },
      {
        # Sin esto, con el bucket y los secretos cifrados con la CMK del
        # proyecto (terraform/kms.tf), cada s3:GetObject/PutObject y cada
        # secretsmanager:GetSecretValue falla con AccessDenied: KMS exige
        # permiso explicito sobre la clave ADEMAS del permiso sobre el
        # recurso.
        Sid = "KmsProjectKey"
        Action = [
          "kms:Decrypt",
          "kms:GenerateDataKey",
          "kms:DescribeKey"
        ]
        Effect   = "Allow"
        Resource = [aws_kms_key.mlops.arn]
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "gitlab_ci_policy_attachment" {
  role       = aws_iam_role.gitlab_ci_role.name
  policy_arn = aws_iam_policy.gitlab_ci_policy.arn
}

output "gitlab_ci_role_arn" {
  value = aws_iam_role.gitlab_ci_role.arn
}
