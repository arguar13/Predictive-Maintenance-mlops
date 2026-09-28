# ============================================================
# Acceso de los nodos de EKS a S3 (artefactos de MLflow, remoto de DVC)
#
# El rol de instancia del node group (compartido por todos los pods que
# corren en esos nodos) recibe permisos directos sobre el bucket de
# artefactos via `iam_role_additional_policies` (ver terraform/eks.tf), en
# vez de un rol IRSA granular por Service Account. Es menos aislado --
# cualquier pod del nodo podria, en teoria, usar este permiso, no solo
# mlflow/api -- pero mas simple de operar y depurar. Un rol IRSA dedicado
# por servicio es la mejora obvia si el aislamiento por pod se vuelve un
# requisito.
# ============================================================
resource "aws_iam_policy" "node_s3_access" {
  name        = "${var.project_name}-node-s3-access"
  description = "Acceso de los nodos EKS al bucket de artefactos (MLflow/DVC)"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "S3ArtifactsAccess"
        Effect = "Allow"
        Action = [
          "s3:ListBucket",
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject"
        ]
        Resource = [
          aws_s3_bucket.mlflow_artifacts.arn,
          "${aws_s3_bucket.mlflow_artifacts.arn}/*"
        ]
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

# Politica del pipeline de CI: push/pull a ECR, lectura/escritura del bucket
# de artefactos (DVC), y permiso para autenticarse contra el cluster de EKS
# -- el stage "deploy" de .gitlab-ci.yml corre `kubectl apply -k` directo
# contra el cluster, asi que este rol necesita poder generar un kubeconfig
# valido.
resource "aws_iam_policy" "gitlab_ci_policy" {
  name        = "${var.project_name}-gitlab-ci-policy"
  description = "Permisos minimos para el pipeline de CI: ECR, S3 y despliegue a EKS"

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
        Sid    = "EcrPushPull"
        Effect = "Allow"
        Action = [
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
          "ecr:PutImage",
          "ecr:InitiateLayerUpload",
          "ecr:UploadLayerPart",
          "ecr:CompleteLayerUpload"
        ]
        Resource = [aws_ecr_repository.api_repo.arn]
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
        # `aws eks update-kubeconfig` (stage "deploy") necesita poder leer
        # los metadatos del cluster para armar el kubeconfig.
        Sid      = "EksDescribe"
        Effect   = "Allow"
        Action   = ["eks:DescribeCluster"]
        Resource = [module.eks.cluster_arn]
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "gitlab_ci_policy_attachment" {
  role       = aws_iam_role.gitlab_ci_role.name
  policy_arn = aws_iam_policy.gitlab_ci_policy.arn
}

# Autorizacion DENTRO de Kubernetes (RBAC) para que GitLabCIRole pueda
# desplegar: autenticarse contra AWS (arriba) no alcanza, EKS tambien exige
# un access entry explicito -- mismo mecanismo que
# enable_cluster_creator_admin_permissions en eks.tf usa para quien aplica
# Terraform. ClusterAdminPolicy es deliberadamente amplio para mantener esto
# simple; acotarlo a un Role/RoleBinding de namespace es la mejora obvia si
# el pipeline necesita permisos mas finos.
resource "aws_eks_access_entry" "gitlab_ci" {
  cluster_name  = module.eks.cluster_name
  principal_arn = aws_iam_role.gitlab_ci_role.arn
}

resource "aws_eks_access_policy_association" "gitlab_ci_admin" {
  cluster_name  = module.eks.cluster_name
  principal_arn = aws_iam_role.gitlab_ci_role.arn
  policy_arn    = "arn:aws:eks::aws:cluster-access-policy/AmazonEKSClusterAdminPolicy"

  access_scope {
    type = "cluster"
  }
}

output "gitlab_ci_role_arn" {
  value = aws_iam_role.gitlab_ci_role.arn
}
