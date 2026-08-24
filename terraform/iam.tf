# ============================================================
# Rol para acceso desde EKS a S3 (mlops-env:mlops-sa)
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
  description = "Permisos granulares para que MLflow en EKS acceda a S3"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "s3:ListBucket",
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject"
        ]
        Effect   = "Allow"
        Resource = [
          aws_s3_bucket.mlflow_artifacts.arn,
          "${aws_s3_bucket.mlflow_artifacts.arn}/*"
        ]
      }
    ]
  })
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
          Service = "ec2.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })
}

# Adjuntar políticas administradas de AWS para ECR, EKS y S3
resource "aws_iam_role_policy_attachment" "gitlab_ci_ecr" {
  role       = aws_iam_role.gitlab_ci_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}

resource "aws_iam_role_policy_attachment" "gitlab_ci_eks" {
  role       = aws_iam_role.gitlab_ci_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
}

resource "aws_iam_role_policy_attachment" "gitlab_ci_s3" {
  role       = aws_iam_role.gitlab_ci_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
}