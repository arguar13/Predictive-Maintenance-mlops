module "iam_eks_role" {
  source    = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version   = "~> 5.30"
  role_name = "${var.project_name}-s3-access-role-v2"   # <- cambiado

  role_policy_arns = {
    policy = aws_iam_policy.s3_access_policy_v2.arn     # <- referencia al nuevo recurso
  }

  oidc_providers = {
    main = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["mlops-env:mlops-sa"]
    }
  }
}

resource "aws_iam_policy" "s3_access_policy_v2" {       # <- nombre del recurso cambiado
  name        = "${var.project_name}-s3-policy-v2"      # <- cambiado
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
        Effect = "Allow"
        Resource = [
          aws_s3_bucket.mlflow_artifacts.arn,
          "${aws_s3_bucket.mlflow_artifacts.arn}/*"
        ]
      }
    ]
  })
}