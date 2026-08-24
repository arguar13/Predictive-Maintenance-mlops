module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.0"
  cluster_name    = "mlops-cluster"
  cluster_version = var.cluster_version # Toma el valor 1.36 de variables.tf
  vpc_id                   = module.vpc.vpc_id
  subnet_ids               = module.vpc.private_subnets
  
  # Acceso a la API de K8s (IPs permitidas pueden restringirse aquí)
  cluster_endpoint_public_access = true

  # Fundamental para seguridad MLOps (IRSA)
  enable_irsa = true

  eks_managed_node_groups = {
    ml_workers = {
      min_size       = 1
      max_size       = 3
      desired_size   = 2
      instance_types = ["t3.large"]
      disk_size      = 40  
    }
  }
}

output "eks_cluster_name" {
  value = module.eks.cluster_name
}