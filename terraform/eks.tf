module "eks" {
  source          = "terraform-aws-modules/eks/aws"
  version         = "~> 20.0"
  cluster_name    = "mlops-cluster"
  cluster_version = var.cluster_version # Toma el valor 1.36 de variables.tf
  vpc_id          = module.vpc.vpc_id
  subnet_ids      = module.vpc.private_subnets

  # Acceso a la API de K8s.
  #
  # AWS-0041 (CRITICAL): el modulo, por defecto, abre el endpoint publico a
  # 0.0.0.0/0 -- el servidor de API de Kubernetes accesible desde todo
  # Internet. Se restringe a las CIDR declaradas en var.cluster_public_access_cidrs.
  #
  # El endpoint sigue siendo publico (AWS-0040) de forma deliberada: hacerlo
  # privado obligaria a un bastion o VPN para cualquier `kubectl`, y ArgoCD
  # -- que es quien realmente despliega -- vive DENTRO del cluster y usa el
  # endpoint interno. Ver .trivyignore para la aceptacion documentada.
  cluster_endpoint_public_access       = true
  cluster_endpoint_public_access_cidrs = var.cluster_public_access_cidrs

  # AWS-0038: sin esto no hay traza de autenticacion/autorizacion ni del
  # api-server en CloudWatch; una investigacion post-incidente se queda sin
  # datos.
  cluster_enabled_log_types = ["api", "audit", "authenticator", "controllerManager", "scheduler"]

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