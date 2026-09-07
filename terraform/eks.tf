module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.0"
  # "mlops-cluster" (nombre generico, sin el prefijo var.project_name que usa
  # el resto de los recursos de este proyecto -- RDS, S3, ECR, KMS) colisiono
  # con otro proyecto de la misma cuenta AWS que sigue una
  # plantilla de curso similar: su `aws eks update-kubeconfig --name
  # mlops-cluster` se conecto a ESTE cluster (ya existente) en vez de crear
  # uno propio, y su ArgoCD borro el namespace "argocd" (incluida la
  # Application de este proyecto) al instalarse pensando que el cluster
  # estaba vacio. var.project_name (unico) sin sufijo "-cluster" adicional:
  # el modulo ya agrega su propio sufijo "-cluster-" al name_prefix del rol
  # IAM del cluster, con un limite de 38 caracteres -- "${var.project_name}-
  # cluster" + ese sufijo interno lo supera.
  cluster_name    = var.project_name
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

  # El modulo v20 NO le da admin al identity que corre `terraform apply` por
  # defecto (a diferencia de versiones anteriores): sin esto, `aws eks
  # update-kubeconfig` + `kubectl get nodes` responde "the server has asked
  # for the client to provide credentials" pese a que el cluster esta ACTIVE
  # y la cuenta AWS es la correcta -- el problema es autorizacion dentro de
  # Kubernetes (RBAC/access entries), no autenticacion contra AWS. Esto crea
  # un access entry declarativo (API_AND_CONFIG_MAP, modo por defecto del
  # modulo) para quien aplique Terraform.
  enable_cluster_creator_admin_permissions = true

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
