# ElastiCache Redis: Feast usa Redis como Online Store (ver
# core_ml/feature_store/feature_store.yaml), pero su connection_string
# estaba hardcodeada a "localhost:6379" -- nunca hubiera funcionado fuera
# de un laptop. Ahora feature_store.yaml lee ${REDIS_CONNECTION_STRING} y
# kubernetes/base/configmap.yaml la inyecta desde este output.
resource "aws_security_group" "redis_sg" {
  name        = "${var.project_name}-redis-sg"
  description = "Permitir trafico Redis exclusivamente desde EKS"
  vpc_id      = module.vpc.vpc_id

  ingress {
    description     = "Redis desde los nodos de EKS (Feast Online Store)"
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [module.eks.node_security_group_id]
  }
}

resource "aws_elasticache_subnet_group" "redis" {
  name       = "${var.project_name}-redis-subnet-group"
  subnet_ids = module.vpc.private_subnets
}

resource "aws_elasticache_cluster" "feast_online_store" {
  cluster_id         = "${var.project_name}-feast-redis"
  engine             = "redis"
  engine_version     = var.redis_engine_version
  node_type          = var.redis_node_type
  num_cache_nodes    = 1
  port               = 6379
  subnet_group_name  = aws_elasticache_subnet_group.redis.name
  security_group_ids = [aws_security_group.redis_sg.id]
  apply_immediately  = var.environment != "prod"

  # AWS-0050: sin retencion de snapshots, un fallo del nodo pierde todo el
  # Online Store. Es reconstruible con `feast materialize`, pero eso son
  # minutos de features no servibles.
  snapshot_retention_limit = var.environment == "prod" ? 7 : 1
}

output "redis_connection_string" {
  value = "${aws_elasticache_cluster.feast_online_store.cache_nodes[0].address}:${aws_elasticache_cluster.feast_online_store.cache_nodes[0].port}"
}
