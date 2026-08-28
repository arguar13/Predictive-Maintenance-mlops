# Amazon MSK: el kubernetes/01-config.yaml original ya referenciaba un
# KAFKA_BROKER de MSK ("b-1.tustreamingcluster...") pero ningun .tf lo
# provisionaba -- streaming-consumer/producer_sim nunca hubieran tenido un
# broker real al que conectarse. Ver kubernetes/base/configmap.yaml para
# como se inyecta `msk_bootstrap_brokers` (output de abajo) en el cluster.
resource "aws_security_group" "msk_sg" {
  name        = "${var.project_name}-msk-sg"
  description = "Permitir trafico Kafka (TLS, 9094) exclusivamente desde EKS"
  vpc_id      = module.vpc.vpc_id

  ingress {
    # Puerto 9094 = listener TLS de MSK. Con client_broker = "TLS" el 9092
    # (plaintext) ni siquiera esta abierto en el broker; dejar la regla
    # apuntando a 9092 habria producido timeouts de conexion sin causa
    # aparente.
    description     = "Kafka broker (TLS)"
    from_port       = 9094
    to_port         = 9094
    protocol        = "tcp"
    security_groups = [module.eks.node_security_group_id]
  }

  ingress {
    description     = "Zookeeper (herramientas de administracion)"
    from_port       = 2181
    to_port         = 2181
    protocol        = "tcp"
    security_groups = [module.eks.node_security_group_id]
  }
}

resource "aws_msk_cluster" "kafka" {
  cluster_name           = "${var.project_name}-kafka"
  kafka_version          = var.msk_kafka_version
  number_of_broker_nodes = length(module.vpc.private_subnets) # 1 broker por AZ privada

  broker_node_group_info {
    instance_type   = var.msk_instance_type
    client_subnets  = module.vpc.private_subnets
    security_groups = [aws_security_group.msk_sg.id]

    storage_info {
      ebs_storage_info {
        volume_size = var.msk_ebs_volume_size
      }
    }
  }

  encryption_info {
    # AWS-0179 (HIGH): sin esta clave, MSK cifra en reposo con una clave
    # gestionada por AWS. Se usa la misma CMK del proyecto (terraform/kms.tf).
    encryption_at_rest_kms_key_arn = aws_kms_key.mlops.arn

    encryption_in_transit {
      # AWS-0073 (HIGH): antes era "PLAINTEXT", que deja toda la telemetria
      # y las alertas de inferencia en claro para cualquiera con acceso de
      # red a la VPC. Ahora TLS obligatorio: los clientes de kafka-python
      # (core_ml/streaming/*.py) se conectan con security_protocol=SSL --
      # ver streaming/kafka_security.py y la variable KAFKA_SECURITY_PROTOCOL
      # de kubernetes/base/configmap.yaml. MSK usa certificados de Amazon
      # Trust Services, ya presentes en el almacen de CAs del sistema: no
      # hace falta distribuir ningun truststore.
      # Siguiente endurecimiento recomendado: autenticacion IAM
      # (aws-msk-iam-auth) ademas del cifrado.
      client_broker = "TLS"
      in_cluster    = true
    }
  }

  logging_info {
    broker_logs {
      cloudwatch_logs {
        enabled   = true
        log_group = aws_cloudwatch_log_group.msk.name
      }
    }
  }
}

resource "aws_cloudwatch_log_group" "msk" {
  name              = "/aws/msk/${var.project_name}"
  retention_in_days = 14
}

# Con client_broker = "TLS" el cluster NO expone el listener 9092 en claro:
# el output bootstrap_brokers (plaintext) queda vacio y usarlo daria un
# error de conexion opaco. El endpoint correcto es el TLS (puerto 9094).
output "msk_bootstrap_brokers" {
  description = "Usar como KAFKA_BROKER en kubernetes/base/configmap.yaml (listener TLS, puerto 9094)"
  value       = aws_msk_cluster.kafka.bootstrap_brokers_tls
}
