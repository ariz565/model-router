# Production topology for ModelRouter's data plane — the direct Terraform
# mapping of docker-compose.yml's four local services (redis/postgres/
# redpanda/localstack) onto real AWS-managed equivalents, matching exactly
# what store/redis_events.py, store/postgres_events.py,
# accounting/ledger.py, observability/async_publish.py, and
# tenancy/byok.py's KMS path each expect to connect to.
#
# Deliberately does NOT include: the VPC itself, the ModelRouter compute
# layer (see ecs.tf — a starting skeleton, not a full autoscaling story),
# or IAM policy documents beyond the one KMS key policy below. A reusable
# module that silently opinionates its caller's entire network and IAM
# posture is a worse default than one that is explicit about needing both
# handed to it.

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
  }
}

locals {
  name_prefix = "modelrouter-${var.environment}"
}

# ── Redis (ElastiCache) — store/redis_events.py's EventStore tier AND
# accounting/ledger.py's reservation-ledger fast path. Two separate
# replication groups on purpose (see .env.example's own comment: isolating
# the ledger's hot-path traffic from the event log's) — this module still
# only stands up ONE by default; duplicate the resource block below with a
# different name for the ledger if you want that isolation in a real deploy. ──

resource "aws_elasticache_subnet_group" "this" {
  name       = "${local.name_prefix}-redis"
  subnet_ids = var.private_subnet_ids
}

resource "aws_security_group" "redis" {
  name_prefix = "${local.name_prefix}-redis-"
  vpc_id      = var.vpc_id

  ingress {
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [var.app_security_group_id]   # ONLY the app's own SG -- never a CIDR block
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_elasticache_replication_group" "this" {
  replication_group_id       = "${local.name_prefix}-redis"
  description                 = "ModelRouter EventStore + reservation-ledger tier"
  node_type                   = var.redis_node_type
  num_cache_clusters           = 2   # primary + 1 replica -- automatic failover requires >= 2
  automatic_failover_enabled  = true
  engine                       = "redis"
  engine_version               = "7.1"
  port                         = 6379
  subnet_group_name           = aws_elasticache_subnet_group.this.name
  security_group_ids           = [aws_security_group.redis.id]
  at_rest_encryption_enabled  = true
  transit_encryption_enabled  = true   # requires TLS on the client -- redis.ConnectionPool.from_url("rediss://...")
  auto_minor_version_upgrade  = true
  snapshot_retention_limit     = 5
}

# ── Postgres (Aurora Serverless v2) — store/postgres_events.py's EventStore
# tier: the durable system of record every event-sourced subsystem's audit
# trail depends on. Serverless v2 (not a fixed instance) because event
# volume is inherently spiky with request traffic, and scaling to near-zero
# ACUs outside business hours is a real cost lever a fixed db.r6g.large
# instance can't give you. ──

resource "aws_security_group" "postgres" {
  name_prefix = "${local.name_prefix}-postgres-"
  vpc_id      = var.vpc_id

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [var.app_security_group_id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_db_subnet_group" "postgres" {
  name       = "${local.name_prefix}-postgres"
  subnet_ids = var.private_subnet_ids
}

resource "random_password" "postgres_master" {
  length  = 32
  special = false   # avoid characters psycopg's DSN parser would need escaping
}

resource "aws_secretsmanager_secret" "postgres_dsn" {
  name = "${local.name_prefix}-postgres-dsn"
}

resource "aws_secretsmanager_secret_version" "postgres_dsn" {
  secret_id = aws_secretsmanager_secret.postgres_dsn.id
  secret_string = "postgresql://modelrouter:${random_password.postgres_master.result}@${aws_rds_cluster.postgres.endpoint}:5432/modelrouter"
}

resource "aws_rds_cluster" "postgres" {
  cluster_identifier     = "${local.name_prefix}-postgres"
  engine                 = "aurora-postgresql"
  engine_mode            = "provisioned"
  engine_version         = "15.4"
  database_name          = "modelrouter"
  master_username        = "modelrouter"
  master_password        = random_password.postgres_master.result
  db_subnet_group_name   = aws_db_subnet_group.postgres.name
  vpc_security_group_ids = [aws_security_group.postgres.id]
  storage_encrypted      = true
  backup_retention_period = 7
  skip_final_snapshot    = false

  serverlessv2_scaling_configuration {
    min_capacity = var.postgres_min_capacity
    max_capacity = var.postgres_max_capacity
  }
}

resource "aws_rds_cluster_instance" "postgres" {
  cluster_identifier = aws_rds_cluster.postgres.id
  instance_class     = "db.serverless"
  engine             = aws_rds_cluster.postgres.engine
  engine_version     = aws_rds_cluster.postgres.engine_version
}

# ── Kafka (MSK) — observability/async_publish.py's KafkaTracePublisher.
# The direct production swap-in for docker-compose.yml's Redpanda container:
# same Kafka wire protocol, so MODELROUTER_TRACE_KAFKA_BOOTSTRAP_SERVERS is
# the only thing that changes between environments. ──

resource "aws_security_group" "msk" {
  name_prefix = "${local.name_prefix}-msk-"
  vpc_id      = var.vpc_id

  ingress {
    from_port       = 9092
    to_port          = 9098   # covers plaintext, TLS, and SASL/IAM broker ports
    protocol        = "tcp"
    security_groups = [var.app_security_group_id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_msk_cluster" "this" {
  cluster_name           = "${local.name_prefix}-msk"
  kafka_version           = var.msk_kafka_version
  number_of_broker_nodes  = var.msk_broker_count

  broker_node_group_info {
    instance_type   = var.msk_broker_instance_type
    client_subnets  = var.private_subnet_ids
    security_groups = [aws_security_group.msk.id]

    storage_info {
      ebs_storage_info {
        volume_size = 100
      }
    }
  }

  encryption_info {
    encryption_in_transit {
      client_broker = "TLS"
      in_cluster    = true
    }
  }
}

# ── SQS — observability/async_publish.py's SqsTracePublisher alternative
# to Kafka, and tenancy/byok.py's KMS key below for envelope-encrypted BYOK
# credentials. ──

resource "aws_sqs_queue" "traces" {
  name                       = "${local.name_prefix}-traces"
  message_retention_seconds = 86400   # 1 day -- a durable EventStore copy already exists; this queue is a side channel
  visibility_timeout_seconds = 30
  kms_master_key_id          = aws_kms_key.byok.arn   # encrypted at rest with the SAME CMK BYOK uses -- one key, one audit trail
}

resource "aws_kms_key" "byok" {
  description             = "ModelRouter BYOK envelope-encryption CMK (tenancy/byok.py resolve_kms_sealed_key())"
  enable_key_rotation     = true
  deletion_window_in_days = 30
}

resource "aws_kms_alias" "byok" {
  name          = "alias/${local.name_prefix}-byok"
  target_key_id = aws_kms_key.byok.key_id
}

output "redis_primary_endpoint" {
  value = aws_elasticache_replication_group.this.primary_endpoint_address
}

output "postgres_dsn_secret_arn" {
  value       = aws_secretsmanager_secret.postgres_dsn.arn
  description = "Fetch this at deploy time -- MODELROUTER_POSTGRES_DSN is never a plain env var in production."
}

output "msk_bootstrap_brokers_tls" {
  value = aws_msk_cluster.this.bootstrap_brokers_tls
}

output "traces_queue_url" {
  value = aws_sqs_queue.traces.url
}

output "byok_kms_key_id" {
  value = aws_kms_key.byok.key_id
}
