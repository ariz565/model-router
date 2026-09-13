# ModelRouter's own compute layer — ECS Fargate, autoscaled on request
# concurrency rather than CPU (router.py's own work is I/O-bound: it's
# mostly awaiting provider HTTP calls, not computing — CPU utilization
# stays low right up until the pod is actually saturated with in-flight
# requests, so it's a lagging, unsafe autoscaling signal here specifically).
#
# Does NOT create: the ALB/listener/target group (var.alb_target_group_arn),
# the ECR repository or its image (var.container_image), or the IAM
# execution/task roles (var.execution_role_arn/var.task_role_arn) — every
# real deployment already has its own conventions for image publishing and
# IAM boundaries, and hardcoding one here would be the same "works on my
# account" mistake variables.tf's own top comment already calls out for
# networking.

variable "container_image" {
  description = "Full ECR image URI, e.g. 123456789.dkr.ecr.us-east-1.amazonaws.com/modelrouter:latest"
  type        = string
}

variable "alb_target_group_arn" {
  type = string
}

variable "execution_role_arn" {
  type = string
}

variable "task_role_arn" {
  description = "Needs, at minimum: secretsmanager:GetSecretValue (Postgres DSN), kms:Decrypt+GenerateDataKey (BYOK CMK), sqs:SendMessage (traces queue) -- scoped to exactly the ARNs this module outputs, never *."
  type        = string
}

variable "desired_count" {
  default = 3
}

variable "min_capacity" {
  default = 3
}

variable "max_capacity" {
  description = "Sizing note from the product-vision doc: ~20-30 pods covers 1000 req/sec at ~150ms internal p50 / ~50 concurrent requests per pod -- this default is deliberately below that ceiling; raise it once real load data confirms the per-pod concurrency assumption."
  default     = 30
}

resource "aws_ecs_cluster" "this" {
  name = "${local.name_prefix}-cluster"
}

resource "aws_cloudwatch_log_group" "app" {
  name              = "/ecs/${local.name_prefix}"
  retention_in_days = 30
}

resource "aws_ecs_task_definition" "app" {
  family                   = "${local.name_prefix}-app"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = "1024"
  memory                   = "2048"
  execution_role_arn       = var.execution_role_arn
  task_role_arn            = var.task_role_arn

  container_definitions = jsonencode([{
    name  = "modelrouter"
    image = var.container_image
    portMappings = [{ containerPort = 8000, protocol = "tcp" }]
    environment = [
      { name = "MODELROUTER_STORAGE", value = "postgres" },
      { name = "MODELROUTER_REDIS_URL", value = "rediss://${aws_elasticache_replication_group.this.primary_endpoint_address}:6379/0" },
      { name = "MODELROUTER_REDIS_LEDGER_URL", value = "rediss://${aws_elasticache_replication_group.this.primary_endpoint_address}:6379/1" },
      { name = "MODELROUTER_TRACE_SQS_QUEUE_URL", value = aws_sqs_queue.traces.url },
      { name = "MODELROUTER_BYOK_KMS_KEY_ID", value = aws_kms_key.byok.key_id },
      { name = "AWS_REGION", value = data.aws_region.current.name },
    ]
    secrets = [
      { name = "MODELROUTER_POSTGRES_DSN", valueFrom = aws_secretsmanager_secret.postgres_dsn.arn },
    ]
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.app.name
        "awslogs-region"        = data.aws_region.current.name
        "awslogs-stream-prefix" = "app"
      }
    }
  }])
}

data "aws_region" "current" {}

resource "aws_ecs_service" "app" {
  name            = "${local.name_prefix}-app"
  cluster         = aws_ecs_cluster.this.id
  task_definition = aws_ecs_task_definition.app.arn
  desired_count   = var.desired_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets         = var.private_subnet_ids
    security_groups = [var.app_security_group_id]
  }

  load_balancer {
    target_group_arn = var.alb_target_group_arn
    container_name    = "modelrouter"
    container_port    = 8000
  }

  deployment_minimum_healthy_percent = 100   # never drop below full capacity mid-deploy
  deployment_maximum_percent         = 200   # allow a full extra generation during rolling deploy
}

resource "aws_appautoscaling_target" "app" {
  service_namespace  = "ecs"
  resource_id        = "service/${aws_ecs_cluster.this.name}/${aws_ecs_service.app.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  min_capacity       = var.min_capacity
  max_capacity       = var.max_capacity
}

resource "aws_appautoscaling_policy" "requests_per_target" {
  name               = "${local.name_prefix}-scale-on-alb-request-count"
  service_namespace  = aws_appautoscaling_target.app.service_namespace
  resource_id        = aws_appautoscaling_target.app.resource_id
  scalable_dimension = aws_appautoscaling_target.app.scalable_dimension
  policy_type        = "TargetTrackingScaling"

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ALBRequestCountPerTarget"
      resource_label          = var.alb_target_group_arn
    }
    target_value       = 50   # matches the ~50-concurrent-per-pod sizing assumption above
    scale_in_cooldown  = 120
    scale_out_cooldown = 60   # scale out faster than in -- a slow scale-out under a real spike costs latency; a slow scale-in only costs a little idle spend
  }
}

output "ecs_service_name" {
  value = aws_ecs_service.app.name
}
