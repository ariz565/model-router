# No default for anything network-shaped (vpc_id/subnet_ids) — hardcoding a
# VPC topology into a reusable module is exactly the kind of "works on my
# account" mistake that breaks the moment this runs against a real org's
# existing network. Every other default below is a genuinely reasonable
# starting point for a first environment, not a production-sized guess.

variable "environment" {
  description = "Deploy environment name, used as a resource-naming prefix (e.g. \"staging\", \"prod\")."
  type        = string
}

variable "vpc_id" {
  description = "VPC these resources deploy into. No default on purpose — must be an existing VPC."
  type        = string
}

variable "private_subnet_ids" {
  description = "Private subnet IDs for ElastiCache/RDS/MSK — never public, these hold tenant data and credentials."
  type        = list(string)
}

variable "app_security_group_id" {
  description = "Security group of the ModelRouter service itself — every ingress rule below scopes to exactly this SG, never 0.0.0.0/0."
  type        = string
}

variable "redis_node_type" {
  default = "cache.r7g.large"
}

variable "postgres_instance_class" {
  default = "db.r6g.large"
}

variable "postgres_min_capacity" {
  description = "Aurora Serverless v2 min ACUs — 0.5 lets it scale to near-zero cost outside business hours."
  default     = 0.5
}

variable "postgres_max_capacity" {
  default = 8
}

variable "msk_kafka_version" {
  default = "3.6.0"
}

variable "msk_broker_instance_type" {
  default = "kafka.m7g.large"
}

variable "msk_broker_count" {
  description = "Must be a multiple of the number of AZs in private_subnet_ids — MSK's own constraint, not this module's."
  default     = 3
}
