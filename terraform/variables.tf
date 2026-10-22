variable "project_name" {
  description = "Prefix applied to every resource name."
  type        = string
  default     = "crop-bandit"
}

variable "environment" {
  description = "Environment tag (dev, staging, prod)."
  type        = string
  default     = "dev"
}

variable "aws_region" {
  description = "AWS region. Floci ignores this but the provider requires it."
  type        = string
  default     = "us-east-1"
}

variable "use_floci" {
  description = "Target the local Floci AWS emulator instead of real AWS."
  type        = bool
  default     = true
}

variable "floci_endpoint" {
  description = "Endpoint used for every service when use_floci = true."
  type        = string
  default     = "http://localhost:4566"
}

variable "artifact_retention_days" {
  description = "Days before old policy artifact versions are expired from S3."
  type        = number
  default     = 90

  validation {
    condition     = var.artifact_retention_days >= 1
    error_message = "artifact_retention_days must be at least 1."
  }
}

variable "training_schedule" {
  description = "EventBridge cron expression for the retraining job."
  type        = string
  default     = "cron(0 3 * * ? *)"
}
