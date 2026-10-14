variable "project_name" {
  description = "Name prefix used for all resources."
  type        = string
  default     = "replica-ingestion"
}

variable "environment" {
  description = "Environment name used in resource names and tags."
  type        = string
  default     = "local"
}

variable "aws_region" {
  description = "AWS region for the stack."
  type        = string
  default     = "ap-south-1"
}

variable "use_localstack" {
  description = "Route AWS provider and Lambda SDK traffic to LocalStack."
  type        = bool
  default     = true
}

variable "localstack_endpoint" {
  description = "LocalStack endpoint used by Terraform on the host."
  type        = string
  default     = "http://localhost:4566"
}

variable "localstack_lambda_endpoint" {
  description = "LocalStack endpoint reachable from Lambda containers."
  type        = string
  default     = "http://localhost.localstack.cloud:4566"
}

variable "enable_deletion_protection" {
  description = "Protect DynamoDB tables from accidental deletion in AWS."
  type        = bool
  default     = false
}

variable "enable_point_in_time_recovery" {
  description = "Enable DynamoDB point-in-time recovery."
  type        = bool
  default     = false
}

variable "log_retention_days" {
  description = "CloudWatch log retention for Lambda functions."
  type        = number
  default     = 7
}

variable "crawler_schedules" {
  description = "Crawler Lambda placeholders and their schedules. Platform choices can be changed before crawler implementation."
  type = map(object({
    schedule_expression = string
    enabled             = optional(bool, false)
    payload             = optional(map(string), {})
  }))

  default = {
    public_accounts = {
      schedule_expression = "rate(1 day)"
      enabled             = false
      payload = {
        mode = "incremental"
      }
    }
  }
}

