provider "aws" {
  region = var.aws_region

  access_key = var.use_localstack ? "test" : null
  secret_key = var.use_localstack ? "test" : null

  skip_credentials_validation = var.use_localstack
  skip_metadata_api_check     = var.use_localstack
  skip_region_validation      = var.use_localstack
  skip_requesting_account_id  = var.use_localstack

  endpoints {
    dynamodb = var.use_localstack ? var.localstack_endpoint : null
    events   = var.use_localstack ? var.localstack_endpoint : null
    iam      = var.use_localstack ? var.localstack_endpoint : null
    lambda   = var.use_localstack ? var.localstack_endpoint : null
    logs     = var.use_localstack ? var.localstack_endpoint : null
    sqs      = var.use_localstack ? var.localstack_endpoint : null
    sts      = var.use_localstack ? var.localstack_endpoint : null
  }

  default_tags {
    tags = var.use_localstack ? {} : {
      Project     = var.project_name
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}

data "aws_caller_identity" "current" {}

locals {
  name_prefix = "${var.environment}-${var.project_name}"
  account_id  = var.use_localstack ? "000000000000" : data.aws_caller_identity.current.account_id

  local_lambda_environment = var.use_localstack ? {
    AWS_ENDPOINT_URL = var.localstack_lambda_endpoint
  } : {}
}
