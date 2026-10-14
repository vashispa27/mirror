project_name   = "replica-ingestion"
environment    = "local"
aws_region     = "ap-south-1"
use_localstack = true

enable_deletion_protection    = false
enable_point_in_time_recovery = false
log_retention_days            = 1

# This creates infrastructure only. The disabled function is a placeholder and
# contains no crawler behavior.
crawler_schedules = {
  public_accounts = {
    schedule_expression = "rate(1 day)"
    enabled             = false
    payload = {
      mode = "incremental"
    }
  }
}

