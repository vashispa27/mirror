data "archive_file" "crawler_placeholder" {
  type        = "zip"
  source_file = "${path.module}/../src/functions/crawler_placeholder/handler.py"
  output_path = "${path.module}/.build/crawler-placeholder.zip"
}

data "archive_file" "stream_forwarder" {
  type        = "zip"
  source_file = "${path.module}/../src/functions/stream_forwarder/handler.py"
  output_path = "${path.module}/.build/stream-forwarder.zip"
}

resource "aws_cloudwatch_log_group" "crawler" {
  for_each = var.crawler_schedules

  name              = "/aws/lambda/${local.name_prefix}-crawler-${each.key}"
  retention_in_days = var.log_retention_days
}

resource "aws_lambda_function" "crawler" {
  for_each = var.crawler_schedules

  function_name    = "${local.name_prefix}-crawler-${each.key}"
  description      = "Placeholder for the ${each.key} crawler; crawler logic is intentionally not implemented."
  role             = aws_iam_role.crawler.arn
  runtime          = "python3.12"
  handler          = "handler.handler"
  filename         = data.archive_file.crawler_placeholder.output_path
  source_code_hash = data.archive_file.crawler_placeholder.output_base64sha256
  timeout          = 60
  memory_size      = 256

  environment {
    variables = merge(
      {
        CRAWLER_NAME    = each.key
        SOURCES_TABLE   = aws_dynamodb_table.sources.name
        DOCUMENTS_TABLE = aws_dynamodb_table.documents.name
      },
      local.local_lambda_environment,
    )
  }

  depends_on = [
    aws_cloudwatch_log_group.crawler,
    aws_iam_role_policy.crawler,
  ]
}

resource "aws_cloudwatch_log_group" "stream_forwarder" {
  name              = "/aws/lambda/${local.name_prefix}-stream-forwarder"
  retention_in_days = var.log_retention_days
}

resource "aws_lambda_function" "stream_forwarder" {
  function_name    = "${local.name_prefix}-stream-forwarder"
  description      = "Forwards DynamoDB document change events to SQS."
  role             = aws_iam_role.stream_forwarder.arn
  runtime          = "python3.12"
  handler          = "handler.handler"
  filename         = data.archive_file.stream_forwarder.output_path
  source_code_hash = data.archive_file.stream_forwarder.output_base64sha256
  timeout          = 30
  memory_size      = 256

  environment {
    variables = merge(
      {
        DOCUMENT_EVENTS_QUEUE_URL = aws_sqs_queue.document_events.url
        EVENT_SCHEMA_VERSION      = "1"
      },
      local.local_lambda_environment,
    )
  }

  depends_on = [
    aws_cloudwatch_log_group.stream_forwarder,
    aws_iam_role_policy.stream_forwarder,
  ]
}

resource "aws_lambda_event_source_mapping" "documents_stream" {
  event_source_arn  = aws_dynamodb_table.documents.stream_arn
  function_name     = aws_lambda_function.stream_forwarder.arn
  starting_position = "LATEST"
  batch_size        = 10

  bisect_batch_on_function_error = true
  maximum_retry_attempts         = 3
  function_response_types        = ["ReportBatchItemFailures"]

  destination_config {
    on_failure {
      destination_arn = aws_sqs_queue.cdc_failure_dlq.arn
    }
  }
}

