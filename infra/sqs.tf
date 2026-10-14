resource "aws_sqs_queue" "document_events_dlq" {
  name                      = "${local.name_prefix}-document-events-dlq"
  message_retention_seconds = 1209600
  sqs_managed_sse_enabled   = true
}

resource "aws_sqs_queue" "document_events" {
  name                       = "${local.name_prefix}-document-events"
  visibility_timeout_seconds = 60
  message_retention_seconds  = 345600
  receive_wait_time_seconds  = 10
  sqs_managed_sse_enabled    = true

  redrive_policy = jsonencode({
    deadLetterTargetArn = aws_sqs_queue.document_events_dlq.arn
    maxReceiveCount     = 5
  })
}

resource "aws_sqs_queue_redrive_allow_policy" "document_events" {
  queue_url = aws_sqs_queue.document_events_dlq.id

  redrive_allow_policy = jsonencode({
    redrivePermission = "byQueue"
    sourceQueueArns   = [aws_sqs_queue.document_events.arn]
  })
}

resource "aws_sqs_queue" "cdc_failure_dlq" {
  name                      = "${local.name_prefix}-cdc-failure-dlq"
  message_retention_seconds = 1209600
  sqs_managed_sse_enabled   = true
}

resource "aws_sqs_queue" "crawler_schedule_dlq" {
  name                      = "${local.name_prefix}-crawler-schedule-dlq"
  message_retention_seconds = 1209600
  sqs_managed_sse_enabled   = true
}

data "aws_iam_policy_document" "crawler_schedule_dlq" {
  statement {
    sid     = "AllowEventBridgeFailures"
    effect  = "Allow"
    actions = ["sqs:SendMessage"]

    principals {
      type        = "Service"
      identifiers = ["events.amazonaws.com"]
    }

    resources = [aws_sqs_queue.crawler_schedule_dlq.arn]
  }
}

resource "aws_sqs_queue_policy" "crawler_schedule_dlq" {
  queue_url = aws_sqs_queue.crawler_schedule_dlq.id
  policy    = data.aws_iam_policy_document.crawler_schedule_dlq.json
}
