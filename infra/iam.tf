data "aws_iam_policy_document" "lambda_assume_role" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "crawler" {
  name               = "${local.name_prefix}-crawler-role"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume_role.json
}

data "aws_iam_policy_document" "crawler" {
  statement {
    sid = "WriteLogs"
    actions = [
      "logs:CreateLogStream",
      "logs:PutLogEvents",
    ]
    resources = ["arn:aws:logs:${var.aws_region}:${local.account_id}:log-group:/aws/lambda/${local.name_prefix}-crawler-*:*"]
  }

  statement {
    sid = "ReadSourceConfiguration"
    actions = [
      "dynamodb:GetItem",
      "dynamodb:Query",
      "dynamodb:Scan",
    ]
    resources = [aws_dynamodb_table.sources.arn]
  }

  statement {
    sid = "WriteDocuments"
    actions = [
      "dynamodb:BatchWriteItem",
      "dynamodb:PutItem",
      "dynamodb:UpdateItem",
    ]
    resources = [aws_dynamodb_table.documents.arn]
  }
}

resource "aws_iam_role_policy" "crawler" {
  name   = "${local.name_prefix}-crawler-policy"
  role   = aws_iam_role.crawler.id
  policy = data.aws_iam_policy_document.crawler.json
}

resource "aws_iam_role" "stream_forwarder" {
  name               = "${local.name_prefix}-stream-forwarder-role"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume_role.json
}

data "aws_iam_policy_document" "stream_forwarder" {
  statement {
    sid = "WriteLogs"
    actions = [
      "logs:CreateLogStream",
      "logs:PutLogEvents",
    ]
    resources = ["arn:aws:logs:${var.aws_region}:${local.account_id}:log-group:/aws/lambda/${local.name_prefix}-stream-forwarder:*"]
  }

  statement {
    sid = "ReadDocumentStream"
    actions = [
      "dynamodb:DescribeStream",
      "dynamodb:GetRecords",
      "dynamodb:GetShardIterator",
      "dynamodb:ListStreams",
    ]
    resources = [aws_dynamodb_table.documents.stream_arn]
  }

  statement {
    sid       = "PublishDocumentEvents"
    actions   = ["sqs:SendMessage"]
    resources = [aws_sqs_queue.document_events.arn]
  }

  statement {
    sid       = "PublishFailedBatches"
    actions   = ["sqs:SendMessage"]
    resources = [aws_sqs_queue.cdc_failure_dlq.arn]
  }
}

resource "aws_iam_role_policy" "stream_forwarder" {
  name   = "${local.name_prefix}-stream-forwarder-policy"
  role   = aws_iam_role.stream_forwarder.id
  policy = data.aws_iam_policy_document.stream_forwarder.json
}
