resource "aws_cloudwatch_event_rule" "crawler" {
  for_each = var.crawler_schedules

  name                = "${local.name_prefix}-crawler-${each.key}"
  description         = "Schedule for the ${each.key} crawler Lambda placeholder."
  schedule_expression = each.value.schedule_expression
  state               = each.value.enabled ? "ENABLED" : "DISABLED"
}

resource "aws_cloudwatch_event_target" "crawler" {
  for_each = var.crawler_schedules

  rule = aws_cloudwatch_event_rule.crawler[each.key].name
  arn  = aws_lambda_function.crawler[each.key].arn
  input = jsonencode(merge(
    {
      crawler_name = each.key
      trigger      = "scheduled"
    },
    each.value.payload,
  ))

  dead_letter_config {
    arn = aws_sqs_queue.crawler_schedule_dlq.arn
  }
}

resource "aws_lambda_permission" "allow_eventbridge" {
  for_each = var.crawler_schedules

  statement_id  = "AllowExecutionFromEventBridge-${each.key}"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.crawler[each.key].function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.crawler[each.key].arn
}
