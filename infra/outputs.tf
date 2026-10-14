output "sources_table_name" {
  value = aws_dynamodb_table.sources.name
}

output "documents_table_name" {
  value = aws_dynamodb_table.documents.name
}

output "documents_stream_arn" {
  value = aws_dynamodb_table.documents.stream_arn
}

output "document_events_queue_url" {
  value = aws_sqs_queue.document_events.url
}

output "document_events_dlq_url" {
  value = aws_sqs_queue.document_events_dlq.url
}

output "crawler_function_names" {
  value = { for name, function in aws_lambda_function.crawler : name => function.function_name }
}

output "stream_forwarder_function_name" {
  value = aws_lambda_function.stream_forwarder.function_name
}

output "crawler_schedule_rule_names" {
  value = { for name, rule in aws_cloudwatch_event_rule.crawler : name => rule.name }
}
