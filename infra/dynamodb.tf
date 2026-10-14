resource "aws_dynamodb_table" "sources" {
  name         = "${local.name_prefix}-sources"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "source_id"

  attribute {
    name = "source_id"
    type = "S"
  }

  deletion_protection_enabled = var.enable_deletion_protection

  point_in_time_recovery {
    enabled = var.enable_point_in_time_recovery
  }
}

resource "aws_dynamodb_table" "documents" {
  name         = "${local.name_prefix}-documents"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "person_id"
  range_key    = "document_id"

  attribute {
    name = "person_id"
    type = "S"
  }

  attribute {
    name = "document_id"
    type = "S"
  }

  stream_enabled   = true
  stream_view_type = "NEW_AND_OLD_IMAGES"

  deletion_protection_enabled = var.enable_deletion_protection

  point_in_time_recovery {
    enabled = var.enable_point_in_time_recovery
  }
}

