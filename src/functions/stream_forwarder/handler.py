import json
import os
from datetime import datetime, timezone

import boto3
from boto3.dynamodb.types import TypeDeserializer


_deserializer = TypeDeserializer()
_sqs = boto3.client("sqs")
_queue_url = os.environ["DOCUMENT_EVENTS_QUEUE_URL"]
_schema_version = int(os.environ.get("EVENT_SCHEMA_VERSION", "1"))


def _deserialize(values):
    return {key: _deserializer.deserialize(value) for key, value in values.items()}


def _event_type(operation):
    return {
        "INSERT": "document.created",
        "MODIFY": "document.updated",
        "REMOVE": "document.deleted",
    }.get(operation, "document.changed")


def _message(record):
    dynamodb = record["dynamodb"]
    keys = _deserialize(dynamodb["Keys"])
    source_arn = record.get("eventSourceARN", "")
    table_name = source_arn.split("/")[1] if "/" in source_arn else None

    return {
        "schema_version": _schema_version,
        "event_id": record["eventID"],
        "event_type": _event_type(record["eventName"]),
        "occurred_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "service": "dynamodb",
            "table": table_name,
            "stream_sequence_number": dynamodb.get("SequenceNumber"),
        },
        "document": {
            "person_id": keys.get("person_id"),
            "document_id": keys.get("document_id"),
        },
    }


def handler(event, context):
    failures = []

    for record in event.get("Records", []):
        sequence_number = record.get("dynamodb", {}).get("SequenceNumber")
        try:
            _sqs.send_message(
                QueueUrl=_queue_url,
                MessageBody=json.dumps(_message(record), separators=(",", ":")),
            )
        except Exception:
            if sequence_number:
                failures.append({"itemIdentifier": sequence_number})

    return {"batchItemFailures": failures}

