#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
terraform_dir="${repo_dir}/infra"
endpoint="http://localhost:4566"

export AWS_ACCESS_KEY_ID=test
export AWS_SECRET_ACCESS_KEY=test
export AWS_DEFAULT_REGION=ap-south-1

aws_local() {
  aws --endpoint-url="${endpoint}" "$@"
}

documents_table="$(terraform -chdir="${terraform_dir}" output -raw documents_table_name)"
events_queue="$(terraform -chdir="${terraform_dir}" output -raw document_events_queue_url)"
forwarder_function="$(terraform -chdir="${terraform_dir}" output -raw stream_forwarder_function_name)"
document_id="smoke-$(date +%s)"

aws_local lambda get-function --function-name "${forwarder_function}" >/dev/null
aws_local events list-rules --name-prefix "local-replica-ingestion-crawler" >/dev/null

aws_local dynamodb put-item \
  --table-name "${documents_table}" \
  --item "{\"person_id\":{\"S\":\"smoke-person\"},\"document_id\":{\"S\":\"${document_id}\"},\"source\":{\"S\":\"smoke-test\"},\"content\":{\"S\":\"synthetic infrastructure test\"}}" \
  >/dev/null

for attempt in $(seq 1 30); do
  response="$(aws_local sqs receive-message \
    --queue-url "${events_queue}" \
    --max-number-of-messages 10 \
    --wait-time-seconds 1 \
    --visibility-timeout 5 \
    --output json)"

  match="$(SMOKE_RESPONSE="${response}" SMOKE_DOCUMENT_ID="${document_id}" python3 -c '
import json, os
response = json.loads(os.environ["SMOKE_RESPONSE"] or "{}")
target = os.environ["SMOKE_DOCUMENT_ID"]
for message in response.get("Messages", []):
    body = json.loads(message["Body"])
    if body.get("document", {}).get("document_id") == target:
        print(message["ReceiptHandle"])
        break
')"

  if [[ -n "${match}" ]]; then
    aws_local sqs delete-message --queue-url "${events_queue}" --receipt-handle "${match}" >/dev/null
    echo "PASS: DynamoDB stream event for ${document_id} reached SQS."
    exit 0
  fi

  sleep 1
done

echo "FAIL: no SQS event received for ${document_id}." >&2
exit 1
