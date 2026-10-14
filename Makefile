SHELL := /bin/bash

TERRAFORM_DIR := infra
LOCALSTACK_ENDPOINT := http://localhost:4566
AWS_LOCAL := AWS_ACCESS_KEY_ID=test AWS_SECRET_ACCESS_KEY=test AWS_DEFAULT_REGION=ap-south-1 AWS_ENDPOINT_URL=$(LOCALSTACK_ENDPOINT)
PYTHON := .venv/bin/python

.PHONY: help local-up local-down local-logs qdrant-up qdrant-down qdrant-status qdrant-smoke-test memory-up test-whatsapp-memory-integration test-whatsapp-semantic-integration test-whatsapp-index-integration test-whatsapp-retrieval-integration test-whatsapp-eval-integration test-rag-lab tf-init tf-fmt tf-validate tf-plan tf-apply tf-destroy smoke-test test validate

help:
	@echo "local-up       Start LocalStack"
	@echo "local-down     Stop LocalStack"
	@echo "qdrant-up      Start Qdrant"
	@echo "qdrant-down    Stop Qdrant"
	@echo "qdrant-status  Show Qdrant status"
	@echo "qdrant-smoke-test  Verify Qdrant vector operations"
	@echo "memory-up      Start Postgres and Qdrant for contact memory"
	@echo "test-whatsapp-memory-integration  Run Postgres/Qdrant memory tests"
	@echo "test-whatsapp-semantic-integration  Run local Hinglish semantic boundary probes"
	@echo "test-whatsapp-index-integration  Run Docker-backed WhatsApp indexing smoke test"
	@echo "test-whatsapp-retrieval-integration  Run Docker-backed WhatsApp retrieval test"
	@echo "test-whatsapp-eval-integration  Run Docker-backed retrieval evaluation smoke test"
	@echo "test-rag-lab     Run RAG lab unit tests"
	@echo "tf-init        Install Terraform providers"
	@echo "tf-plan        Plan the local infrastructure"
	@echo "tf-apply       Apply the local infrastructure"
	@echo "tf-destroy     Destroy Terraform-managed local resources"
	@echo "smoke-test     Verify DynamoDB Streams -> Lambda -> SQS"
	@echo "test           Run the Python unit tests"
	@echo "validate       Format-check and validate Terraform"

local-up:
	docker compose up -d --wait localstack

local-down:
	docker compose down

local-logs:
	docker compose logs -f localstack

qdrant-up:
	docker compose up -d --wait qdrant

qdrant-down:
	docker compose stop qdrant

qdrant-status:
	docker compose ps qdrant

qdrant-smoke-test: qdrant-up
	./scripts/qdrant-smoke-test.sh

memory-up:
	docker compose up -d --wait postgres qdrant

test-whatsapp-memory-integration: memory-up
	WHATSAPP_MEMORY_INTEGRATION=1 PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -p 'test_whatsapp_memory_integration.py' -v

test-whatsapp-semantic-integration:
	WHATSAPP_SEMANTIC_INTEGRATION=1 PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -p 'test_whatsapp_semantic_topics.py' -v

test-whatsapp-index-integration: qdrant-up
	WHATSAPP_QDRANT_INTEGRATION=1 PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -p 'test_whatsapp_index_integration.py' -v

test-whatsapp-retrieval-integration: qdrant-up
	WHATSAPP_QDRANT_INTEGRATION=1 PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -p 'test_whatsapp_retrieval_integration.py' -v

test-whatsapp-eval-integration: qdrant-up
	WHATSAPP_QDRANT_INTEGRATION=1 PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -p 'test_whatsapp_eval_integration.py' -v

test-rag-lab:
	cd rag_lab && PYTHONPATH=src ../$(PYTHON) -m unittest discover -s tests -v

tf-init:
	$(AWS_LOCAL) terraform -chdir=$(TERRAFORM_DIR) init

tf-fmt:
	terraform -chdir=$(TERRAFORM_DIR) fmt -recursive

tf-validate: tf-init
	terraform -chdir=$(TERRAFORM_DIR) validate

tf-plan: tf-init
	$(AWS_LOCAL) terraform -chdir=$(TERRAFORM_DIR) plan -var-file=environments/local.tfvars

tf-apply: tf-init
	$(AWS_LOCAL) terraform -chdir=$(TERRAFORM_DIR) apply -auto-approve -var-file=environments/local.tfvars

tf-destroy: tf-init
	$(AWS_LOCAL) terraform -chdir=$(TERRAFORM_DIR) destroy -auto-approve -var-file=environments/local.tfvars

smoke-test:
	./scripts/smoke-test.sh

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v

validate: tf-fmt tf-validate
	terraform -chdir=$(TERRAFORM_DIR) fmt -check -recursive
