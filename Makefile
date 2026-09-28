.DEFAULT_GOAL := help

API := api
CORE := core_ml
PY := python

AWS_ACCOUNT_ID ?= 040175285118
AWS_REGION ?= us-east-1
ECR_REGISTRY ?= $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com
API_IMAGE := $(ECR_REGISTRY)/predictive-maintenance-mlops-api
# Tag inmutable y trazable: por defecto el commit de git (igual que
# CI_COMMIT_SHA en GitLab) -- nunca "latest" en un release real.
IMAGE_TAG ?= $(shell git rev-parse --short HEAD)
K8S_OVERLAY := kubernetes/overlays/production
EKS_CLUSTER_NAME ?= predictive-maintenance-mlops

.PHONY: help install install-api install-core \
	format format-check lint lint-fix typecheck \
	test test-api test-core coverage \
	precommit-install precommit-run \
	build-toy-dataset prepare-toy prepare-data train train-toy smoke-test \
	dvc-pull dvc-push dvc-use-localstack \
	compose-up compose-down compose-destroy compose-logs compose-ps localstack-env \
	docker-build docker-push \
	k8s-build k8s-diff deploy \
	terraform-fmt terraform-validate terraform-plan terraform-apply \
	ci-local \
	ci clean

help: ## Muestra esta ayuda
	@echo "Interfaz unica de ejecucion (local y CI). Targets disponibles:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

## ---------------------------------------------------------------------
## Dependencias deterministas (Poetry + lockfile)
## ---------------------------------------------------------------------

install: install-api install-core ## Instala dependencias (runtime + dev) de api/ y core_ml/

install-api: ## Instala dependencias de api/ (incluye grupo dev)
	poetry -C $(API) install --with dev --no-interaction

install-core: ## Instala dependencias de core_ml/ (incluye grupo dev)
	poetry -C $(CORE) install --with dev --no-interaction

## ---------------------------------------------------------------------
## Validacion shift-left (Python)
## ---------------------------------------------------------------------

format: ## Aplica ruff format (modifica archivos) en api/ y core_ml/
	poetry -C $(API) run ruff format .
	poetry -C $(CORE) run ruff format .

format-check: ## Verifica formato sin modificar (usado en CI)
	poetry -C $(API) run ruff format --check .
	poetry -C $(CORE) run ruff format --check .

lint: ## Ejecuta ruff (lint) en api/ y core_ml/
	poetry -C $(API) run ruff check .
	poetry -C $(CORE) run ruff check .

lint-fix: ## Ejecuta ruff con autofix en api/ y core_ml/
	poetry -C $(API) run ruff check --fix .
	poetry -C $(CORE) run ruff check --fix .

typecheck: ## Ejecuta mypy en api/ y core_ml/ (opcional, no bloquea CI)
	poetry -C $(API) run mypy .
	poetry -C $(CORE) run mypy src

test: test-api test-core ## Ejecuta pytest en api/ y core_ml/

test-api: ## Ejecuta pytest solo en api/
	poetry -C $(API) run pytest

test-core: ## Ejecuta pytest solo en core_ml/
	poetry -C $(CORE) run pytest

coverage: ## Ejecuta pytest con cobertura en api/ y core_ml/
	poetry -C $(API) run pytest --cov=. --cov-report=term-missing
	poetry -C $(CORE) run pytest --cov=src --cov-report=term-missing

## ---------------------------------------------------------------------
## Datos y reproducibilidad
## ---------------------------------------------------------------------

build-toy-dataset: ## Regenera el dataset toy (~1000 filas, fijo) desde el dataset completo
	poetry -C $(CORE) run python scripts/build_toy_dataset.py

prepare-toy: ## Construye engine_features.parquet + scaler.joblib del dataset toy
	poetry -C $(CORE) run python src/prepare_training_data.py --data-dir data_toy

prepare-data: ## Construye engine_features.parquet + scaler.joblib del dataset completo
	poetry -C $(CORE) run python src/prepare_training_data.py --data-dir data

train: prepare-data ## Entrena con el dataset completo (exige el quality gate)
	poetry -C $(CORE) run python src/train.py --data-dir data --epochs 25 --patience 5

# MLFLOW_TRACKING_URI por defecto: un store SQLite local y efimero. Sin
# esto el comando no seria autonomo: config.yaml resuelve
# ${MLFLOW_TRACKING_URI} y, sin la variable, el fallback heredado era
# "localhost:5000" (una URI SIN esquema), que MLflow rechaza.
# Se respeta el valor externo si ya esta definido (p.ej. el MLflow de
# docker-compose: export MLFLOW_TRACKING_URI=http://localhost:5001).
train-toy: prepare-toy ## Entrena end-to-end con el dataset toy: segundos, sin GPU
	MLFLOW_TRACKING_URI="$${MLFLOW_TRACKING_URI:-sqlite:///mlruns/smoke_test.db}" \
	poetry -C $(CORE) run python src/train.py \
		--data-dir data_toy --no-enforce-quality-gate

smoke-test: train-toy ## Valida el pipeline E2E (datos+contratos+MLflow) antes de gastar computo real
	@echo "OK: pipeline end-to-end verificado con el dataset toy."

dvc-pull: ## Descarga datasets versionados desde el remoto S3 configurado en DVC
	poetry -C $(CORE) run dvc pull

dvc-push: ## Sube datasets versionados (data/, data_toy/) al remoto S3 de DVC
	poetry -C $(CORE) run dvc push

dvc-use-localstack: ## Redirige el remoto S3 de DVC a LocalStack (solo esta maquina; ver .dvc/config.local)
	poetry -C $(CORE) run dvc remote modify --local s3remote endpointurl http://localhost:4566
	poetry -C $(CORE) run dvc remote modify --local s3remote access_key_id test
	poetry -C $(CORE) run dvc remote modify --local s3remote secret_access_key test
	@echo "OK: core_ml/.dvc/config.local apunta 's3remote' a LocalStack (http://localhost:4566)."
	@echo "    core_ml/.dvc/config (compartido en git) sigue apuntando al S3 real; sin cambios."

## ---------------------------------------------------------------------
## Contenedores locales (Docker Compose + LocalStack)
## ---------------------------------------------------------------------

compose-up: ## Levanta el stack local completo: Postgres, MLflow, API, LocalStack
	docker compose up -d --build
	@echo "API:        http://localhost:8001 (503 en /health hasta que exista un modelo 'champion')"
	@echo "MLflow UI:  http://localhost:5001"
	@echo "LocalStack: http://localhost:4567"

compose-down: ## Detiene el stack local (conserva los volumenes: datos de Postgres/LocalStack)
	docker compose down

compose-destroy: ## Detiene el stack local y BORRA sus volumenes (reinicio limpio)
	docker compose down --volumes

compose-logs: ## Sigue los logs de todos los servicios del stack local
	docker compose logs -f

compose-ps: ## Muestra el estado/healthcheck de cada servicio del stack local
	docker compose ps

localstack-env: ## Imprime las variables para apuntar un shell local a LocalStack
	@echo "export AWS_ENDPOINT_URL=http://localhost:4566"
	@echo "export AWS_ACCESS_KEY_ID=test"
	@echo "export AWS_SECRET_ACCESS_KEY=test"
	@echo "export AWS_DEFAULT_REGION=us-east-1"

## ---------------------------------------------------------------------
## Build/push de la imagen de la API y despliegue a Kubernetes
## ---------------------------------------------------------------------
# MLflow no tiene una imagen propia que construir/publicar: corre desde la
# imagen publica ghcr.io/mlflow/mlflow (ver docker-compose.yml y
# kubernetes/base/mlflow.yaml).

docker-build: ## Construye la imagen de la API (Dockerfile de la raiz)
	docker build -t $(API_IMAGE):$(IMAGE_TAG) -f Dockerfile .

docker-push: ## Publica la imagen de la API en ECR (requiere `docker login` a ECR ya hecho)
	docker push $(API_IMAGE):$(IMAGE_TAG)

k8s-build: ## Renderiza el overlay de produccion (kustomize build) para inspeccion/dry-run
	kubectl kustomize $(K8S_OVERLAY)

k8s-diff: ## Muestra el diff del overlay contra el cluster actual (kubectl diff -k)
	kubectl diff -k $(K8S_OVERLAY) || true

deploy: ## Fija IMAGE_TAG en el overlay y aplica directo contra el cluster (kubectl apply -k)
	cd $(K8S_OVERLAY) && kustomize edit set image $(API_IMAGE)=$(API_IMAGE):$(IMAGE_TAG)
	kubectl apply -k $(K8S_OVERLAY)

## ---------------------------------------------------------------------
## Infraestructura como Codigo (Terraform)
## ---------------------------------------------------------------------

terraform-fmt: ## Verifica el formato canonico de todos los .tf (incluye terraform/bootstrap)
	terraform -chdir=terraform fmt -check -diff -recursive

terraform-validate: ## Valida sintaxis/tipos de terraform/ (requiere `terraform init` previo)
	terraform -chdir=terraform validate

terraform-plan: ## Muestra el plan de cambios contra AWS real (requiere credenciales validas)
	terraform -chdir=terraform plan

terraform-apply: ## Aplica el plan contra AWS real (requiere credenciales validas)
	terraform -chdir=terraform apply

## ---------------------------------------------------------------------
## Validar el pipeline de GitLab CI localmente
## ---------------------------------------------------------------------

ci-local: ## Corre .gitlab-ci.yml localmente con gitlab-ci-local (requiere Docker + Node.js)
	@command -v npx >/dev/null 2>&1 || { \
		echo "Node.js/npx no esta instalado. La alternativa mantenida por la"; \
		echo "comunidad para correr pipelines de GitLab CI en local es"; \
		echo "gitlab-ci-local: https://github.com/firecow/gitlab-ci-local"; \
		echo "Instala Node.js (nodejs.org) y vuelve a correr 'make ci-local'."; \
		exit 1; \
	}
	MSYS_NO_PATHCONV=1 npx --yes gitlab-ci-local $(JOB)

## ---------------------------------------------------------------------
## Git hooks
## ---------------------------------------------------------------------

precommit-install: ## Instala los git hooks de pre-commit en este repo
	pre-commit install --install-hooks
	pre-commit install --hook-type commit-msg || true

precommit-run: ## Ejecuta todos los hooks de pre-commit sobre todo el repo
	pre-commit run --all-files

## ---------------------------------------------------------------------
## Agregados
## ---------------------------------------------------------------------

ci: format-check lint test ## Pipeline completo (el mismo que corre en GitLab CI, stage lint_test)
	@echo "OK: todos los checks de calidad pasaron."

clean: ## Limpia caches locales de herramientas
	find . -type d -name "__pycache__" -not -path "*/.venv/*" -prune -exec rm -rf {} +
	rm -rf $(API)/.mypy_cache $(API)/.ruff_cache $(API)/.pytest_cache $(API)/htmlcov
	rm -rf $(CORE)/.mypy_cache $(CORE)/.ruff_cache $(CORE)/.pytest_cache $(CORE)/htmlcov
