.DEFAULT_GOAL := help

API := api
CORE := core_ml
PY := python

AWS_ACCOUNT_ID ?= 040175285118
AWS_REGION ?= us-east-1
ECR_REGISTRY ?= $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com
API_IMAGE := $(ECR_REGISTRY)/predictive-maintenance-mlops-streaming
CONSUMER_IMAGE := $(ECR_REGISTRY)/predictive-maintenance-mlops-consumer
MONITORING_IMAGE := $(ECR_REGISTRY)/predictive-maintenance-mlops-monitoring
MLFLOW_IMAGE := $(ECR_REGISTRY)/predictive-maintenance-mlops-mlflow
BUILD_CACHE_IMAGE := $(ECR_REGISTRY)/predictive-maintenance-mlops-build-cache
# Tag inmutable y trazable: por defecto el commit de git (igual que
# CI_COMMIT_SHA en GitLab) -- nunca "latest" en un release real.
IMAGE_TAG ?= $(shell git rev-parse --short HEAD)
GITOPS_OVERLAY := kubernetes/overlays/production

.PHONY: help install install-api install-core \
	format format-check lint lint-fix typecheck \
	test test-api test-core coverage test-integration \
	security bandit trivy yamllint \
	precommit-install precommit-run \
	build-toy-dataset prepare-toy prepare-data train train-toy smoke-test msk-bootstrap-topics \
	dvc-pull dvc-push dvc-use-localstack \
	compose-up compose-down compose-destroy compose-logs compose-ps localstack-env \
	docker-build-api docker-build-consumer docker-build-monitoring docker-build-mlflow docker-build \
	docker-push-api docker-push-consumer docker-push-monitoring docker-push-mlflow docker-push \
	docker-buildx-push-api docker-buildx-push-consumer docker-buildx-push-monitoring \
	docker-buildx-push-mlflow docker-buildx-push \
	k8s-build k8s-diff gitops-set-image gitops-release \
	terraform-fmt terraform-validate terraform-plan \
	ci-local \
	runner-register runner-up runner-down runner-status runner-logs runner-unregister \
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

format: ## Aplica isort + black (modifica archivos) en api/ y core_ml/
	poetry -C $(API) run isort .
	poetry -C $(API) run black .
	poetry -C $(CORE) run isort .
	poetry -C $(CORE) run black .

format-check: ## Verifica formato/orden de imports sin modificar (usado en CI)
	poetry -C $(API) run isort --check-only .
	poetry -C $(API) run black --check .
	poetry -C $(CORE) run isort --check-only .
	poetry -C $(CORE) run black --check .

lint: ## Ejecuta ruff (lint) en api/ y core_ml/
	poetry -C $(API) run ruff check .
	poetry -C $(CORE) run ruff check .

lint-fix: ## Ejecuta ruff con autofix en api/ y core_ml/
	poetry -C $(API) run ruff check --fix .
	poetry -C $(CORE) run ruff check --fix .

typecheck: ## Ejecuta mypy en api/ y core_ml/
	poetry -C $(API) run mypy .
	poetry -C $(CORE) run mypy src streaming feature_store

test: test-api test-core ## Ejecuta pytest en api/ y core_ml/

test-api: ## Ejecuta pytest solo en api/
	poetry -C $(API) run pytest

test-core: ## Ejecuta pytest solo en core_ml/
	poetry -C $(CORE) run pytest

coverage: ## Ejecuta pytest con cobertura en api/ y core_ml/
	poetry -C $(API) run pytest --cov=. --cov-report=term-missing
	poetry -C $(CORE) run pytest --cov=src --cov=streaming --cov=feature_store --cov-report=term-missing

test-integration: ## Pruebas de integracion (Testcontainers: Postgres/Kafka/LocalStack). Requiere Docker.
	poetry -C $(API) run pytest -m integration -v
	poetry -C $(CORE) run pytest -m integration -v

## ---------------------------------------------------------------------
## Seguridad (secretos, dependencias, IaC)
## ---------------------------------------------------------------------

bandit: ## Analisis estatico de seguridad (SAST) del codigo Python
	poetry -C $(API) run bandit -q -r . -x ./tests,./.venv
	poetry -C $(CORE) run bandit -q -r src streaming feature_store

# El gate rompe la build en HIGH/CRITICAL y solo REPORTA lo demas. Antes
# usaba --exit-code 1 sobre TODAS las severidades: con 144 hallazgos (63 LOW
# y 39 MEDIUM, buena parte dentro de modulos Terraform de terceros como
# terraform-aws-modules/eks y /vpc, que no podemos editar), el gate era
# imposible de pasar -- `make ci` fallaba siempre y, con el, el stage
# "quality" de .gitlab-ci.yml. Un gate que nadie puede pasar acaba
# desactivado, que es peor que un gate calibrado.
# --ignore-unfixed: no rompe la build por CVEs que aun no tienen version
# corregida publicada (no hay accion posible salvo dejar de usar el paquete).
trivy: ## Escaneo de secretos, vulnerabilidades de dependencias e IaC
	@command -v trivy >/dev/null 2>&1 || { \
		echo "trivy no esta instalado. Instalacion:"; \
		echo "  macOS   : brew install aquasecurity/trivy/trivy"; \
		echo "  Windows : choco install trivy  |  scoop install trivy"; \
		echo "  Linux   : curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin"; \
		exit 1; \
	}
	trivy fs --scanners vuln,secret,misconfig --exit-code 1 \
		--severity HIGH,CRITICAL --ignore-unfixed --ignorefile .trivyignore.yaml \
		--skip-dirs .git,mlruns,models,core_ml/data,core_ml/data_toy,core_ml/.dvc,terraform/.terraform,terraform/bootstrap/.terraform,.venv,api/.venv,core_ml/.venv,.gitlab-ci-local \
		--skip-files terraform/terraform.tfvars,terraform/terraform.tfstate,terraform/terraform.tfstate.backup,terraform/main.tfplan,terraform/bootstrap/terraform.tfstate,terraform/bootstrap/terraform.tfstate.backup \
		.

security: bandit trivy ## Ejecuta bandit + trivy (SAST + secretos + vulnerabilidades + IaC)

## ---------------------------------------------------------------------
## Datos, contratos y reproducibilidad (Fase 2)
## ---------------------------------------------------------------------

build-toy-dataset: ## Regenera el dataset toy (~1000 filas, fijo) desde el dataset completo
	poetry -C $(CORE) run python scripts/build_toy_dataset.py

msk-bootstrap-topics: ## Crea (idempotente) los topics de Kafka; ver docstring del script
	# terraform/msk.tf provisiona el CLUSTER, nunca los topics (no son un
	# recurso de AWS, y MSK esta en subnets privadas sin acceso publico desde
	# donde corre terraform). Correr UNA VEZ por cluster, desde dentro de la
	# VPC -- localmente contra docker-compose (KAFKA_BROKER=localhost:9092),
	# o contra el MSK real via `kubectl exec` en cualquier pod de mlops-env
	# (ya tiene kafka-python-ng y la variable KAFKA_BROKER del ConfigMap).
	poetry -C $(CORE) run python scripts/bootstrap_kafka_topics.py

prepare-toy: ## Construye engine_features/training_entities/scaler.joblib del dataset toy
	poetry -C $(CORE) run python src/prepare_feast_data.py --data-dir data_toy

prepare-data: ## Construye engine_features/training_entities/scaler.joblib del dataset completo
	poetry -C $(CORE) run python src/prepare_feast_data.py --data-dir data

train: ## Entrena con el dataset completo via Feast (produccion; exige el quality gate)
	poetry -C $(CORE) run python src/train.py --data-dir data --data-source feast

# MLFLOW_TRACKING_URI por defecto: un store SQLite local y efimero, igual
# que hace el job smoke_test de .gitlab-ci.yml. Sin esto el comando NO era
# autonomo: config.yaml resuelve ${MLFLOW_TRACKING_URI} y, sin la variable,
# el fallback heredado era "localhost:5000" (una URI SIN esquema), que
# MLflow rechaza con UnsupportedModelRegistryStoreURIException -- pese a que
# README y la guia documentan `make smoke-test` como ejecutable en un
# portatil sin levantar nada previo.
# Se respeta el valor externo si ya esta definido (p.ej. el MLflow de
# docker-compose: export MLFLOW_TRACKING_URI=http://localhost:5000).
train-toy: prepare-toy ## Entrena end-to-end con el dataset toy: segundos, sin GPU/Feast/S3
	MLFLOW_TRACKING_URI="$${MLFLOW_TRACKING_URI:-sqlite:///mlruns/smoke_test.db}" \
	poetry -C $(CORE) run python src/train.py \
		--data-dir data_toy --data-source parquet --no-enforce-quality-gate

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
## Fase 3: contenedores locales (Docker Compose + LocalStack)
## ---------------------------------------------------------------------

compose-up: ## Levanta el stack local completo: Postgres, Kafka, Redis, MLflow, API, LocalStack
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
## Fase 4: build/push de imagenes y release GitOps (sin sed, sin kubectl apply)
## ---------------------------------------------------------------------

docker-build-api: ## Construye la imagen de la API (Dockerfile de la raiz)
	docker build -t $(API_IMAGE):$(IMAGE_TAG) -f Dockerfile .

docker-build-consumer: ## Construye la imagen del streaming consumer (core_ml/Dockerfile)
	docker build -t $(CONSUMER_IMAGE):$(IMAGE_TAG) -f core_ml/Dockerfile .

docker-build-monitoring: ## Construye la imagen del servicio de deteccion de drift (monitoring/Dockerfile)
	docker build -t $(MONITORING_IMAGE):$(IMAGE_TAG) -f monitoring/Dockerfile .

docker-build-mlflow: ## Construye la imagen de MLflow con psycopg2 (Dockerfile.mlflow; backend-store-uri es Postgres/RDS)
	docker build -t $(MLFLOW_IMAGE):$(IMAGE_TAG) -f Dockerfile.mlflow .

docker-build: docker-build-api docker-build-consumer docker-build-monitoring docker-build-mlflow ## Construye las cuatro imagenes (api + consumer + monitoring + mlflow)

docker-push-api: ## Publica la imagen de la API en ECR (requiere `docker login` a ECR ya hecho)
	docker push $(API_IMAGE):$(IMAGE_TAG)

docker-push-consumer: ## Publica la imagen del consumer en ECR
	docker push $(CONSUMER_IMAGE):$(IMAGE_TAG)

docker-push-monitoring: ## Publica la imagen del servicio de monitoring en ECR
	docker push $(MONITORING_IMAGE):$(IMAGE_TAG)

docker-push-mlflow: ## Publica la imagen de MLflow en ECR
	docker push $(MLFLOW_IMAGE):$(IMAGE_TAG)

docker-push: docker-push-api docker-push-consumer docker-push-monitoring docker-push-mlflow ## Publica las cuatro imagenes en ECR

## ---------------------------------------------------------------------
## Fase 4: build+push con cache remoto de BuildKit (para CI, no uso local)
## ---------------------------------------------------------------------
# El servicio dind del job build_image (.gitlab-ci.yml) es efimero: sin esto,
# CADA corrida reconstruye y resube desde cero la capa de dependencias de
# torch (~250MB, compartida por api y streaming-consumer), lo que en un
# ancho de banda de subida domestico limitado hizo fallar build_image por el
# timeout de 1h del job. `docker buildx build --push` con
# --cache-from/--cache-to type=registry persiste esa capa en el repo ECR
# dedicado predictive-maintenance-mlops-build-cache (ver terraform/ecr.tf):
# solo se vuelve a subir cuando el poetry.lock correspondiente cambia
# realmente. docker-build-*/docker-push-* (arriba) se conservan intactos
# para desarrollo local, donde Docker Desktop ya cachea capas entre builds
# y este problema no existe.

# Un UNICO tag de cache compartido entre las 4 imagenes (no uno por imagen):
# api y streaming-consumer instalan la MISMA rueda pesada de torch+CPU (ver
# api/pyproject.toml y core_ml/pyproject.toml, ambos alineados a la misma
# version); con cache separado por imagen, esa capa se resubia dos veces en
# el registro de cache en vez de una. BuildKit reutiliza una capa cacheada
# entre Dockerfiles distintos con normalidad: lo que importa es que la
# instruccion y sus inputs coincidan exactamente, no que sea "la misma
# imagen". Los 4 builds corren secuenciales (ver target docker-buildx-push
# mas abajo), asi que no hay condicion de carrera escribiendo el mismo tag.
# Idempotencia (skip-if-exists): las 4 imagenes se publican en repos ECR
# IMMUTABLE (terraform/ecr.tf) -- correcto para produccion (un tag jamas
# cambia de contenido bajo los pies de un despliegue), pero significa que
# reintentar `docker-buildx-push` tras un fallo PARCIAL (p.ej. build_image
# se cuelga en la imagen 3 de 4, ya con la 1 y 2 subidas con exito) revienta
# con "tag already exists ... cannot be overwritten" al re-intentar
# re-publicar una imagen que ya habia llegado a ECR en el intento anterior.
# Cada target comprueba primero si $(IMAGE_TAG) ya existe en su repo y, si
# es asi, omite el build+push -- necesario para que un retry del job
# build_image (manual o automatico, ver .gitlab-ci.yml) sea seguro.
#
# --cache-to SIN ",mode=max": los 4 Dockerfiles son de una sola etapa (un
# unico FROM, sin build multi-stage) -- "mode=max" solo aporta valor
# exportando capas que un build multi-stage descarta del resultado final,
# algo que aqui no existe. Con un solo stage, el modo por defecto ("min")
# ya cachea exactamente las mismas capas que importan (incluida la de
# poetry install), asi que "mode=max" era trafico extra sin beneficio real
# -- y ademas la fase mas propensa a colgarse contra la red domestica de
# este runner (ver comentario de resiliencia en .gitlab-ci.yml).
docker-buildx-push-api: ## Build+push de la API con cache remoto de BuildKit (CI)
	@if aws ecr describe-images --region $(AWS_REGION) --repository-name predictive-maintenance-mlops-streaming --image-ids imageTag=$(IMAGE_TAG) >/dev/null 2>&1; then \
		echo "predictive-maintenance-mlops-streaming:$(IMAGE_TAG) ya existe -- omitiendo (repo inmutable, retry idempotente)"; \
	else \
		docker buildx build --push \
			--cache-from type=registry,ref=$(BUILD_CACHE_IMAGE):shared \
			--cache-to type=registry,ref=$(BUILD_CACHE_IMAGE):shared \
			-t $(API_IMAGE):$(IMAGE_TAG) -f Dockerfile . ; \
	fi

docker-buildx-push-consumer: ## Build+push del streaming-consumer con cache remoto de BuildKit (CI)
	@if aws ecr describe-images --region $(AWS_REGION) --repository-name predictive-maintenance-mlops-consumer --image-ids imageTag=$(IMAGE_TAG) >/dev/null 2>&1; then \
		echo "predictive-maintenance-mlops-consumer:$(IMAGE_TAG) ya existe -- omitiendo (repo inmutable, retry idempotente)"; \
	else \
		docker buildx build --push \
			--cache-from type=registry,ref=$(BUILD_CACHE_IMAGE):shared \
			--cache-to type=registry,ref=$(BUILD_CACHE_IMAGE):shared \
			-t $(CONSUMER_IMAGE):$(IMAGE_TAG) -f core_ml/Dockerfile . ; \
	fi

docker-buildx-push-monitoring: ## Build+push de monitoring con cache remoto de BuildKit (CI)
	@if aws ecr describe-images --region $(AWS_REGION) --repository-name predictive-maintenance-mlops-monitoring --image-ids imageTag=$(IMAGE_TAG) >/dev/null 2>&1; then \
		echo "predictive-maintenance-mlops-monitoring:$(IMAGE_TAG) ya existe -- omitiendo (repo inmutable, retry idempotente)"; \
	else \
		docker buildx build --push \
			--cache-from type=registry,ref=$(BUILD_CACHE_IMAGE):shared \
			--cache-to type=registry,ref=$(BUILD_CACHE_IMAGE):shared \
			-t $(MONITORING_IMAGE):$(IMAGE_TAG) -f monitoring/Dockerfile . ; \
	fi

# SIN --cache-from/--cache-to: a diferencia de api/consumer/monitoring
# (las 3 sobre python:3.12-slim + poetry, compartiendo la capa pesada de
# torch), esta imagen parte de `ghcr.io/mlflow/mlflow` -- no comparte NINGUNA
# capa con las otras 3, asi que exportar cache aqui no ahorra nada en el
# futuro y solo agregaba una subida extra de varios cientos de MB, siendo
# ademas la que mas se atascaba contra la red domestica inestable del
# runner (ver comentario de resiliencia mas arriba). El build en si ya es
# rapido (imagen base chica, sin poetry install pesado de por medio).
docker-buildx-push-mlflow: ## Build+push de MLflow (sin cache remoto: no comparte capas con las otras 3 imagenes)
	@if aws ecr describe-images --region $(AWS_REGION) --repository-name predictive-maintenance-mlops-mlflow --image-ids imageTag=$(IMAGE_TAG) >/dev/null 2>&1; then \
		echo "predictive-maintenance-mlops-mlflow:$(IMAGE_TAG) ya existe -- omitiendo (repo inmutable, retry idempotente)"; \
	else \
		docker buildx build --push -t $(MLFLOW_IMAGE):$(IMAGE_TAG) -f Dockerfile.mlflow . ; \
	fi

docker-buildx-push: docker-buildx-push-api docker-buildx-push-consumer docker-buildx-push-monitoring docker-buildx-push-mlflow ## Build+push de las 4 imagenes con cache remoto (usado por build_image en CI)

k8s-build: ## Renderiza el overlay de produccion (kustomize build) para inspeccion/dry-run
	kubectl kustomize $(GITOPS_OVERLAY)

k8s-diff: ## Muestra el diff del overlay contra el cluster actual (kubectl diff -k)
	kubectl diff -k $(GITOPS_OVERLAY) || true

gitops-set-image: ## Fija IMAGE_TAG en el overlay de forma declarativa (kustomize edit, NUNCA sed)
	cd $(GITOPS_OVERLAY) && kustomize edit set image \
		$(API_IMAGE)=$(API_IMAGE):$(IMAGE_TAG) \
		$(CONSUMER_IMAGE)=$(CONSUMER_IMAGE):$(IMAGE_TAG) \
		$(MONITORING_IMAGE)=$(MONITORING_IMAGE):$(IMAGE_TAG) \
		$(MLFLOW_IMAGE)=$(MLFLOW_IMAGE):$(IMAGE_TAG)
	@echo "OK: $(GITOPS_OVERLAY)/kustomization.yaml -> tag $(IMAGE_TAG)"

gitops-release: gitops-set-image ## Commitea+pushea el nuevo tag: ArgoCD sincroniza el cluster (GitOps real)
	git diff --quiet -- $(GITOPS_OVERLAY)/kustomization.yaml && echo "Sin cambios de imagen; nada que liberar." && exit 0; \
	git add $(GITOPS_OVERLAY)/kustomization.yaml && \
	git commit -m "chore(gitops): release $(IMAGE_TAG) [skip ci]" && \
	git push origin HEAD:$${CI_COMMIT_REF_NAME:-$$(git rev-parse --abbrev-ref HEAD)}

## ---------------------------------------------------------------------
## Fase 4: Infraestructura como Codigo (Terraform)
## ---------------------------------------------------------------------

terraform-fmt: ## Verifica el formato canonico de todos los .tf (incluye terraform/bootstrap)
	terraform -chdir=terraform fmt -check -diff -recursive

terraform-validate: ## Valida sintaxis/tipos de terraform/ (requiere `terraform init` previo)
	terraform -chdir=terraform validate

terraform-plan: ## Muestra el plan de cambios contra AWS real (requiere credenciales validas)
	terraform -chdir=terraform plan

## ---------------------------------------------------------------------
## Fase 4: validar el pipeline de GitLab CI localmente
## ---------------------------------------------------------------------

ci-local: ## Corre .gitlab-ci.yml localmente con gitlab-ci-local (requiere Docker + Node.js)
	@command -v npx >/dev/null 2>&1 || { \
		echo "Node.js/npx no esta instalado. gitlab-runner ya no trae 'exec' (retirado"; \
		echo "en versiones recientes); la alternativa mantenida por la comunidad es"; \
		echo "gitlab-ci-local: https://github.com/firecow/gitlab-ci-local"; \
		echo "Instala Node.js (nodejs.org) y vuelve a correr 'make ci-local'."; \
		exit 1; \
	}
	MSYS_NO_PATHCONV=1 npx --yes gitlab-ci-local $(JOB)

## ---------------------------------------------------------------------
## Fase 4: runner self-hosted de GitLab CI/CD (cero minutos en la nube)
## ---------------------------------------------------------------------
# Flujo: 1) make runner-register TOKEN=glrt-xxx (una sola vez por maquina)
#        2) make runner-up  (queda corriendo en background, restart automatico)
#        git push -> gitlab.com asigna los jobs (tag local-hardware) a
#        este runner, que los ejecuta en tu propio hardware.

RUNNER_COMPOSE := docker compose -f runner/docker-compose.yml
GITLAB_URL ?= https://gitlab.com

runner-register: ## Registra este host como runner (make runner-register TOKEN=glrt-xxxxx)
	@test -n "$(TOKEN)" || { \
		echo "Uso: make runner-register TOKEN=glrt-xxxxx"; \
		echo "El token se genera en: gitlab.com -> tu proyecto -> Settings > CI/CD > Runners"; \
		echo "-> 'New project runner' -> plataforma Linux, tag 'local-hardware' -> Create runner."; \
		exit 1; \
	}
	$(RUNNER_COMPOSE) run --rm gitlab-runner register \
		--non-interactive \
		--url "$(GITLAB_URL)" \
		--token "$(TOKEN)" \
		--executor "docker" \
		--docker-image "python:3.12-slim" \
		--docker-privileged="true" \
		--description "local-hardware"
	@echo "OK: runner registrado. Ahora: make runner-up"

runner-up: ## Levanta el runner self-hosted (background, restart automatico)
	$(RUNNER_COMPOSE) up -d

runner-down: ## Detiene el runner self-hosted (conserva el token/config registrado)
	$(RUNNER_COMPOSE) down

runner-status: ## Muestra el estado y verifica las credenciales del runner
	@# 'gitlab-runner status' busca un pidfile de instalacion como servicio del
	@# sistema, que no existe corriendo via 'docker compose up -d' (PID 1 en
	@# modo foreground) -- devuelve exit 1 aunque el runner este sano. Se
	@# ignora ese resultado; 'verify' (que SI valida credenciales contra
	@# gitlab.com) es la fuente de verdad real.
	@$(RUNNER_COMPOSE) exec gitlab-runner gitlab-runner status || true
	$(RUNNER_COMPOSE) exec gitlab-runner gitlab-runner verify

runner-logs: ## Sigue los logs del runner self-hosted (Ctrl+C para salir)
	$(RUNNER_COMPOSE) logs -f

runner-unregister: ## Da de baja el runner en GitLab y borra su configuracion local
	$(RUNNER_COMPOSE) exec gitlab-runner gitlab-runner unregister --all-runners || true
	$(RUNNER_COMPOSE) down --volumes

## ---------------------------------------------------------------------
## YAML y git hooks
## ---------------------------------------------------------------------

yamllint: ## Valida todos los YAML del repo (estructura + estilo)
	pre-commit run yamllint --all-files

precommit-install: ## Instala los git hooks de pre-commit en este repo
	pre-commit install --install-hooks
	pre-commit install --hook-type commit-msg || true

precommit-run: ## Ejecuta todos los hooks de pre-commit sobre todo el repo
	pre-commit run --all-files

## ---------------------------------------------------------------------
## Agregados
## ---------------------------------------------------------------------

ci: format-check lint typecheck test security yamllint ## Pipeline completo (el mismo que corre en GitLab CI)
	@echo "OK: todos los checks de calidad pasaron."

clean: ## Limpia caches locales de herramientas
	find . -type d -name "__pycache__" -not -path "*/.venv/*" -prune -exec rm -rf {} +
	rm -rf $(API)/.mypy_cache $(API)/.ruff_cache $(API)/.pytest_cache $(API)/htmlcov
	rm -rf $(CORE)/.mypy_cache $(CORE)/.ruff_cache $(CORE)/.pytest_cache $(CORE)/htmlcov
