.PHONY: help install lint format test test-gate generate-data train stage1 stage2 stage3 stage4 stage5 stage6 docker-up docker-down clean

help:
	@echo "Available commands:"
	@echo "  make install        Install package and dev dependencies with uv"
	@echo "  make lint           Check code formatting and run ruff and mypy"
	@echo "  make format         Auto-format code with ruff"
	@echo "  make test           Run all pytest unit and integration tests"
	@echo "  make test-gate      Run promotion gate unit tests specifically"
	@echo "  make generate-data  Generate synthetic demo dataset for local runs"
	@echo "  make train          Run full end-to-end training pipeline flow"
	@echo "  make stage1         Run Stage 1: Ingest & Validate"
	@echo "  make stage2         Run Stage 2: Preprocess & Split"
	@echo "  make stage3         Run Stage 3: Model Training"
	@echo "  make stage4         Run Stage 4: Test Evaluation & Baseline Gate"
	@echo "  make stage5         Run Stage 5: MLflow Model Registration"
	@echo "  make stage6         Run Stage 6: ONNX Export & Parity Check"
	@echo "  make docker-up      Start local MLflow server via docker-compose"
	@echo "  make docker-down    Stop docker-compose containers"
	@echo "  make clean          Clean temporary files and test caches"

install:
	uv pip install -e ".[dev]"

lint:
	ruff check .
	ruff format --check .
	mypy src/

format:
	ruff check --fix .
	ruff format .

test:
	pytest tests/ -v --cov=src/pipeline

test-gate:
	pytest tests/unit/test_eval_gate.py -v

generate-data:
	python scripts/generate_synthetic_data.py --samples-per-class 25

train:
	python -m src.pipeline.flow --config configs/config.yaml

stage1:
	python -m src.pipeline.stages.s01_ingest_validate --config configs/config.yaml

stage2:
	python -m src.pipeline.stages.s02_preprocess_split --config configs/config.yaml

stage3:
	python -m src.pipeline.stages.s03_train --config configs/config.yaml

stage4:
	python -m src.pipeline.stages.s04_eval_gate --config configs/config.yaml

stage5:
	python -m src.pipeline.stages.s05_register --config configs/config.yaml

stage6:
	python -m src.pipeline.stages.s06_export_package --config configs/config.yaml

docker-up:
	docker compose -f docker/docker-compose.yml up -d mlflow

docker-down:
	docker compose -f docker/docker-compose.yml down

clean:
	python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in (list(pathlib.Path('.').rglob('__pycache__')) + [pathlib.Path(d) for d in ['.pytest_cache', '.coverage', 'htmlcov', '.mypy_cache', '.ruff_cache']])]"
