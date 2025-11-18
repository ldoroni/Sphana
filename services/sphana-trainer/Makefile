.PHONY: install lint fmt test unit integration e2e ingest train-embedding train-relation train-gnn workflow dataset-validate

install:
	python -m pip install -r requirements.txt

lint:
	ruff check src tests

fmt:
	ruff format src tests

unit:
	pytest src/tests/unit

integration:
	pytest src/tests/integration

e2e:
	pytest src/tests/e2e

test:
	pytest

ingest:
	sphana-trainer ingest --config configs/ingest/base.yaml

train-embedding:
	sphana-trainer train embedding --config configs/embedding/base.yaml

train-relation:
	sphana-trainer train relation --config configs/relation/base.yaml

train-gnn:
	sphana-trainer train gnn --config configs/gnn/base.yaml

workflow:
	sphana-trainer workflow run --ingest-config configs/ingest/base.yaml --embedding-config configs/embedding/base.yaml --relation-config configs/relation/base.yaml --gnn-config configs/gnn/base.yaml --export-config configs/export/base.yaml --package-config configs/export/base.yaml

dataset-validate:
	sphana-trainer dataset-validate src/tests/data/embedding/train.jsonl --type embedding

dataset-stats:
	sphana-trainer dataset-stats src/tests/data/embedding/train.jsonl

dataset-from-ingest:
	sphana-trainer dataset-build-from-ingest target/wiki/ingest --output-dir target/datasets/wiki

dataset-download-wiki:
	sphana-trainer dataset-download-wiki --titles-file samples/ai-ml-wiki-titles.small.txt --output target/data/wiki/docs.jsonl

wiki-ingest:
	sphana-trainer ingest --config configs/ingest/wiki.yaml --force

wiki-datasets:
	sphana-trainer dataset-build-from-ingest target/wiki/ingest --output-dir target/datasets/wiki --min-confidence 0.35

wiki-train:
	sphana-trainer train embedding --config configs/embedding/wiki.yaml
	sphana-trainer train relation --config configs/relation/wiki.yaml
	sphana-trainer train gnn --config configs/gnn/wiki.yaml

sweep-embedding:
	sphana-trainer train sweep embedding --config configs/embedding/base.yaml --lr 2e-5 --lr 5e-5 --batch-size 16 32 --temperature 0.05 0.07

