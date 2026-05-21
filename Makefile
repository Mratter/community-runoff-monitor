.PHONY: install run ingest features train backtest test lint

install:
	python -m pip install -r requirements.txt

run:
	streamlit run app.py

ingest:
	python scripts/ingest.py

features:
	python scripts/build_features.py

train:
	python scripts/train_model.py

backtest:
	python scripts/backtest_model.py

test:
	python -m pytest -q

lint:
	python -m compileall app.py src scripts tests

