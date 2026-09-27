.PHONY: setup db api web ingest eval test lint demo down
setup:
	uv sync --frozen
	cd frontend && npm ci
db:
	docker compose up -d db
api:
	uv run uvicorn app.main:app --reload --port 18080
web:
	cd frontend && npm run dev
ingest:
	uv run python -m app.ingest
eval:
	uv run python -m eval.run --check
test:
	uv run pytest -q
lint:
	uv run ruff check .
demo:
	docker compose up -d --build --wait
	docker compose exec -T api python -m app.ingest
	docker compose exec -T api python -m eval.run --check
	docker compose cp api:/app/reports/. reports/
down:
	docker compose down
