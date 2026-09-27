"""Run the fixed demo corpus evaluation and write JSON + Markdown reports."""

import argparse
import json
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import psycopg
from psycopg import sql
from psycopg.conninfo import make_conninfo

from app.config import get_settings
from app.db import connection, initialize
from app.service import RAGService

ROOT = Path(__file__).resolve().parents[1]


def metrics(rows: list[dict]) -> dict:
    supported = [r for r in rows if not r["expected_refusal"]]
    unsupported = [r for r in rows if r["expected_refusal"]]
    return {
        "questions": len(rows),
        "source_hit_rate": sum(r["source_hit"] for r in supported) / max(len(supported), 1),
        "answer_match_rate": sum(r["answer_match"] for r in supported) / max(len(supported), 1),
        "faithfulness_proxy": sum(r["grounded"] for r in supported) / max(len(supported), 1),
        "refusal_accuracy": sum(r["refused"] for r in unsupported) / max(len(unsupported), 1),
    }


@contextmanager
def evaluation_database(settings):
    """Keep a fixed benchmark corpus separate from uploaded user documents."""
    schema = "eval_" + uuid.uuid4().hex
    with psycopg.connect(settings.database_url, autocommit=True) as admin:
        admin.execute("CREATE EXTENSION IF NOT EXISTS vector")
        admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        try:
            yield settings.model_copy(
                update={
                    "database_url": make_conninfo(
                        settings.database_url, options=f"-csearch_path={schema},public"
                    )
                }
            )
        finally:
            admin.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


def evaluate(settings):
    initialize(settings)
    service = RAGService(settings)
    for path in sorted((ROOT / "data/handbook").glob("*.md")):
        service.ingest(path.name, path.read_text())
    with connection(settings) as conn:
        bodies = {r["id"]: r["body"] for r in conn.execute("SELECT id, body FROM chunks")}
    rows = []
    for case in json.loads((ROOT / "eval/questions.json").read_text()):
        response = service.query(case["question"])
        citations = response["citations"]
        rows.append(
            {
                **case,
                "expected_refusal": case.get("expected_refusal", False),
                "source_hit": bool(
                    set(case["expected_sources"]) & set(response["retrieved_sources"])
                ),
                "answer_match": case.get("expected_text", "").lower() in response["answer"].lower(),
                "grounded": bool(citations)
                and all(c["quote"] in bodies.get(c["chunk_id"], "") for c in citations),
                **response,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports")
    parser.add_argument("--check", action="store_true", help="Fail below documented MVP gates")
    args = parser.parse_args()
    settings = get_settings()
    with evaluation_database(settings) as isolated:
        rows = evaluate(isolated)
    summary = metrics(rows)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "mock" if settings.mock_llm else "provider",
        "embedding": settings.embedding_signature,
        "threshold": settings.similarity_threshold,
        "top_k": settings.top_k,
        "metrics": summary,
        "results": rows,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "eval.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# DocuPilot evaluation",
        "",
        f"Mode: {report['mode']}",
        "",
        "Faithfulness proxy checks exact quotes against stored chunks; it does not prove "
        "that every answer claim is entailed. This is a small synthetic smoke benchmark.",
        "",
    ]
    lines += [
        f"- {key}: {value:.3f}" if isinstance(value, float) else f"- {key}: {value}"
        for key, value in summary.items()
    ]
    lines += ["", "## Per-question results", ""]
    for row in rows:
        passed = (
            row["refused"]
            if row["expected_refusal"]
            else row["source_hit"] and row["grounded"] and row["answer_match"]
        )
        lines.append(f"- {'PASS' if passed else 'FAIL'}: {row['question']}")
    (args.output / "eval.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))
    if args.check and not (
        summary["source_hit_rate"] >= 0.9
        and summary["faithfulness_proxy"] == 1
        and summary["answer_match_rate"] >= 0.8
        and summary["refusal_accuracy"] == 1
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
