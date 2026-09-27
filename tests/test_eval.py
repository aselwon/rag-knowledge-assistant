import json
from pathlib import Path

from eval.run import metrics


def test_eval_contains_twenty_source_labelled_questions_and_negatives():
    cases = json.loads(Path("eval/questions.json").read_text())
    assert len([c for c in cases if c["expected_sources"]]) >= 20
    assert len([c for c in cases if c.get("expected_refusal")]) >= 3
    for case in cases:
        for source in case["expected_sources"]:
            assert (Path("data/handbook") / source).is_file()


def test_metrics_penalize_missing_evidence_and_separate_refusals():
    summary = metrics(
        [
            dict(expected_refusal=False, source_hit=True, answer_match=True, grounded=True),
            dict(expected_refusal=False, source_hit=False, answer_match=False, grounded=False),
            dict(expected_refusal=True, refused=True),
        ]
    )
    assert summary["source_hit_rate"] == 0.5
    assert summary["faithfulness_proxy"] == 0.5
    assert summary["answer_match_rate"] == 0.5
    assert summary["refusal_accuracy"] == 1
