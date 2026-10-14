from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from whatsapp import parse_whatsapp_chat
from whatsapp.authors import resolve_target_author
from whatsapp.eval import (
    DEFAULT_OVERLAP_THRESHOLD,
    DatasetError,
    EvalExample,
    duplicate_rate,
    evaluate_modes,
    load_dataset,
    ndcg_at_k,
    render_table,
    report_to_dict,
    resolve_scope,
    score_query,
    target_overlap,
)
from whatsapp.retrieval import WhatsAppSearchHit


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "synthetic_whatsapp_chat.txt"
SYNTHETIC_LABELS = Path(__file__).resolve().parents[1] / "evals" / "whatsapp.synthetic.jsonl"
FIXTURE_CHAT_ID = "eval-synthetic"


def hit(
    point_id: str,
    message_ids: tuple[str, ...],
    *,
    target_ids: tuple[str, ...] | None = None,
    rank: int = 1,
) -> WhatsAppSearchHit:
    target_ids = target_ids if target_ids is not None else (message_ids[-1],)
    return WhatsAppSearchHit(
        point_id=point_id,
        rank=rank,
        score=0.9 - rank / 100,
        schema_version="v",
        user_id="u",
        persona_id="p",
        source="whatsapp",
        target_author="Target",
        requested_target_author="Target",
        chat_id="c",
        chunk_id=f"chunk-{point_id}",
        started_at="",
        ended_at="",
        start_sequence=1,
        end_sequence=2,
        message_ids=message_ids,
        target_message_ids=target_ids,
        authors=("Target",),
        text="",
        context=(),
        target_responses=(),
    )


def example(query: str, expected: tuple[str, ...], query_type: str = "paraphrase") -> EvalExample:
    return EvalExample(
        line_number=1,
        query=query,
        user_id="u",
        persona_id="p",
        expected_target_message_ids=expected,
        query_type=query_type,
    )


class MetricTests(unittest.TestCase):
    def test_relevant_message_at_rank_one(self) -> None:
        hits = [hit("a", ("m1",)), hit("b", ("m2",))]
        metrics = score_query(hits, frozenset({"m1"}), k=2)
        self.assertEqual(1.0, metrics.success_at_k)
        self.assertEqual(1.0, metrics.recall_at_k)
        self.assertEqual(1.0, metrics.mrr_at_k)
        self.assertEqual((1,), metrics.relevant_ranks)

    def test_relevant_message_at_rank_five(self) -> None:
        hits = [hit(f"p{index}", (f"x{index}",)) for index in range(1, 5)]
        hits.append(hit("p5", ("m1",), rank=5))
        metrics = score_query(hits, frozenset({"m1"}), k=5)
        self.assertEqual(1.0, metrics.success_at_k)
        self.assertAlmostEqual(0.2, metrics.mrr_at_k)
        self.assertEqual((5,), metrics.relevant_ranks)

    def test_no_relevant_result(self) -> None:
        hits = [hit("a", ("x1",)), hit("b", ("x2",))]
        metrics = score_query(hits, frozenset({"m1"}), k=2)
        self.assertEqual(0.0, metrics.success_at_k)
        self.assertEqual(0.0, metrics.recall_at_k)
        self.assertEqual(0.0, metrics.mrr_at_k)
        self.assertEqual(0.0, metrics.ndcg_at_k)
        self.assertEqual((), metrics.relevant_ranks)

    def test_multiple_expected_messages_and_overlapping_chunks_count_once(self) -> None:
        # m1 and m2 each appear in two overlapping chunks but count toward recall once.
        hits = [
            hit("a", ("m1", "m2"), target_ids=("m1", "m2")),
            hit("b", ("m1", "m2", "m3"), target_ids=("m1", "m2")),
            hit("c", ("x",)),
        ]
        metrics = score_query(hits, frozenset({"m1", "m2"}), k=3)
        self.assertEqual(1.0, metrics.recall_at_k)
        self.assertEqual(("m1", "m2"), metrics.matched_message_ids)
        self.assertEqual(2, len(metrics.matched_message_ids))

    def test_k_cutoff_limits_recall_and_success(self) -> None:
        hits = [hit("a", ("x",)), hit("b", ("m1",))]
        self.assertEqual(0.0, score_query(hits, frozenset({"m1"}), k=1).success_at_k)
        self.assertEqual(1.0, score_query(hits, frozenset({"m1"}), k=2).success_at_k)

    def test_ndcg_at_k(self) -> None:
        self.assertEqual(1.0, ndcg_at_k([1, 0, 0], k=3, ideal_count=1))
        # One expected message at rank 3 with K=3: DCG = 1/log2(4) = 0.5.
        self.assertAlmostEqual(0.5, ndcg_at_k([0, 0, 1], k=3, ideal_count=1))
        # One expected message at rank 5 with K=5: DCG = 1/log2(6).
        self.assertAlmostEqual(1 / math.log2(6), ndcg_at_k([0, 0, 0, 0, 1], k=5, ideal_count=1))
        self.assertEqual(0.0, ndcg_at_k([0, 0, 0], k=3, ideal_count=1))
        # Two expected messages: ranks 1 and 2 are the ideal ordering.
        self.assertEqual(1.0, ndcg_at_k([1, 1, 0], k=3, ideal_count=2))
        self.assertLess(ndcg_at_k([0, 1, 0], k=3, ideal_count=2), 1.0)

    def test_duplicate_rate_threshold(self) -> None:
        # 4-message chunk then a 3-message chunk sharing 2 ids -> 2/min(4,3)=0.67.
        hits = [hit("a", ("m1", "m2", "m3", "m4")), hit("b", ("m1", "m2", "m5"))]
        self.assertEqual(0.0, duplicate_rate(hits, overlap_threshold=0.7))
        self.assertEqual(0.5, duplicate_rate(hits, overlap_threshold=0.6))
        self.assertEqual(0.0, duplicate_rate([], overlap_threshold=0.7))

    def test_evidence_scope_any_author_vs_target_only(self) -> None:
        # The expected message was written by the *other* speaker: it is present
        # in message_ids but not in target_message_ids.
        hits = [hit("a", ("other-1", "target-1"), target_ids=("target-1",))]

        any_author = score_query(
            hits, frozenset({"other-1"}), k=1, evidence_scope="any_author"
        )
        self.assertEqual(1.0, any_author.success_at_k)
        self.assertEqual(("other-1",), any_author.matched_message_ids)

        target_only = score_query(
            hits, frozenset({"other-1"}), k=1, evidence_scope="target_only"
        )
        self.assertEqual(0.0, target_only.success_at_k)

    def test_invalid_evidence_scope_is_rejected(self) -> None:
        hits = [hit("a", ("m1",))]
        with self.assertRaisesRegex(ValueError, "evidence_scope"):
            score_query(hits, frozenset({"m1"}), k=1, evidence_scope="bogus")

    def test_target_overlap_uses_smaller_chunk(self) -> None:
        self.assertEqual(1.0, target_overlap(("m1", "m2"), ("m1", "m2", "m3", "m4")))
        self.assertEqual(0.5, target_overlap(("m1", "m9"), ("m1", "m2")))
        self.assertEqual(0.0, target_overlap((), ("m1",)))


class DatasetTests(unittest.TestCase):
    def _write(self, directory: str, text: str) -> Path:
        path = Path(directory) / "labels.jsonl"
        path.write_text(text, encoding="utf-8")
        return path

    def test_loads_and_normalizes_valid_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(
                directory,
                '{"query": "q", "user_id": "u", "persona_id": "p", '
                '"expected_target_message_ids": ["m1", "m1", "m2"]}\n',
            )
            examples = load_dataset(path)
        self.assertEqual(1, len(examples))
        self.assertEqual(("m1", "m2"), examples[0].expected_target_message_ids)
        self.assertIsNone(examples[0].query_type)

    def test_missing_and_malformed_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(DatasetError, "does not exist"):
                load_dataset(Path(directory) / "nope.jsonl")
            with self.assertRaisesRegex(DatasetError, "no labelled examples"):
                load_dataset(self._write(directory, "\n\n"))
            with self.assertRaisesRegex(DatasetError, "Invalid JSON"):
                load_dataset(self._write(directory, "{not json}\n"))
            with self.assertRaisesRegex(DatasetError, "query"):
                load_dataset(
                    self._write(directory, '{"user_id":"u","persona_id":"p","expected_target_message_ids":["m1"]}\n')
                )
            with self.assertRaisesRegex(DatasetError, "expected_target_message_ids"):
                load_dataset(
                    self._write(directory, '{"query":"q","user_id":"u","persona_id":"p","expected_target_message_ids":[]}\n')
                )
            with self.assertRaisesRegex(DatasetError, "must be a list"):
                load_dataset(
                    self._write(directory, '{"query":"q","user_id":"u","persona_id":"p","expected_target_message_ids":"m1"}\n')
                )

    def test_template_placeholders_are_rejected(self) -> None:
        checks = (
            '{"query": "q", "user_id": "REPLACE_WITH_YOUR_USER_ID", "persona_id": "p", "expected_target_message_ids": ["m1"]}\n',
            '{"query": "q", "user_id": "u", "persona_id": "REPLACE_WITH_YOUR_PERSONA_ID", "expected_target_message_ids": ["m1"]}\n',
            '{"query": "PASTE_AN_UNUSUAL_PHRASE_FROM_THE_CHAT", "user_id": "u", "persona_id": "p", "expected_target_message_ids": ["m1"]}\n',
            '{"query": "q", "user_id": "u", "persona_id": "p", "expected_target_message_ids": ["REPLACE_WITH_TARGET_MESSAGE_ID"]}\n',
        )
        with tempfile.TemporaryDirectory() as directory:
            for text in checks:
                with self.assertRaisesRegex(DatasetError, "placeholder"):
                    load_dataset(self._write(directory, text))

    def test_alias_key_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(
                directory,
                '{"query": "q", "user_id": "u", "persona_id": "p", "relevant_message_ids": ["m1"]}\n',
            )
            examples = load_dataset(path)
        self.assertEqual(("m1",), examples[0].expected_target_message_ids)

    def test_scope_resolution_and_mismatch(self) -> None:
        rows = [example("q1", ("m1",)), example("q2", ("m2",))]
        self.assertEqual(("u", "p"), resolve_scope(rows))
        self.assertEqual(("u", "p"), resolve_scope(rows, user_id="u", persona_id="p"))
        with self.assertRaisesRegex(DatasetError, "must be supplied together"):
            resolve_scope(rows, user_id="u")
        with self.assertRaisesRegex(DatasetError, "does not match"):
            resolve_scope(rows, user_id="other", persona_id="p")
        mixed = [
            EvalExample(1, "q1", "u1", "p1", ("m1",)),
            EvalExample(2, "q2", "u2", "p2", ("m2",)),
        ]
        with self.assertRaisesRegex(DatasetError, "multiple"):
            resolve_scope(mixed)


class FakeRetriever:
    """Mode-aware fake; higher modes rank the relevant hit correctly."""

    def __init__(self, mapping):
        self.mapping = mapping
        self.calls: list[dict] = []

    def search(self, query, **kwargs):
        self.calls.append({"query": query, **kwargs})
        return self.mapping[kwargs["mode"]][query]


class EvaluationTests(unittest.TestCase):
    def test_evaluate_modes_aggregates_and_exposes_failures(self) -> None:
        mapping = {
            "dense": {"good": [hit("d1", ("m1",), rank=1)], "bad": [hit("d2", ("x",), rank=1)]},
            "sparse": {"good": [hit("s1", ("m1",), rank=1)], "bad": [hit("s2", ("x",), rank=1)]},
            "hybrid_rrf": {
                "good": [hit("r1", ("x",), rank=1), hit("r2", ("m1",), rank=2)],
                "bad": [hit("r3", ("x",), rank=1)],
            },
            "hybrid_dbsf": {"good": [hit("f1", ("m1",), rank=1)], "bad": [hit("f2", ("x",), rank=1)]},
        }
        ticks = iter(range(0, 100000))
        retriever = FakeRetriever(mapping)
        report = evaluate_modes(
            [example("good", ("m1",)), example("bad", ("m1",), "situation")],
            retriever,
            collection_name="synthetic",
            user_id="u",
            persona_id="p",
            k=2,
            prefetch_limit=4,
            clock=lambda: next(ticks) / 1000.0,
        )
        self.assertEqual(2, report.config["query_count"])
        self.assertEqual(0.5, report.aggregate["dense"].success_at_k)
        self.assertEqual(0.5, report.aggregate["dense"].recall_at_k)
        self.assertAlmostEqual(0.25, report.aggregate["hybrid_rrf"].mrr_at_k)
        self.assertGreater(report.aggregate["dense"].latency_p95_ms, 0.0)
        self.assertEqual("situation", report.per_query[1].query_type)
        self.assertEqual(0.0, report.per_query[1].modes["dense"].metrics.success_at_k)
        self.assertEqual("d2", report.per_query[1].modes["dense"].results[0].point_id)
        # Every mode call preserves the mandatory user/persona scope and rank cutoff.
        self.assertEqual(8, len(retriever.calls))
        self.assertTrue(all(call["user_id"] == "u" and call["persona_id"] == "p" for call in retriever.calls))
        self.assertTrue(all(call["limit"] == 2 and call["prefetch_limit"] == 4 for call in retriever.calls))
        self.assertEqual(
            {"dense", "sparse", "hybrid_rrf", "hybrid_dbsf"},
            {call["mode"] for call in retriever.calls},
        )

    def test_zero_result_run_emits_an_explicit_warning(self) -> None:
        mapping = {
            mode: {"q": []}
            for mode in ("dense", "sparse", "hybrid_rrf", "hybrid_dbsf")
        }
        report = evaluate_modes(
            [example("q", ("m1",))],
            FakeRetriever(mapping),
            collection_name="synthetic",
            user_id="wrong-user",
            persona_id="p",
        )
        self.assertTrue(report.warnings)
        self.assertIn("zero results", report.warnings[0])
        self.assertEqual(0.0, report.aggregate["dense"].success_at_k)
        self.assertIn("warnings", report_to_dict(report))
        self.assertIn("WARNING", render_table(report))

    def test_evaluate_modes_validates_options(self) -> None:
        retriever = FakeRetriever({mode: {} for mode in ("dense", "sparse", "hybrid_rrf", "hybrid_dbsf")})
        base = dict(collection_name="c", user_id="u", persona_id="p")
        with self.assertRaisesRegex(ValueError, "k must be"):
            evaluate_modes([example("q", ("m1",))], retriever, k=0, **base)
        with self.assertRaisesRegex(ValueError, "prefetch_limit must be greater"):
            evaluate_modes([example("q", ("m1",))], retriever, k=5, prefetch_limit=4, **base)
        with self.assertRaisesRegex(ValueError, "unknown retrieval mode"):
            evaluate_modes([example("q", ("m1",))], retriever, modes=("dense", "manual"), **base)
        with self.assertRaisesRegex(ValueError, "at least one labelled example"):
            evaluate_modes([], retriever, **base)


class CliValidationTests(unittest.TestCase):
    def _dataset(self, directory: str, text: str) -> str:
        path = Path(directory) / "labels.jsonl"
        path.write_text(text, encoding="utf-8")
        return str(path)

    def test_cli_rejects_missing_and_empty_expected_ids(self) -> None:
        from whatsapp import eval_cli

        with tempfile.TemporaryDirectory() as directory:
            missing = str(Path(directory) / "absent.jsonl")
            with self.assertRaises(SystemExit):
                eval_cli.main(["--dataset", missing])
            empty = self._dataset(
                directory,
                '{"query":"q","user_id":"u","persona_id":"p","expected_target_message_ids":[]}\n',
            )
            with self.assertRaises(SystemExit):
                eval_cli.main(["--dataset", empty])

    def test_cli_rejects_invalid_k_before_searching(self) -> None:
        from whatsapp import eval_cli

        with tempfile.TemporaryDirectory() as directory:
            path = self._dataset(
                directory,
                '{"query":"q","user_id":"u","persona_id":"p","expected_target_message_ids":["m1"]}\n',
            )
            with self.assertRaises(SystemExit):
                eval_cli.main(["--dataset", path, "--collection", "synthetic", "--k", "0"])


class SyntheticFixtureTests(unittest.TestCase):
    def test_labels_reference_target_authored_messages(self) -> None:
        messages = parse_whatsapp_chat(
            FIXTURE.read_text(encoding="utf-8"), chat_id=FIXTURE_CHAT_ID
        )
        target_author = resolve_target_author(messages, "Replica A")
        target_ids = {m.message_id for m in messages if m.author == target_author}
        examples = load_dataset(SYNTHETIC_LABELS)
        self.assertEqual(6, len(examples))
        self.assertEqual({"exact_phrase", "paraphrase", "situation", "style"}, {
            example.query_type for example in examples
        })
        for example in examples:
            self.assertTrue(
                set(example.expected_target_message_ids) <= target_ids,
                msg=f"labels for {example.query!r} are not all target-authored",
            )
            self.assertEqual((example.user_id, example.persona_id), ("eval-user", "replica-a"))


if __name__ == "__main__":
    unittest.main()
