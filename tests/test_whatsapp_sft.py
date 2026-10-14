from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from whatsapp.onboard_cli import _approved_messages, prepare_chat
from whatsapp.parser import parse_whatsapp_chat
from whatsapp.sft import build_sft, candidate_turns
from whatsapp.sft_cli import main as sft_main


def chat_export() -> str:
    lines = []
    for turn in range(18):
        minute = turn * 2
        lines.append(f"[08/08/25, 10:{minute:02d} AM] Rohan: kal ke plan {turn} ka kya scene hai?")
        lines.append(f"[08/08/25, 10:{minute + 1:02d} AM] Asha: haan plan {turn} dekhte hain, thoda late aaungi")
    return "\n".join(lines)


class SFTPreparationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workspace = Path(self.temp.name) / "reviewed"
        source = Path(self.temp.name) / "chat.txt"
        source.write_text(chat_export(), encoding="utf-8")
        manifest = prepare_chat(source, self.workspace, target_author="Asha",
                                user_id="u", contact_id="c", chat_id="chat",
                                timezone_name="Asia/Kolkata", all_messages=True)
        review = self.workspace / manifest["review_files"][0]
        rows = [json.loads(line) for line in review.read_text().splitlines()]
        rows[1]["include"] = False
        review.write_text("".join(json.dumps(row) + "\n" for row in rows))
        _, review_hash = _approved_messages(self.workspace, manifest)
        (self.workspace / "index_report.json").write_text(json.dumps({"review_hash": review_hash}))

    def test_single_build_writes_real_answers_and_chronological_holdout(self) -> None:
        class Model:
            def __call__(self, prompt: str, *, max_tokens: int) -> str:
                self_prompt = prompt
                assert "Real reply:" in self_prompt
                return '{"query":"kal kis time aaogi?","fit":"good"}'

        report = build_sft(self.workspace, model=Model(), query_limit=1)
        self.assertEqual(1, report["synthetic_queries"])
        self.assertEqual(1, report["query_model_calls"])
        self.assertGreater(report["examples"]["train"], 0)
        self.assertGreater(report["examples"]["validation"], 0)
        self.assertGreater(report["examples"]["test"], 0)
        output = self.workspace / "sft"
        train = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
        heldout = [json.loads(line) for line in (output / "test.jsonl").read_text().splitlines()]
        self.assertIn("kal kis time aaogi?", train[0]["messages"][1]["content"])
        self.assertEqual("haan plan 1 dekhte hain, thoda late aaungi",
                         train[0]["messages"][2]["content"])
        self.assertNotIn("plan 0 dekhte", json.dumps(train))
        self.assertIn("Rohan:", heldout[0]["messages"][1]["content"])
        provenance = [json.loads(line) for line in
                      (output / "provenance.jsonl").read_text().splitlines()]
        self.assertEqual("synthetic", provenance[0]["variant"])
        self.assertTrue(all(item["variant"] == "real" for item in provenance
                            if item["split"] != "train"))

    def test_bad_query_falls_back_to_real_context_and_source_hash_is_enforced(self) -> None:
        class PoorModel:
            def __call__(self, prompt: str, *, max_tokens: int) -> str:
                return '{"query":"irrelevant prompt","fit":"poor"}'

        report = build_sft(self.workspace, model=PoorModel(), query_limit=2)
        self.assertEqual(0, report["synthetic_queries"])
        self.assertEqual(2, report["query_model_calls"])
        first = json.loads((self.workspace / "sft" / "train.jsonl").read_text().splitlines()[0])
        self.assertIn("Rohan:", first["messages"][1]["content"])
        with self.assertRaises(FileExistsError):
            build_sft(self.workspace, model=None)
        review = next((self.workspace / "review").glob("part-*.jsonl"))
        reviewed = [json.loads(line) for line in review.read_text().splitlines()]
        reviewed[-1]["text"] = "changed after indexing"
        review.write_text("".join(json.dumps(row) + "\n" for row in reviewed))
        with self.assertRaisesRegex(ValueError, "differs from the indexed review"):
            build_sft(self.workspace, output=Path(self.temp.name) / "another", model=None)

    def test_excluded_incoming_message_does_not_create_a_false_pair(self) -> None:
        messages = parse_whatsapp_chat(chat_export(), chat_id="chat",
                                       timezone_name="Asia/Kolkata")
        without_prompt = [message for message in messages if message.sequence != 18]
        turns = candidate_turns(without_prompt, "Asha")
        self.assertFalse(any("plan 9 dekhte" in row["response"] for row in turns))

    def test_cli_real_only_is_one_command(self) -> None:
        output = io.StringIO()
        with (patch("sys.argv", ["prepare-sft", str(self.workspace), "--real-only"]),
              redirect_stdout(output)):
            sft_main()
        report = json.loads(output.getvalue())
        self.assertEqual(0, report["query_model_calls"])
        self.assertTrue((self.workspace / "sft" / "train.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
