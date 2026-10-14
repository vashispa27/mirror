from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from whatsapp.onboard_cli import _approved_messages, prepare_chat


def synthetic_export() -> str:
    lines = []
    for minute in range(10):
        author = "Asha" if minute % 2 == 0 else "Rohan"
        lines.append(f"[08/08/25, 10:{minute:02d} AM] {author}: message {minute}")
    return "\n".join(lines)


class OnboardPreparationTests(unittest.TestCase):
    def test_recent_selection_review_edits_and_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "chat.txt"
            source.write_text(synthetic_export(), encoding="utf-8")
            workspace = root / "reviewed"
            manifest = prepare_chat(source, workspace, target_author="Asha",
                                    user_id="u", contact_id="c", chat_id="chat",
                                    timezone_name="Asia/Kolkata", review_batch_size=2)
            self.assertEqual(10, manifest["parsed_messages"])
            self.assertEqual(3, manifest["selected_messages"])
            self.assertEqual(2, len(manifest["review_files"]))
            self.assertEqual(10, len((workspace / "cleaned_all.jsonl").read_text().splitlines()))
            path = workspace / manifest["review_files"][0]
            entries = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([7, 8], [entry["sequence"] for entry in entries])
            entries[0]["include"] = False
            entries[1]["text"] = "redacted"
            path.write_text("".join(json.dumps(row) + "\n" for row in entries))
            approved, _ = _approved_messages(workspace, manifest)
            self.assertEqual([8, 9], [message.sequence for message in approved])
            self.assertEqual("redacted", approved[0].text)
            entries.pop()
            path.write_text(json.dumps(entries[0]) + "\n")
            with self.assertRaisesRegex(ValueError, "removed or duplicated"):
                _approved_messages(workspace, manifest)

    def test_all_mode_and_ambiguous_target_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "chat.txt"
            source.write_text(synthetic_export(), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "No export author"):
                prepare_chat(source, root / "bad", target_author="Unknown",
                             user_id="u", contact_id="c", chat_id="chat",
                             timezone_name="Asia/Kolkata")
            self.assertFalse((root / "bad").exists())
            manifest = prepare_chat(source, root / "all", target_author="Asha",
                                    user_id="u", contact_id="c", chat_id="chat",
                                    timezone_name="Asia/Kolkata", all_messages=True)
            self.assertEqual(10, manifest["selected_messages"])


if __name__ == "__main__":
    unittest.main()
