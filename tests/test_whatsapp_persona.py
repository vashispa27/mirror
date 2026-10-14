from __future__ import annotations

import json
import unittest
from dataclasses import replace

from whatsapp import build_persona_records, chunk_conversation, parse_whatsapp_chat
from whatsapp.authors import AuthorResolutionError, resolve_target_author
from whatsapp.store import (
    DEFAULT_COLLECTION,
    LEGACY_COLLECTION,
    WhatsAppQdrantStore,
    stable_point_id,
)


CHAT = """[08/08/25, 11:34 AM] Rohan Verma: First response, typo and emoji 🙂!
[08/08/25, 11:35 AM] Aditi: Context that should remain separate.
[08/08/25, 11:36 AM] Rohan Verma: Second response.
[08/08/25, 4:00 PM] Aditi: A later context-only session."""


class AuthorResolutionTests(unittest.TestCase):
    def _messages(self, authors: list[str]):
        lines = [
            f"[08/08/25, 11:{index:02d} AM] {author}: hello"
            for index, author in enumerate(authors)
        ]
        return parse_whatsapp_chat("\n".join(lines), chat_id="authors")

    def test_casefolded_exact_match(self) -> None:
        self.assertEqual(
            "Rohan", resolve_target_author(self._messages(["Rohan"]), "rohan")
        )

    def test_unique_partial_name_match(self) -> None:
        self.assertEqual(
            "Rohan Verma",
            resolve_target_author(self._messages(["Rohan Verma"]), "Rohan"),
        )

    def test_partial_name_ambiguity_is_rejected(self) -> None:
        with self.assertRaisesRegex(AuthorResolutionError, "ambiguous"):
            resolve_target_author(
                self._messages(["Rohan Verma", "Rohan Sharma"]), "Rohan"
            )

    def test_unknown_author_lists_available_authors(self) -> None:
        with self.assertRaisesRegex(AuthorResolutionError, "Rohan Verma"):
            resolve_target_author(self._messages(["Rohan Verma", "Aditi"]), "Other")

    def test_unicode_whitespace_and_casing_are_normalized(self) -> None:
        self.assertEqual(
            "Élodie Durand",
            resolve_target_author(
                self._messages(["Élodie Durand"]), "  e\u0301LODIE\tDURAND  "
            ),
        )

    def test_literal_selector_does_not_treat_regex_characters_as_regex(self) -> None:
        with self.assertRaisesRegex(AuthorResolutionError, "No export author"):
            resolve_target_author(self._messages(["Rohan Verma"]), "Rohan.*")


class PersonaRecordTests(unittest.TestCase):
    def setUp(self) -> None:
        self.messages = parse_whatsapp_chat(CHAT, chat_id="personal-chat")
        self.chunks = chunk_conversation(self.messages, max_gap_minutes=120)
        self.records = build_persona_records(
            self.messages,
            self.chunks,
            user_id="rohan",
            persona_id="rohan-verma",
            requested_target_author="rohan",
            target_author="Rohan Verma",
        )

    def test_labels_context_and_target_responses_without_changing_wording(self) -> None:
        self.assertEqual(1, len(self.records))
        record = self.records[0]
        self.assertEqual(["Aditi"], [message.author for message in record.context])
        self.assertEqual(
            ["First response, typo and emoji 🙂!", "Second response."],
            [message.text for message in record.target_responses],
        )
        self.assertEqual(2, len(record.target_message_ids))
        self.assertIn("Aditi: Context that should remain separate.", record.text)

    def test_context_only_chunks_are_skipped(self) -> None:
        self.assertEqual(2, len(self.chunks))
        self.assertEqual(1, len(self.records))
        self.assertNotIn(self.messages[-1].message_id, self.records[0].message_ids)

    def test_payload_is_json_serializable_and_complete(self) -> None:
        payload = self.records[0].to_payload()
        json.dumps(payload, ensure_ascii=False)
        self.assertEqual("whatsapp_persona_v2", payload["schema_version"])
        self.assertEqual("whatsapp", payload["source"])
        self.assertEqual("rohan", payload["user_id"])
        self.assertEqual("rohan-verma", payload["persona_id"])
        self.assertEqual("Rohan Verma", payload["target_author"])
        self.assertEqual("rohan", payload["requested_target_author"])
        self.assertIn("target_responses", payload)
        self.assertIn("context", payload)

    def test_point_ids_are_stable_and_user_and_persona_scoped(self) -> None:
        record = self.records[0]
        self.assertEqual(stable_point_id(record), stable_point_id(record))
        self.assertNotEqual(
            stable_point_id(record), stable_point_id(replace(record, user_id="other-user"))
        )
        self.assertNotEqual(
            stable_point_id(record),
            stable_point_id(replace(record, persona_id="other-persona")),
        )

    def test_rejects_blank_user_or_persona_identity(self) -> None:
        with self.assertRaisesRegex(ValueError, "user_id"):
            build_persona_records(
                self.messages,
                self.chunks,
                user_id="  ",
                persona_id="persona",
                requested_target_author="rohan",
                target_author="Rohan Verma",
            )

    def test_legacy_v1_collection_is_never_mutated(self) -> None:
        self.assertEqual("mirror_whatsapp_v2", DEFAULT_COLLECTION)
        store = object.__new__(WhatsAppQdrantStore)
        with self.assertRaisesRegex(ValueError, "legacy collection"):
            store.ensure_collection(LEGACY_COLLECTION)
        with self.assertRaisesRegex(ValueError, "persona_id"):
            build_persona_records(
                self.messages,
                self.chunks,
                user_id="user",
                persona_id="  ",
                requested_target_author="rohan",
                target_author="Rohan Verma",
            )


if __name__ == "__main__":
    unittest.main()
