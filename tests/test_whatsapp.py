from __future__ import annotations

import unittest

from whatsapp import chunk_conversation, parse_whatsapp_chat


SAMPLE_CHAT = """[08/08/25, 11:34:28\u202fAM] Rohan Verma: :( i hope it gets better
[08/08/25, 11:35:05\u202fAM] Rohan Verma: You go work , call me in the night if youre not too tired
[08/08/25, 12:09:31\u202fPM] Aditi: I tried to text last night. \u200e<This message was edited>
[08/08/25, 12:09:38\u202fPM] Aditi: Your net was off
[08/08/25, 12:19:30\u202fPM] Rohan Verma: Thats fine, ping me tonight i’ll call you.
I think we need to have one adult-adult talk before next week.
[08/08/25, 12:35:41\u202fPM] Aditi: Ok
[08/08/25, 11:57:02\u202fPM] Aditi: Hey
[08/08/25, 11:57:34\u202fPM] Aditi: Got home a few mins ago
[08/08/25, 11:57:38\u202fPM] Aditi: What's up
[09/08/25, 12:07:45\u202fAM] Rohan Verma: Hey"""


class WhatsAppParserTests(unittest.TestCase):
    def test_parses_ios_export_and_preserves_multiline_messages(self) -> None:
        messages = parse_whatsapp_chat(
            SAMPLE_CHAT,
            chat_id="chat-1",
            timezone_name="Asia/Kolkata",
        )

        self.assertEqual(10, len(messages))
        self.assertEqual("Rohan Verma", messages[0].author)
        self.assertEqual(11, messages[0].timestamp.hour)
        self.assertEqual(34, messages[0].timestamp.minute)
        self.assertEqual("+05:30", messages[0].timestamp.strftime("%z")[:3] + ":30")
        self.assertEqual("I tried to text last night.", messages[2].text)
        self.assertTrue(messages[2].is_edited)
        self.assertEqual(
            "Thats fine, ping me tonight i’ll call you.\n"
            "I think we need to have one adult-adult talk before next week.",
            messages[4].text,
        )
        self.assertEqual(0, messages[9].timestamp.hour)

    def test_supports_android_export_and_system_messages(self) -> None:
        chat = """\u200e09/08/2025, 00:07 - Rohan: Hey
\u200e09/08/2025, 00:08 - Messages and calls are end-to-end encrypted.
\u200e09/08/2025, 00:09 - Aditi: café 👋🏽"""

        messages = parse_whatsapp_chat(chat, chat_id="android")

        self.assertEqual(3, len(messages))
        self.assertEqual("Hey", messages[0].text)
        self.assertTrue(messages[1].is_system)
        self.assertEqual("café 👋🏽", messages[2].text)

    def test_marks_deleted_and_media_messages_as_unusable(self) -> None:
        chat = """[09/08/25, 1:00 PM] A: <Media omitted>
[09/08/25, 1:01 PM] B: This message was deleted
[09/08/25, 1:02 PM] A: usable"""

        messages = parse_whatsapp_chat(chat, chat_id="notices")

        self.assertTrue(messages[0].has_media)
        self.assertTrue(messages[1].is_deleted)
        self.assertEqual("", messages[0].text)
        self.assertEqual("", messages[1].text)
        self.assertEqual(
            (messages[2].message_id,),
            chunk_conversation(messages)[0].message_ids,
        )

    def test_chunks_on_message_boundaries_with_overlap(self) -> None:
        messages = parse_whatsapp_chat(
            SAMPLE_CHAT,
            chat_id="chat-1",
            timezone_name="Asia/Kolkata",
        )

        chunks = chunk_conversation(
            messages,
            max_messages=4,
            max_characters=10_000,
            overlap_messages=1,
        )

        self.assertEqual(3, len(chunks))
        self.assertEqual(messages[3].message_id, chunks[1].message_ids[0])
        self.assertIn("Rohan Verma: :( i hope it gets better", chunks[0].text)
        self.assertNotIn("[08/08/25", chunks[0].text)

    def test_does_not_join_separate_chat_sessions(self) -> None:
        chat = """[09/08/25, 1:00 PM] A: first session
[09/08/25, 5:00 PM] B: second session"""
        messages = parse_whatsapp_chat(chat, chat_id="sessions")

        chunks = chunk_conversation(messages, max_gap_minutes=120)

        self.assertEqual(2, len(chunks))
        self.assertEqual("A: first session", chunks[0].text)
        self.assertEqual("B: second session", chunks[1].text)

    def test_rejects_content_before_first_timestamp(self) -> None:
        with self.assertRaisesRegex(ValueError, "before the first message"):
            parse_whatsapp_chat("not an export", chat_id="invalid")


if __name__ == "__main__":
    unittest.main()
