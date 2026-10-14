"""Local model probes; run with WHATSAPP_SEMANTIC_INTEGRATION=1."""

from __future__ import annotations

import os
import unittest

from whatsapp.semantic_topics import semantic_topic_boundaries


@unittest.skipUnless(os.getenv("WHATSAPP_SEMANTIC_INTEGRATION") == "1", "requires local multilingual embedding models")
class SemanticTopicTests(unittest.TestCase):
    def test_hinglish_topic_changes_and_continuity(self):
        messages = [
            "Mummy ke birthday pe kya plan hai?",
            "Cake aur dinner ka soch rahe hain",
            "Party ke liye hall book karna hai",
            "Haan, main decoration arrange kar dunga",
            "Kal office meeting hai",
            "Presentation ready karni hai",
            "Boss ko slides bhej dena",
            "Mera ghutna dard kar raha hai",
            "Ab pain thoda better hai",
            "Doctor ne rest bola tha",
        ]
        rows = [{"normalized_text": text} for text in messages]
        self.assertEqual({4, 7}, semantic_topic_boundaries(rows))

    def test_code_switched_health_followups_do_not_split(self):
        messages = [
            "Mera ghutna dard kar raha hai",
            "Doctor ke paas jaana padega",
            "Dawai se thoda relief mila",
            "Ab pain better hai",
            "Kal physiotherapy karwayi",
            "Walking karne par dard hai",
            "Rest karunga kuch din",
            "Hope jaldi recover ho jaaye",
        ]
        self.assertEqual(set(), semantic_topic_boundaries(
            [{"normalized_text": text} for text in messages]
        ))


if __name__ == "__main__":
    unittest.main()
