from __future__ import annotations

import unittest

from rag_lab.chunking import (
    ChunkingConfig,
    SourceDocument,
    chunk_document,
    fixed_chunks,
    semantic_chunks,
)


class ChunkingTests(unittest.TestCase):
    def test_fixed_chunks_overlap_without_exceeding_limit(self) -> None:
        chunks = fixed_chunks(
            "one two three four five six seven eight",
            max_characters=18,
            overlap_characters=4,
        )

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 18 for chunk in chunks))

    def test_sentence_chunking_preserves_atomic_sentences(self) -> None:
        document = SourceDocument(
            document_id="1",
            title="A title",
            text="First fact. Second fact. Third fact.",
        )
        chunks = chunk_document(
            document,
            ChunkingConfig(
                method="sentence",
                max_characters=30,
                overlap_sentences=1,
            ),
        )

        self.assertGreater(len(chunks), 1)
        self.assertTrue(any("Second fact." in chunk.text for chunk in chunks))

    def test_semantic_chunking_breaks_at_topic_change(self) -> None:
        text = "Cats purr. Kittens meow. Rockets launch. Spacecraft orbit."

        vectors = {
            "Cats purr.": [1.0, 0.0],
            "Kittens meow.": [0.9, 0.1],
            "Rockets launch.": [0.0, 1.0],
            "Spacecraft orbit.": [0.1, 0.9],
        }

        chunks = semantic_chunks(
            text,
            embed_sentences=lambda sentences: [vectors[value] for value in sentences],
            max_characters=1_000,
            break_percentile=75,
        )

        self.assertEqual(
            ["Cats purr. Kittens meow.", "Rockets launch. Spacecraft orbit."],
            chunks,
        )


if __name__ == "__main__":
    unittest.main()

