"""Run local Qwen boundary probes on invented chats only (no databases)."""

from __future__ import annotations

import json

from whatsapp.topic_memory import LocalQwen, detect_boundaries

from run_hinglish_topic_demo import read_messages


def rows(lines: list[str]) -> list[dict[str, str]]:
    return [{"author": "Aditi" if i % 2 else "Rohan", "normalized_text": line}
            for i, line in enumerate(lines)]


def cases() -> list[tuple[str, list[dict[str, str]], list[int]]]:
    original = [dict(author=body.split(": ", 1)[0],
                     normalized_text=body.split(": ", 1)[1])
                for _, body in read_messages()]
    commute = [
        "Aaj metro bahut late thi", "Haan station pe bheed thi", "Platform 2 pe khade the?",
        "Haan wahi", "Train 15 min late aayi", "Kal bhi aisa hua tha", "Blue line ka issue hai kya?",
        "Shayad signal problem", "Office pahunchne mein time laga", "Mera bhi same scene tha",
        "Subah wali train better hai", "Haan 8 baje wali", "Usmein seats milti hain?",
        "Kabhi kabhi", "Aaj nahi mili", "Tum kis station se chadhe?",
        "Rajiv Chowk se", "Wahan to bheed rehti hai", "Yellow line change karni padti hai",
        "Haan interchange tough hai", "Kal thoda jaldi niklenge", "Main bhi",
        "Traffic se better metro hi hai", "Bilkul", "Return mein kab nikloge?",
        "Sham 6 baje", "Peak hour hoga", "Haan but manage ho jayega",
        "Ticket app pe recharge kar lena", "Kar liya metro card recharge",
        "Waise mummy ka birthday Sunday ko hai", "Oh cake order kiya?",
        "Chocolate cake lene ka soch raha hoon", "Mummy ko chocolate pasand hai",
        "Gift bhi lena hai", "Haan uske liye market chalte hain",
    ]
    vague_return = [
        "Hey", "Hi", "Kya haal", "Theek tum batao",
        "Mera knee pain kal se zyada hai", "Arre doctor ko dikhaya?",
        "Haan wohi purani injury", "Abhi bhi?", "Haan wahi", "Ice lagayi thi",
        "Aaj dinner kis restaurant mein karein?", "Jo naya Italian wala hai",
        "Uska pasta acha hai", "Booking karni padegi", "Haan 8 baje",
        "Waise knee pain ab thoda better hai", "Good, ice se farak pada?",
        "Haan, kal follow up bhi hai", "Doctor kya bola tha?",
    ]
    return [("original", original, [4, 13, 21, 34]),
            ("real_block_edge", rows(commute), [30]),
            ("vague_and_resume", rows(vague_return), [4, 10, 15])]


def main() -> None:
    model = LocalQwen()
    results = []
    for name, chat, expected in cases():
        found = detect_boundaries(chat, model)
        indices = [item["index"] for item in found]
        results.append({"case": name, "expected": expected, "found": found,
                        "exact": indices == expected})
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
