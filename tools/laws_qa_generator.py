"""Generate law Q&A SFT data from markdown."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

from io_utils import append_jsonl
from paraphrase import paraphrase_instruction
from schemas import make_sft_chat

SYSTEM_PROMPT = (
    "Du bist ein sachlicher Assistent. Antworte kurz und ohne Halluzinationen."
)


def _split_sections(text: str) -> List[Tuple[str, str, int]]:
    sections = []
    title = ""
    lines: List[str] = []
    start_line = 1
    for idx, line in enumerate(text.splitlines(), start=1):
        if line.startswith("## "):
            if lines:
                sections.append((title, "\n".join(lines).strip(), start_line))
            title = line[3:].strip()
            lines = []
            start_line = idx + 1
        else:
            lines.append(line)
    if lines:
        sections.append((title, "\n".join(lines).strip(), start_line))
    return sections


def _make_qa(title: str, body: str) -> Tuple[str, str]:
    snippet = " ".join(body.splitlines())[:400].strip()
    question = f"Worum geht es in '{title}'?"
    answer = snippet if snippet else "Kein Inhalt vorhanden."
    return question, answer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--out-sft", default="train_sft_laws.jsonl")
    parser.add_argument("--paraphrase", action="store_true")
    args = parser.parse_args()

    Path(args.out_sft).write_text("", encoding="utf-8")

    for file_path in args.input:
        text = Path(file_path).read_text(encoding="utf-8")
        sections = _split_sections(text)
        for idx, (title, body, line) in enumerate(sections):
            if not title:
                continue
            question, answer = _make_qa(title, body)
            if args.paraphrase:
                question = paraphrase_instruction(question)

            meta = {
                "source": "processed_law",
                "file": str(file_path),
                "section_title": title,
                "section_line": line,
                "section_index": idx,
            }
            sft_row = make_sft_chat(
                SYSTEM_PROMPT,
                question,
                json.dumps({"answer": answer}, ensure_ascii=False),
                meta,
            )
            append_jsonl(args.out_sft, sft_row)


if __name__ == "__main__":
    main()
