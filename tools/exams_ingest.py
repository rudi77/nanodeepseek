"""Ingest exam markdown into SFT/DPO datasets with provenance."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from io_utils import append_jsonl
from paraphrase import paraphrase_instruction
from schemas import make_dpo_pair, make_sft_chat

SYSTEM_PROMPT = (
    "Du bist Buchhaltungsassistent. "
    "Antworte im JSON-Format, falls die Frage eine Buchung verlangt; "
    "ansonsten antworte knapp und direkt."
)


def _split_sections(text: str) -> List[Tuple[str, str, int]]:
    sections: List[Tuple[str, str, int]] = []
    current_title = ""
    current_lines: List[str] = []
    start_line = 1
    for idx, line in enumerate(text.splitlines(), start=1):
        if line.startswith("## "):
            if current_lines:
                sections.append((current_title, "\n".join(current_lines).strip(), start_line))
            current_title = line.strip()[3:]
            current_lines = []
            start_line = idx + 1
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_title, "\n".join(current_lines).strip(), start_line))
    return sections


def _extract_pairs(sections: List[Tuple[str, str, int]]):
    tasks = [s for s in sections if "bungsaufgaben" in s[0] or "Aufgabe" in s[0]]
    solutions = [s for s in sections if "Musterl" in s[0] or "L" in s[0] and "sung" in s[0]]
    pairs = []
    for idx, task in enumerate(tasks):
        sol = solutions[idx] if idx < len(solutions) else ("", "", 0)
        pairs.append((task, sol))
    return pairs


def _mutate_solution(text: str) -> Tuple[str, str]:
    if "Soll" in text or "Haben" in text:
        swapped = text.replace("Soll", "__TMP__").replace("Haben", "Soll").replace("__TMP__", "Haben")
        return swapped, "SWAP_SOLL_HABEN"

    numbers = re.findall(r"\d+[\.,]?\d*", text)
    if numbers:
        target = numbers[0]
        if "," in target:
            mutated = str(target).replace(",", ".")
        else:
            try:
                mutated = str(float(target) + 1)
            except Exception:
                mutated = target + "1"
        return text.replace(target, mutated, 1), "NUMERIC_TWEAK"

    return text + " (falsch)", "TEXT_TWEAK"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--out-sft", default="train_sft_exams.jsonl")
    parser.add_argument("--out-dpo", default="train_dpo_exams.jsonl")
    parser.add_argument("--paraphrase", action="store_true")
    args = parser.parse_args()

    Path(args.out_sft).write_text("", encoding="utf-8")
    Path(args.out_dpo).write_text("", encoding="utf-8")

    for file_path in args.input:
        text = Path(file_path).read_text(encoding="utf-8")
        sections = _split_sections(text)
        pairs = _extract_pairs(sections)

        for idx, (task, sol) in enumerate(pairs):
            task_title, task_body, task_line = task
            sol_title, sol_body, sol_line = sol

            question = f"{task_title}\n{task_body}".strip()
            if args.paraphrase:
                question = paraphrase_instruction(question)

            answer = sol_body.strip() or "(keine Loesung im Dokument gefunden)"

            meta = {
                "source": "processed_exam",
                "file": str(file_path),
                "task_title": task_title,
                "task_line": task_line,
                "solution_title": sol_title,
                "solution_line": sol_line,
                "pair_index": idx,
            }

            sft_row = make_sft_chat(SYSTEM_PROMPT, question, json.dumps({"answer": answer}, ensure_ascii=False), meta)
            append_jsonl(args.out_sft, sft_row)

            rejected, error_class = _mutate_solution(answer)
            dpo_row = make_dpo_pair(
                prompt=question,
                chosen=json.dumps({"answer": answer}, ensure_ascii=False),
                rejected=json.dumps({"answer": rejected}, ensure_ascii=False),
                meta={**meta, "error_class": error_class},
            )
            append_jsonl(args.out_dpo, dpo_row)


if __name__ == "__main__":
    main()
