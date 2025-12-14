#!/usr/bin/env python3

# Description:
# This script processes source code files from a specified input directory,
# applies various transformations, and writes the results to an output directory.

import argparse, os, json, hashlib, random
from pathlib import Path

CPP_EXT = {".c",".cc",".cpp",".cxx",".h",".hh",".hpp",".hxx"}
CSHARTP_EXT = {".cs"}
PY_EXT  = {".py"}

SKIP_DIRS = {".git","node_modules","build","dist",".venv",".idea",".vscode","__pycache__"}

def detect_lang(path: Path):
    ext = path.suffix.lower()
    if ext in CPP_EXT: return "cpp"
    if ext in PY_EXT:  return "py"
    if ext in CSHARTP_EXT: return "csharp"
    return None

def read_text(path: Path, max_bytes: int):
    try:
        if path.stat().st_size > max_bytes:
            return None
        # tolerate weird encodings
        b = path.read_bytes()
        if len(b) > max_bytes:
            return None
        txt = b.decode("utf-8", errors="replace")
        # normalize newlines
        txt = txt.replace("\r\n", "\n").replace("\r", "\n")
        return txt
    except Exception:
        return None

def is_minified(text: str, max_line_len: int):
    for line in text.split("\n"):
        if len(line) > max_line_len:
            return True
    return False

def chunk_lines(lines, target_chars: int, max_lines: int, overlap_lines: int):
    # Greedy chunking by lines, prefer boundaries on blank lines or lines ending with "}"
    chunks = []
    i = 0
    n = len(lines)
    while i < n:
        j = i
        chars = 0
        last_good = None
        while j < n and (j - i) < max_lines and chars < target_chars:
            line = lines[j]
            chars += len(line) + 1
            if line.strip() == "" or line.rstrip().endswith("}"):
                last_good = j
            j += 1

        if last_good is not None and last_good > i:
            end = last_good + 1
        else:
            end = j

        chunk = "\n".join(lines[i:end]).strip("\n")
        if chunk:
            chunks.append(chunk)

        if end >= n:
            break
        i = max(end - overlap_lines, end) if overlap_lines > 0 else end
    return chunks

def stable_split(hash_hex: str, val_pct: int):
    # deterministic: last byte of sha1
    b = int(hash_hex[-2:], 16)
    return (b % 100) < val_pct

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", required=True)
    ap.add_argument("--out", dest="out_dir", required=True)
    ap.add_argument("--langs", default="cpp,py")
    ap.add_argument("--max-lines", type=int, default=300)
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--max-bytes", type=int, default=200_000)
    ap.add_argument("--target-chars", type=int, default=1600)
    ap.add_argument("--overlap-lines", type=int, default=20)
    ap.add_argument("--max-line-len", type=int, default=2000)
    ap.add_argument("--val-pct", type=int, default=2)
    args = ap.parse_args()

    langs = set(x.strip() for x in args.langs.split(",") if x.strip())
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train.jsonl"
    val_path   = out_dir / "val.jsonl"
    stats_path = out_dir / "stats.json"

    seen = set()
    stats = {
        "files_seen": 0,
        "files_used": 0,
        "samples_total": 0,
        "samples_deduped": 0,
        "samples_train": 0,
        "samples_val": 0,
        "by_lang": {},
        "avg_chars": 0.0,
    }
    total_chars = 0

    def should_skip_dir(p: Path):
        return any(part in SKIP_DIRS for part in p.parts)

    with train_path.open("w", encoding="utf-8") as ftrain, val_path.open("w", encoding="utf-8") as fval:
        for path in in_dir.rglob("*"):
            if path.is_dir():
                continue
            if should_skip_dir(path):
                continue

            lang = detect_lang(path)
            if lang is None or lang not in langs:
                continue

            stats["files_seen"] += 1
            txt = read_text(path, args.max_bytes)
            if not txt:
                continue

            lines = txt.split("\n")
            if len(lines) > args.max_lines:
                # we'll still chunk, but skip absurdly huge files for now
                # (optional: could keep only first N lines)
                pass

            if is_minified(txt, args.max_line_len):
                continue

            chunks = chunk_lines(lines, args.target_chars, args.max_lines, args.overlap_lines)
            if not chunks:
                continue

            stats["files_used"] += 1
            stats["by_lang"].setdefault(lang, 0)

            for ch in chunks:
                ch = ch.strip()
                if len(ch) < args.min_chars:
                    continue

                h = hashlib.sha1(ch.encode("utf-8")).hexdigest()
                if h in seen:
                    stats["samples_deduped"] += 1
                    continue
                seen.add(h)

                rec = {"text": ch, "meta": {"path": str(path), "lang": lang}}
                line = json.dumps(rec, ensure_ascii=False)

                if stable_split(h, args.val_pct):
                    fval.write(line + "\n")
                    stats["samples_val"] += 1
                else:
                    ftrain.write(line + "\n")
                    stats["samples_train"] += 1

                stats["samples_total"] += 1
                stats["by_lang"][lang] += 1
                total_chars += len(ch)

    stats["avg_chars"] = (total_chars / stats["samples_total"]) if stats["samples_total"] else 0.0

    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Wrote:", train_path, val_path, stats_path)
    print(json.dumps(stats, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
