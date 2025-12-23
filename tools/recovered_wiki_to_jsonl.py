#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
from pathlib import Path
import xml.etree.ElementTree as ET

PAGE_START = b"<page>"
PAGE_END   = b"</page>"

def iter_recovered_bytes(files):
    """Yield bytes from multiple files sequentially."""
    for fp in files:
        with open(fp, "rb") as f:
            while True:
                chunk = f.read(1024 * 1024)  # 1MB
                if not chunk:
                    break
                yield chunk

def find_pages_from_stream(byte_iter, max_buffer=64 * 1024 * 1024):
    """
    Reconstruct complete <page>...</page> blocks across file boundaries.
    Yields raw bytes of each complete page XML fragment.
    """
    buf = bytearray()
    while True:
        try:
            chunk = next(byte_iter)
            buf.extend(chunk)
        except StopIteration:
            break

        # Prevent runaway buffer if something goes wrong
        if len(buf) > max_buffer:
            # Keep only the tail part; try to resync on next <page>
            idx = buf.find(PAGE_START)
            if idx >= 0:
                buf = buf[idx:]
            else:
                buf = buf[-(max_buffer // 2):]

        while True:
            s = buf.find(PAGE_START)
            if s < 0:
                # keep a small tail in case "<pa" is split across chunks
                if len(buf) > 16:
                    buf = buf[-16:]
                break

            e = buf.find(PAGE_END, s)
            if e < 0:
                # wait for more data
                if s > 0:
                    # discard junk before <page>
                    buf = buf[s:]
                break

            e_end = e + len(PAGE_END)
            page_bytes = bytes(buf[s:e_end])
            del buf[:e_end]
            yield page_bytes

def parse_page_xml(page_bytes):
    """
    Parse a single <page>...</page> fragment. Handle namespace if present.
    Returns dict or None.
    """
    try:
        el = ET.fromstring(page_bytes)
    except ET.ParseError:
        return None

    # MediaWiki exports use namespaces at the root <mediawiki>, but <page> fragment itself often has none.
    # Still, be defensive: strip namespaces.
    def strip_ns(tag):
        return tag.split("}", 1)[-1] if "}" in tag else tag

    title = None
    page_id = None
    text = None

    for child in el.iter():
        t = strip_ns(child.tag)
        if t == "title" and title is None:
            title = child.text or ""
        elif t == "id" and page_id is None:
            # first <id> under <page> is page id (revision id comes later)
            try:
                page_id = int(child.text)
            except Exception:
                page_id = None
        elif t == "text" and text is None:
            text = child.text or ""

    if not title or text is None:
        return None

    return {"title": title, "id": page_id, "text": text}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="D:/data/general_pretrain_corpus/wikipedia", help="Directory containing rec*.xml fragments from bzip2recover")
    ap.add_argument("--out", default="D:/data/processed_general_pretrain_corpus/dewiki_recovered.jsonl", help="Output JSONL")
    ap.add_argument("--pattern", default=r"rec\d+.*\.xml$", help="Regex to pick and sort fragments")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of pages (0 = no limit)")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    rx = re.compile(args.pattern)

    files = sorted([p for p in in_dir.iterdir() if p.is_file() and rx.search(p.name)],
                   key=lambda p: p.name)

    if not files:
        raise SystemExit(f"No input files found in {in_dir} matching {args.pattern}")

    count_in = 0
    count_out = 0
    file_part = 0
    current_size = 0
    max_size = 500 * 1024 * 1024  # 500 MB

    def get_output_path(base_path, part):
        """Generate output path with part number."""
        p = Path(base_path)
        if part == 0:
            return base_path
        else:
            return str(p.parent / f"{p.stem}.part_{part:03d}{p.suffix}")

    byte_iter = iter(iter_recovered_bytes(files))
    out_path = get_output_path(args.out, file_part)
    out_f = open(out_path, "w", encoding="utf-8")
    print(f"Writing to: {out_path}")

    exclued_criteria = [
        "#REDIRECT", "<noinclude>", "__NOTOC__", "__NOEDITSECTION__", "__FORCETOC__", "{{Artikel Jahrgang",
        "{{Begriffsklärungshinweis", "{{Geographische Lage", "{{Infobox", "{{Kfz-Kennzeichen", "#WEITERLEITUNG",
        "[[Datei:"
    ]

    try:
        for page_bytes in find_pages_from_stream(byte_iter):
            count_in += 1
            obj = parse_page_xml(page_bytes)
            if obj is None:
                continue

            # Exclude pages based on criteria
            if any(crit.lower() in obj["text"].lower() for crit in exclued_criteria):
                continue

            # only keep Title and Text
            obj = {"title": obj["title"], "text": obj["text"]}

            line = json.dumps(obj, ensure_ascii=False) + "\n"
            out_f.write(line)
            current_size += len(line.encode("utf-8"))
            count_out += 1

            # Check if we need to rotate to a new file
            if current_size >= max_size:
                out_f.close()
                print(f"File size limit reached. Closed: {out_path} ({current_size:,} bytes)")
                file_part += 1
                current_size = 0
                out_path = get_output_path(args.out, file_part)
                out_f = open(out_path, "w", encoding="utf-8")
                print(f"Writing to: {out_path}")

            if count_out % 10000 == 0:
                print(f"pages_written={count_out:,} (pages_seen={count_in:,}), current_file={out_path}")

            if args.limit and count_out >= args.limit:
                break
    finally:
        out_f.close()

    print(f"Done. pages_written={count_out:,}, pages_seen={count_in:,}, final_file={out_path}")

if __name__ == "__main__":
    main()
