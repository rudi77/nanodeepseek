#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert Wikipedia/Wikimedia Wikitext (in JSONL) to plaintext JSONL.

Input JSONL example:
  {"title": "...", "text": "'''Alan Smithee''' steht als [[Pseudonym]] ... <ref>...</ref> ..."}
Output JSONL:
  {"title":"...", "text":"Alan Smithee steht als Pseudonym für ..."}

Usage:
  python wikitext_to_plaintext.py --in dewiki.jsonl --out dewiki_plain.jsonl
  python wikitext_to_plaintext.py --in dewiki.jsonl --out dewiki_plain.jsonl --drop_title
  python wikitext_to_plaintext.py --in dewiki.jsonl --out dewiki_plain.jsonl --min_chars 500
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path
import glob
from typing import Optional, Dict, Any
from multiprocessing import get_context, cpu_count
from functools import partial

# Try best parser
try:
    import mwparserfromhell
    HAS_MWPH = True
except Exception:
    HAS_MWPH = False


# -----------------------------
# Fallback regex cleaners
# -----------------------------

RE_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
RE_REF_TAG = re.compile(r"<ref\b[^>/]*?>.*?</ref\s*>", re.IGNORECASE | re.DOTALL)
RE_REF_SELF = re.compile(r"<ref\b[^>]*/\s*>", re.IGNORECASE)
RE_REFERENCES = re.compile(r"<references\s*/\s*>", re.IGNORECASE)
RE_TAGS = re.compile(r"</?[^>]+?>", re.DOTALL)  # generic HTML tags
RE_CATEGORY = re.compile(r"\[\[\s*Kategorie\s*:[^\]]+\]\]", re.IGNORECASE)
RE_FILE = re.compile(r"\[\[\s*(Datei|File|Bild|Image)\s*:[^\]]+\]\]", re.IGNORECASE)
RE_EXTLINK = re.compile(r"\[(https?://[^\s\]]+)\s+([^\]]+)\]")  # [url text]
RE_EXTLINK_BARE = re.compile(r"\[(https?://[^\s\]]+)\]")        # [url]
RE_WIKILINK = re.compile(r"\[\[([^\]|]+)\|([^\]]+)\]\]")        # [[A|B]]
RE_WIKILINK2 = re.compile(r"\[\[([^\]]+)\]\]")                  # [[A]]
RE_TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")                      # non-nested templates
RE_QUOTES = re.compile(r"'{2,5}")                                # '' ''' '''''

RE_WS = re.compile(r"[ \t]+\n")
RE_MULTI_NL = re.compile(r"\n{3,}")
RE_MULTI_SPACE = re.compile(r"[ \t]{2,}")


def fallback_clean_wikitext(text: str) -> str:
    """A conservative regex-based cleanup for wikitext (fallback)."""
    if not text:
        return ""

    s = text

    # remove comments
    s = RE_COMMENT.sub(" ", s)

    # remove refs and references
    s = RE_REF_TAG.sub(" ", s)
    s = RE_REF_SELF.sub(" ", s)
    s = RE_REFERENCES.sub(" ", s)

    # drop categories and file links
    s = RE_CATEGORY.sub(" ", s)
    s = RE_FILE.sub(" ", s)

    # external links: keep label
    s = RE_EXTLINK.sub(r"\2", s)
    s = RE_EXTLINK_BARE.sub(" ", s)

    # internal links: keep display text
    s = RE_WIKILINK.sub(r"\2", s)
    s = RE_WIKILINK2.sub(r"\1", s)

    # remove bold/italic quotes
    s = RE_QUOTES.sub("", s)

    # templates: repeatedly remove simple templates (helps with many small ones)
    for _ in range(6):
        s2 = RE_TEMPLATE.sub(" ", s)
        if s2 == s:
            break
        s = s2

    # remove remaining HTML tags
    s = RE_TAGS.sub(" ", s)

    # normalize whitespace
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = RE_WS.sub("\n", s)
    s = RE_MULTI_SPACE.sub(" ", s)
    s = RE_MULTI_NL.sub("\n\n", s)
    return s.strip()


# -----------------------------
# mwparserfromhell cleaner (recommended)
# -----------------------------

def mwph_clean_wikitext(text: str) -> str:
    """Use mwparserfromhell to parse and strip markup more accurately."""
    if not text:
        return ""

    # Aggressive pre-processing with regex to reduce parser work
    s = RE_COMMENT.sub(" ", text)
    s = RE_REF_TAG.sub(" ", s)
    s = RE_REF_SELF.sub(" ", s)
    s = RE_REFERENCES.sub(" ", s)
    s = RE_CATEGORY.sub(" ", s)
    s = RE_FILE.sub(" ", s)
    
    # Remove nested templates iteratively (much faster than parser removal)
    for _ in range(10):  # max 10 nesting levels
        s_new = RE_TEMPLATE.sub(" ", s)
        if s_new == s:
            break
        s = s_new

    # Parse remaining wikitext
    try:
        code = mwparserfromhell.parse(s)
        # strip_code() automatically handles templates, tags, links
        # This is MUCH faster than iterating and removing
        out = code.strip_code(normalize=True, collapse=True, keep_template_params=False)
    except Exception:
        # Fallback to regex if parsing fails
        out = fallback_clean_wikitext(s)

    # Normalize whitespace
    out = out.replace("\r\n", "\n").replace("\r", "\n")
    out = RE_MULTI_SPACE.sub(" ", out)
    out = RE_MULTI_NL.sub("\n\n", out)
    return out.strip()


# -----------------------------
# General normalization
# -----------------------------

def normalize_text(s: str) -> str:
    if not s:
        return ""
    # Remove surrogates and invalid Unicode characters
    # Encode with 'surrogatepass' then decode with 'ignore' to remove them
    try:
        s = s.encode('utf-8', errors='surrogatepass').decode('utf-8', errors='ignore')
    except Exception:
        # Fallback: remove surrogates manually
        s = ''.join(ch for ch in s if not (0xD800 <= ord(ch) <= 0xDFFF))
    
    s = unicodedata.normalize("NFC", s)
    # remove control chars except newline/tab
    s = "".join(ch for ch in s if (ch == "\n" or ch == "\t" or ord(ch) >= 32))
    return s.strip()


def load_json_line(line: str) -> Optional[Dict[str, Any]]:
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except Exception:
        return None


def process_entry(line: str, use_mwph: bool, min_chars: int, drop_title: bool) -> Optional[str]:
    """Process a single JSONL entry. Returns JSON string or None."""
    try:
        obj = load_json_line(line)
        if not obj:
            return None

        title = obj.get("title", "")
        text = obj.get("text", "")
        if not isinstance(text, str):
            return None

        if use_mwph:
            plain = mwph_clean_wikitext(text)
        else:
            plain = fallback_clean_wikitext(text)

        plain = normalize_text(plain)

        if min_chars and len(plain) < min_chars:
            return None

        # Normalize title as well to avoid surrogate issues
        title = normalize_text(title) if title else ""

        if drop_title:
            out_obj = {"text": plain}
        else:
            out_obj = {"title": title, "text": plain}

        # Use ensure_ascii=False but this should be safe now after normalize_text
        return json.dumps(out_obj, ensure_ascii=False)
    except Exception as e:
        # Log error but don't crash - return None to skip this entry
        # Uncomment for debugging: print(f"[ERROR] Failed to process entry: {e}", file=sys.stderr)
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="D:/data/processed_general_pretrain_corpus", help="Input directory which contains JSONL files (title+text wikitext)")
    ap.add_argument("--out", dest="out", default="D:/data/processed_general_pretrain_corpus_normalized", help="Output directory JSONL (plaintext)")
    ap.add_argument("--drop_title", action="store_true", help="Output only {'text': ...} without title")
    ap.add_argument("--min_chars", type=int, default=0, help="Drop entries with plaintext shorter than this")
    ap.add_argument("--force_fallback", action="store_true", help="Use regex fallback even if mwparserfromhell is available")
    ap.add_argument("--progress_every", type=int, default=50000)
    ap.add_argument("--workers", type=int, default=max(1, cpu_count() - 1), help="Number of parallel workers")
    ap.add_argument("--batch_size", type=int, default=1000, help="Batch size for parallel processing")
    ap.add_argument("--no-parallel", action="store_true", help="Disable multiprocessing (useful for debugging)")
    args = ap.parse_args()

    in_path = Path(args.inp)
    out_path = Path(args.out)

    if not in_path.exists():
        print(f"Input not found: {in_path}", file=sys.stderr)
        return 2

    use_mwph = HAS_MWPH and (not args.force_fallback)
    if not use_mwph:
        print("[INFO] Using fallback regex cleaner (install mwparserfromhell for better quality).", file=sys.stderr)
    else:
        print("[INFO] Using mwparserfromhell cleaner.", file=sys.stderr)

    n_in = 0
    n_out = 0
    n_dropped = 0
    n_errors = 0

    # read input files with glob
    input_files = glob.glob(str(in_path / "*.jsonl"))
    if not input_files:
        print(f"No input JSONL files found in {in_path}", file=sys.stderr)
        return 3
    
    # Setup processing (parallel or sequential)
    use_parallel = not args.no_parallel and args.workers > 1
    
    process_func = partial(
        process_entry,
        use_mwph=use_mwph,
        min_chars=args.min_chars,
        drop_title=args.drop_title
    )
    
    for input_file in input_files:
        in_path = Path(input_file)
        out_path = Path(args.out) / in_path.name
        
        if use_parallel:
            print(f"[INFO] Processing {in_path.name} with {args.workers} workers...", file=sys.stderr)
        else:
            print(f"[INFO] Processing {in_path.name} sequentially...", file=sys.stderr)

        # if out_path file exist already, skip
        if out_path.exists():
            print(f"[INFO] Output file {out_path} already exists, skipping.", file=sys.stderr)
            continue    

        with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
            if use_parallel:
                # Parallel processing with spawn context (Windows-safe)
                ctx = get_context('spawn')
                batch = []
                
                with ctx.Pool(args.workers) as pool:
                    for line in fin:
                        n_in += 1
                        batch.append(line)
                        
                        # Process batch when full
                        if len(batch) >= args.batch_size:
                            results = pool.map(process_func, batch)
                            
                            for result in results:
                                if result:
                                    try:
                                        fout.write(result + "\n")
                                        n_out += 1
                                    except UnicodeEncodeError:
                                        n_errors += 1
                                else:
                                    n_dropped += 1
                            
                            batch = []
                            
                            if args.progress_every and (n_out % args.progress_every == 0):
                                print(f"[PROGRESS] in={n_in:,} out={n_out:,} dropped={n_dropped:,}", file=sys.stderr)
                    
                    # Process remaining batch
                    if batch:
                        results = pool.map(process_func, batch)
                        
                        for result in results:
                            if result:
                                try:
                                    fout.write(result + "\n")
                                    n_out += 1
                                except UnicodeEncodeError:
                                    n_errors += 1
                            else:
                                n_dropped += 1
            else:
                # Sequential processing (for debugging)
                for line in fin:
                    n_in += 1
                    result = process_func(line)
                    
                    if result:
                        try:
                            fout.write(result + "\n")
                            n_out += 1
                        except UnicodeEncodeError:
                            n_errors += 1
                    else:
                        n_dropped += 1
                    
                    if args.progress_every and (n_out % args.progress_every == 0):
                        print(f"[PROGRESS] in={n_in:,} out={n_out:,} dropped={n_dropped:,}", file=sys.stderr)

        print(f"Done. in={n_in:,} out={n_out:,} dropped={n_dropped:,} errors={n_errors:,} -> {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    # Required for Windows multiprocessing
    raise SystemExit(main())
