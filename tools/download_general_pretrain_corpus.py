#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
General pretraining corpus downloader (DE + optional EN) for 30–50B tokens scale.

Downloads:
  - Wikipedia dumps (dewiki + optional enwiki) from dumps.wikimedia.org
  - OSCAR (Common Crawl, language-filtered) via Hugging Face datasets (streaming)
Optionally:
  - CulturaX via Hugging Face datasets (streaming) (HUGE)

Writes:
  - data/wikipedia/*.xml.bz2 (raw dumps)
  - data/oscar/*.jsonl.zst shards (streamed text lines)
  - MANIFEST.tsv (urls, files, sha256)

Requires:
  pip install requests datasets zstandard tqdm
Optional for exact token counting:
  pip install transformers tokenizers

Usage examples:
  # 50B tokens target, DE mostly + some EN, exact token count with your tokenizer:
  python download_general_pretrain_corpus.py --out base_corpus --target_tokens 50000000000 \
    --oscar_de_weight 0.85 --oscar_en_weight 0.15 --tokenizer ./tokenizer.json

  # Only download wiki dumps + write OSCAR shards using heuristic counting:
  python download_general_pretrain_corpus.py --out base_corpus --target_tokens 30000000000

Notes:
- Streaming OSCAR means you do NOT download the full dataset; you sample until token budget is reached.
- Ensure you comply with the dataset terms / TDM requirements for your use-case.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Iterable

import requests
from tqdm import tqdm
import zstandard as zstd

# HuggingFace datasets (streaming)
from datasets import load_dataset


# -----------------------------
# Helpers
# -----------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def safe_filename(s: str) -> str:
    s = s.strip().replace("\n", " ")
    s = re.sub(r"[^\w\-.() ]+", "_", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def http_get(session: requests.Session, url: str, *, stream: bool, timeout: int = 180) -> requests.Response:
    headers = {
        "User-Agent": "general-pretrain-downloader/1.0 (contact: rudi.dittrich77@gmail.com)",
        "Accept": "*/*",
    }
    r = session.get(url, headers=headers, stream=stream, timeout=timeout, allow_redirects=True)
    r.raise_for_status()
    return r

def download_file(session: requests.Session, url: str, out_path: Path, *, overwrite: bool = False) -> str:
    if out_path.exists() and not overwrite:
        return sha256_file(out_path)
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    r = http_get(session, url, stream=True)
    total = int(r.headers.get("Content-Length", "0") or "0")
    with tmp.open("wb") as f, tqdm(total=total if total > 0 else None, unit="B", unit_scale=True, desc=out_path.name) as pbar:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
    tmp.replace(out_path)
    return sha256_file(out_path)


# -----------------------------
# Token counting
# -----------------------------
class TokenCounter:
    def __init__(self, tokenizer_ref: Optional[str], heuristic_chars_per_token: float):
        self.heuristic = tokenizer_ref is None
        self.cpt = heuristic_chars_per_token

        self.tokenizer = None
        if tokenizer_ref:
            try:
                # Works for local tokenizer.json or HF name/path
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_ref, use_fast=True)
            except Exception:
                # Fallback: tokenizers.Tokenizer for a local tokenizer.json
                from tokenizers import Tokenizer
                self.tokenizer = Tokenizer.from_file(tokenizer_ref)

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self.heuristic:
            # crude but stable planning heuristic
            return max(1, int(len(text) / self.cpt))
        # HF fast tokenizer
        if hasattr(self.tokenizer, "encode"):
            # transformers tokenizer
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            return len(ids)
        # tokenizers.Tokenizer
        enc = self.tokenizer.encode(text)
        return len(enc.ids)


# -----------------------------
# JSONL.ZST shard writer
# -----------------------------
class JsonlZstShardWriter:
    def __init__(self, out_dir: Path, prefix: str, shard_bytes: int = 512 * 1024 * 1024, level: int = 8):
        self.out_dir = out_dir
        self.prefix = prefix
        self.shard_bytes = shard_bytes
        self.level = level
        self.idx = 0
        self._open_new()

    def _open_new(self):
        if hasattr(self, "fh"):
            self.fh.close()
        self.idx += 1
        self.path = self.out_dir / f"{self.prefix}_shard_{self.idx:05d}.jsonl.zst"
        self.fh = self.path.open("wb")
        self.cctx = zstd.ZstdCompressor(level=self.level).stream_writer(self.fh)
        self.bytes_written = 0

    def write_line(self, line: str):
        b = (line.rstrip("\n") + "\n").encode("utf-8", errors="ignore")
        self.cctx.write(b)
        self.bytes_written += len(b)
        if self.bytes_written >= self.shard_bytes:
            self.close()
            self._open_new()

    def close(self):
        try:
            self.cctx.flush(zstd.FLUSH_FRAME)
            self.cctx.close()
        finally:
            self.fh.close()


# -----------------------------
# Main logic
# -----------------------------
WIKI_URLS = {
    "dewiki": "https://dumps.wikimedia.org/dewiki/latest/dewiki-latest-pages-articles.xml.bz2",
    "enwiki": "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2",
}

def stream_oscar(
    lang_config: str,
    *,
    target_tokens: int,
    counter: TokenCounter,
    out_dir: Path,
    prefix: str,
    max_docs: Optional[int],
    seed: int,
    shard_bytes: int,
) -> Tuple[int, int]:
    """
    Streams OSCAR config from HF and writes jsonl.zst lines:
      {"text": "...", "source": "..."}
    Returns (tokens_written, docs_written).
    """
    random.seed(seed)
    ds = load_dataset("oscar-corpus/oscar", lang_config, split="train", streaming=True, use_auth_token=True, trust_remote_code=True)
    writer = JsonlZstShardWriter(out_dir, prefix=prefix, shard_bytes=shard_bytes)
    tokens = 0
    docs = 0

    # OSCAR items usually have "text". Some configs may differ; handle robustly.
    for item in ds:
        text = item.get("text") or ""
        if not text:
            continue

        t = counter.count(text)
        if t <= 0:
            continue

        # stop conditions
        if tokens + t > target_tokens:
            break
        if max_docs is not None and docs >= max_docs:
            break

        writer.write_line(f'{{"text": {json_escape(text)}, "source": "{prefix}:{lang_config}"}}')
        tokens += t
        docs += 1

        if docs % 2000 == 0:
            print(f"[{prefix}] docs={docs:,} tokens≈{tokens:,}")

    writer.close()
    return tokens, docs

def json_escape(s: str) -> str:
    # minimal JSON string escaping
    s = s.replace("\\", "\\\\").replace('"', '\\"')
    s = s.replace("\r", "\\r").replace("\t", "\\t")
    s = s.replace("\n", "\\n")
    return f'"{s}"'


# "D:\data"

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="D:/data/general_pretrain_corpus", help="Output directory")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    ap.add_argument("--target_tokens", type=int, default=10_000_000_000, help="Target tokens for OSCAR sampling (e.g. 30000000000)")
    ap.add_argument("--include_en", action="store_true", help="Include English OSCAR sampling and enwiki dump")
    ap.add_argument("--oscar_de_weight", type=float, default=0.9, help="Share of OSCAR token budget for DE (rest for EN if enabled)")
    ap.add_argument("--oscar_en_weight", type=float, default=0.1, help="Share of OSCAR token budget for EN (only used if --include_en)")
    ap.add_argument("--tokenizer", type=str, default=None, help="HF tokenizer name/path or local tokenizer.json for exact token counting")
    ap.add_argument("--heuristic_chars_per_token", type=float, default=4.0, help="Heuristic chars/token if no tokenizer provided")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--shard_bytes", type=int, default=512 * 1024 * 1024, help="Approx uncompressed bytes per shard before rotating")
    ap.add_argument("--max_docs", type=int, default=None, help="Optional cap on number of streamed documents")
    ap.add_argument("--skip_wiki", action="store_true", help="Skip Wikipedia dumps")
    args = ap.parse_args()

    base = Path(args.out).resolve()
    ensure_dir(base)
    wiki_dir = base / "wikipedia"
    oscar_dir = base / "oscar"
    ensure_dir(wiki_dir)
    ensure_dir(oscar_dir)

    session = requests.Session()

    manifest_path = base / "MANIFEST.tsv"
    manifest = ["name\turl\tlocal_file\tsha256\tstatus\tnotes\n"]

    # 1) Wikipedia dumps
    if not args.skip_wiki:
        print("\n=== Download Wikipedia dumps ===")
        to_get = ["dewiki"] + (["enwiki"] if args.include_en else [])        
        for key in to_get:
            url = WIKI_URLS[key]
            out_path = wiki_dir / f"{key}-latest-pages-articles.xml.bz2"
            try:
                digest = download_file(session, url, out_path, overwrite=args.overwrite)
                manifest.append(f"{key}_dump\t{url}\t{out_path}\t{digest}\tOK\twikitext xml.bz2\n")
            except Exception as e:
                manifest.append(f"{key}_dump\t{url}\t{out_path}\t\tERR\t{safe_filename(str(e))}\n")
                print(f"[ERR] wiki {key}: {e}", file=sys.stderr)

    # 2) OSCAR sampling (streaming)
    print("\n=== Stream OSCAR and write jsonl.zst shards ===")
    counter = TokenCounter(args.tokenizer, args.heuristic_chars_per_token)

    # Choose OSCAR configs (common ones are "unshuffled_deduplicated_<lang>")
    de_config = "unshuffled_deduplicated_de"
    en_config = "unshuffled_deduplicated_en"

    if args.include_en:
        de_budget = int(args.target_tokens * args.oscar_de_weight)
        en_budget = int(args.target_tokens * args.oscar_en_weight)
    else:
        de_budget = args.target_tokens
        en_budget = 0

    total_tokens = 0

    try:
        t_de, d_de = stream_oscar(
            de_config,
            target_tokens=de_budget,
            counter=counter,
            out_dir=oscar_dir,
            prefix="oscar_de",
            max_docs=args.max_docs,
            seed=args.seed,
            shard_bytes=args.shard_bytes,
        )
        total_tokens += t_de
        manifest.append(f"oscar_de_stream\thttps://huggingface.co/datasets/oscar-corpus/oscar\t{oscar_dir}\t\tOK\tconfig={de_config}; tokens≈{t_de}\n")
        print(f"[OK] OSCAR DE: docs={d_de:,} tokens≈{t_de:,}")
    except Exception as e:
        manifest.append(f"oscar_de_stream\thttps://huggingface.co/datasets/oscar-corpus/oscar\t{oscar_dir}\t\tERR\t{safe_filename(str(e))}\n")
        print(f"[ERR] OSCAR DE: {e}", file=sys.stderr)

    if args.include_en and en_budget > 0:
        try:
            t_en, d_en = stream_oscar(
                en_config,
                target_tokens=en_budget,
                counter=counter,
                out_dir=oscar_dir,
                prefix="oscar_en",
                max_docs=args.max_docs,
                seed=args.seed + 1,
                shard_bytes=args.shard_bytes,
            )
            total_tokens += t_en
            manifest.append(f"oscar_en_stream\thttps://huggingface.co/datasets/oscar-corpus/oscar\t{oscar_dir}\t\tOK\tconfig={en_config}; tokens≈{t_en}\n")
            print(f"[OK] OSCAR EN: docs={d_en:,} tokens≈{t_en:,}")
        except Exception as e:
            manifest.append(f"oscar_en_stream\thttps://huggingface.co/datasets/oscar-corpus/oscar\t{oscar_dir}\t\tERR\t{safe_filename(str(e))}\n")
            print(f"[ERR] OSCAR EN: {e}", file=sys.stderr)

    # 3) Write manifest highlight
    manifest.append(f"TOTAL_streamed_tokens\t-\t-\t-\tOK\t≈{total_tokens}\n")
    manifest_path.write_text("".join(manifest), encoding="utf-8")
    print(f"\nDone.\nOutput: {base}\nManifest: {manifest_path}\nStreamed tokens (approx/exact depending on tokenizer): ≈{total_tokens:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
