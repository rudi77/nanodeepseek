#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Download DACH bookkeeping/accounting raw resources into 4 category folders.

C1: Amtliche Fachtexte & Regelwerke (DE/AT/CH)
C2: OER/Praxis-Erklärtexte via Wikimedia Dumps (dewikibooks/dewikiversity + optional dewiki)
C3: Fallbeispiele / Rechnungsbeispiele (Open Source repos)
C4: Open Data / Tabellen / Kontenrahmen (BMF AfA + OSS SKR)

Creates:
  1_fachtexte_regelwerke
  2_praxis_erklaertexte_oer
  3_fallbeispiele_rechnungen
  4_kontenrahmen_tabellen_open_data

Writes MANIFEST.tsv with sha256.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse, unquote

import requests


# -----------------------------
# Categories
# -----------------------------
C1 = "1_fachtexte_regelwerke"
C2 = "2_praxis_erklaertexte_oer"
C3 = "3_fallbeispiele_rechnungen"
C4 = "4_kontenrahmen_tabellen_open_data"
ALL_CATS = [C1, C2, C3, C4]


# -----------------------------
# Helpers
# -----------------------------
@dataclass(frozen=True)
class Resource:
    category: str
    name: str
    url: str
    filename: Optional[str] = None
    kind: str = "file"  # "file" or "html"
    notes: str = ""


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_filename(s: str) -> str:
    s = s.strip().replace("\n", " ")
    s = re.sub(r"[^\w\-.() ]+", "_", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def infer_filename_from_url(url: str) -> str:
    parsed = urlparse(url)
    base = os.path.basename(parsed.path) or "download"
    base = unquote(base)
    if not base or base.endswith("/"):
        base = "download"
    return safe_filename(base)


def infer_filename_from_headers(headers: dict) -> Optional[str]:
    cd = headers.get("Content-Disposition") or headers.get("content-disposition")
    if not cd:
        return None
    m = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', cd, flags=re.IGNORECASE)
    if not m:
        return None
    return safe_filename(unquote(m.group(1)))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def http_get(session: requests.Session, url: str, *, stream: bool, timeout: int = 180) -> requests.Response:
    # Wikimedia and gov sites like a descriptive UA.
    headers = {
        "User-Agent": "dach-accounting-downloader/2.0 (contact: rudi.dittrich77@gmail.com)",
        "Accept": "*/*",
    }
    r = session.get(url, headers=headers, stream=stream, timeout=timeout, allow_redirects=True)
    r.raise_for_status()
    return r


def download_file(
    session: requests.Session,
    url: str,
    dest_dir: Path,
    filename: Optional[str] = None,
    *,
    overwrite: bool = False,
    retries: int = 3,
    sleep_s: float = 0.5,
) -> Tuple[Path, str]:
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            r = http_get(session, url, stream=True)
            inferred = infer_filename_from_headers(r.headers) or infer_filename_from_url(url)
            out_name = filename or inferred
            out_path = dest_dir / out_name

            if out_path.exists() and not overwrite:
                return out_path, sha256_file(out_path)

            tmp_path = out_path.with_suffix(out_path.suffix + ".part")
            with tmp_path.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

            tmp_path.replace(out_path)
            digest = sha256_file(out_path)
            time.sleep(sleep_s)
            return out_path, digest
        except Exception as e:
            last_err = e
            time.sleep(min(10.0, sleep_s * attempt))
    raise RuntimeError(f"Failed after {retries} retries: {url} :: {last_err}")


def download_html(
    session: requests.Session,
    url: str,
    dest_dir: Path,
    filename: str,
    *,
    overwrite: bool = False,
    retries: int = 3,
    sleep_s: float = 0.5,
) -> Tuple[Path, str]:
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            r = http_get(session, url, stream=False)
            out_path = dest_dir / filename
            if out_path.exists() and not overwrite:
                return out_path, sha256_file(out_path)
            write_text(out_path, r.text)
            digest = sha256_file(out_path)
            time.sleep(sleep_s)
            return out_path, digest
        except Exception as e:
            last_err = e
            time.sleep(min(10.0, sleep_s * attempt))
    raise RuntimeError(f"Failed after {retries} retries: {url} :: {last_err}")


def build_readme(base: Path) -> None:
    txt = [
        "DACH Accounting raw corpus download\n",
        "Folders:\n",
        f"  {C1}: Fachtexte & Regelwerke (amtlich) – DE/AT/CH\n",
        f"  {C2}: Praxis-Erklärtexte (OER) – Wikimedia Dumps (dewikibooks/dewikiversity [+ optional dewiki])\n",
        f"  {C3}: Fallbeispiele/Rechnungsbeispiele (Open Source)\n",
        f"  {C4}: Tabellen/Open Data/Kontenrahmen (Open Data + OSS)\n",
        "\n",
        "Hinweis: 'Downloadbar' != automatisch 'für Training frei'. Bitte Lizenz/Terms prüfen.\n",
    ]
    write_text(base / "README_SOURCES.txt", "".join(txt))


# -----------------------------
# Resource list (DE+AT+CH)
# -----------------------------
RESOURCES: list[Resource] = [
    # =========================
    # C1: Amtliche Fachtexte & Regelwerke
    # =========================
    # DE: Gesetze-im-Internet (PDF)
    Resource(C1, "DE_UStG (Gesetze-im-Internet) PDF", "https://www.gesetze-im-internet.de/ustg_1980/UStG.pdf"),
    Resource(C1, "DE_EStG (Gesetze-im-Internet) PDF", "https://www.gesetze-im-internet.de/estg/EStG.pdf"),
    Resource(C1, "DE_AO (Gesetze-im-Internet) PDF", "https://www.gesetze-im-internet.de/ao_1977/AO.pdf"),
    Resource(C1, "DE_HGB (Gesetze-im-Internet) PDF", "https://www.gesetze-im-internet.de/hgb/HGB.pdf"),

    # DE: BMF
    Resource(C1, "DE_BMF_GoBD Änderungsschreiben PDF",
             "https://www.bundesfinanzministerium.de/Content/DE/Downloads/BMF_Schreiben/Weitere_Steuerthemen/Abgabenordnung/AO-Anwendungserlass/2024-03-11-aenderung-gobd.pdf?__blob=publicationFile&v=4",
             filename="DE_BMF_GoBD_Aenderung_2024-03-11.pdf"),
    Resource(C1, "DE_BMF_UStAE aktuell PDF",
             "https://www.bundesfinanzministerium.de/Content/DE/Downloads/BMF_Schreiben/Steuerarten/Umsatzsteuer/Umsatzsteuer-Anwendungserlass/Umsatzsteuer-Anwendungserlass-aktuell.pdf?__blob=publicationFile&v=23",
             filename="DE_BMF_UStAE_aktuell.pdf"),

    # AT: RIS (PDF)
    Resource(C1, "AT_RIS_UStG1994 PDF",
             "https://www.ris.bka.gv.at/geltendefassung/bundesnormen/10004873/ustg%201994%2C%20fassung%20vom%2030.07.2021.pdf",
             filename="AT_RIS_UStG1994.pdf"),
    Resource(C1, "AT_RIS_EStG1988 PDF",
             "https://www.ris.bka.gv.at/geltendefassung/bundesnormen/10004570/estg%201988%2C%20fassung%20vom%2016.05.2021.pdf",
             filename="AT_RIS_EStG1988.pdf"),
    Resource(C1, "AT_RIS_BAO PDF",
             "https://www.ris.bka.gv.at/geltendefassung/bundesnormen/10003940/bao%2C%20fassung%20vom%2016.06.2021.pdf",
             filename="AT_RIS_BAO.pdf"),
    Resource(C1, "AT_RIS_UGB PDF",
             "https://www.ris.bka.gv.at/geltendefassung/bundesnormen/10001702/ugb%2C%20fassung%20vom%2027.06.2021.pdf",
             filename="AT_RIS_UGB.pdf"),

    # CH: ESTV + Fedlex
    Resource(C1, "CH_ESTV_MWSTG PDF",
             "https://www.estv.admin.ch/dam/estv/de/dokumente/estv/steuerpolitik/21019-mwstg-mit-aenderungen-de.pdf.download.pdf/21019-mwstg-mit-aenderungen-de.pdf",
             filename="CH_ESTV_MWSTG.pdf"),
    Resource(C1, "CH_FEDLEX_OR_ELI HTML",
             "https://www.fedlex.admin.ch/eli/cc/27/317_321_377/de",
             filename="CH_FEDLEX_OR_ELI.html",
             kind="html"),

    # =========================
    # C2: OER Praxis-Erklärtexte (Wikimedia Dumps)
    # =========================
    # Dewikibooks / Dewikiversity dumps (latest)
    Resource(C2, "Wikibooks (DE) latest pages-articles dump",
             "https://dumps.wikimedia.org/dewikibooks/latest/dewikibooks-latest-pages-articles.xml.bz2",
             filename="dewikibooks-latest-pages-articles.xml.bz2",
             notes="CC BY-SA content; wikitext XML dump"),
    Resource(C2, "Wikiversity (DE) latest pages-articles dump",
             "https://dumps.wikimedia.org/dewikiversity/latest/dewikiversity-latest-pages-articles.xml.bz2",
             filename="dewikiversity-latest-pages-articles.xml.bz2",
             notes="CC BY-SA content; wikitext XML dump"),
    # Optional: German Wikipedia (much larger) – keep it optional by default via CLI flag, see below.

    # =========================
    # C3: Fallbeispiele/Rechnungen (Open Source)
    # =========================
    Resource(C3, "ZUGFeRD corpus (Apache-2.0) ZIP",
             "https://github.com/ZUGFeRD/corpus/archive/refs/heads/master.zip",
             filename="ZUGFeRD_corpus_master.zip"),
    Resource(C3, "EU eInvoicing EN16931 examples ZIP",
             "https://github.com/ConnectingEurope/eInvoicing-EN16931/archive/refs/heads/master.zip",
             filename="EU_eInvoicing_EN16931_master.zip"),

    # =========================
    # C4: Open Data / Tabellen / Kontenrahmen
    # =========================
    Resource(C4, "DE_BMF_AfA_kumuliert CSV",
             "https://www.bundesfinanzministerium.de/Datenportal/Daten/offene-daten/steuern-zoelle/afa-tabellen/datensaetze/AfA-Tabellen_kumuliert_csv.csv?__blob=publicationFile&v=1",
             filename="DE_BMF_AfA_kumuliert.csv"),
    Resource(C4, "DE_BMF_AfA_kumuliert XLSX",
             "https://www.bundesfinanzministerium.de/Datenportal/Daten/offene-daten/steuern-zoelle/afa-tabellen/datensaetze/AfA-Tabellen_kumuliert_xlsx.xlsx?__blob=publicationFile&v=1",
             filename="DE_BMF_AfA_kumuliert.xlsx"),

    # OSS SKR dataset (@fin.cx/skr)
    Resource(C4, "fin.cx SKR index.min.js",
             "https://cdn.jsdelivr.net/npm/@fin.cx/skr@1.2.1/dist_ts/index.min.js",
             filename="fin_cx_skr_index.min.js"),
    Resource(C4, "fin.cx SKR LICENSE",
             "https://cdn.jsdelivr.net/npm/@fin.cx/skr@1.2.1/license",
             filename="fin_cx_skr_LICENSE.txt"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../data/dach_accounting_resources_raw", help="Output directory")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    ap.add_argument("--sleep", type=float, default=0.5, help="Sleep seconds between downloads")
    ap.add_argument("--retries", type=int, default=3, help="Download retries")
    ap.add_argument("--include-dewiki", action="store_true",
                    help="Also download German Wikipedia dump into C2 (very large)")
    args = ap.parse_args()

    base = Path(args.out).resolve()
    ensure_dir(base)
    for cat in ALL_CATS:
        ensure_dir(base / cat)
    build_readme(base)

    session = requests.Session()

    # Conditionally add German Wikipedia dump (huge but useful)
    resources = list(RESOURCES)
    if args.include_dewiki:
        resources.append(
            Resource(C2, "Wikipedia (DE) latest pages-articles dump",
                     "https://dumps.wikimedia.org/dewiki/latest/dewiki-latest-pages-articles.xml.bz2",
                     filename="dewiki-latest-pages-articles.xml.bz2",
                     notes="CC BY-SA content; very large wikitext XML dump")
        )

    manifest = ["name\tcategory\turl\tlocal_file\tsha256\tstatus\tnotes\n"]

    for res in resources:
        dest = base / res.category
        try:
            if res.kind == "html":
                p, digest = download_html(
                    session, res.url, dest, res.filename or (infer_filename_from_url(res.url) + ".html"),
                    overwrite=args.overwrite, retries=args.retries, sleep_s=args.sleep
                )
            else:
                p, digest = download_file(
                    session, res.url, dest, filename=res.filename,
                    overwrite=args.overwrite, retries=args.retries, sleep_s=args.sleep
                )
            manifest.append(f"{res.name}\t{res.category}\t{res.url}\t{p.name}\t{digest}\tOK\t{res.notes}\n")
            print(f"[OK] {res.category} :: {res.name} -> {p.name}")
        except Exception as e:
            manifest.append(f"{res.name}\t{res.category}\t{res.url}\t\t\tERR\t{safe_filename(str(e))}\n")
            print(f"[ERR] {res.category} :: {res.name}\n      {res.url}\n      {e}", file=sys.stderr)

    write_text(base / "MANIFEST.tsv", "".join(manifest))
    print(f"\nDone.\nOutput: {base}\nManifest: {base / 'MANIFEST.tsv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
